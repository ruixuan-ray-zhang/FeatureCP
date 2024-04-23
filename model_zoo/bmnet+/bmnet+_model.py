import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

import copy 
import sys

"""
   ************************************
   Vision backbone
    ************************************ 
"""

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layer: str):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #if not train_backbone:
                parameter.requires_grad_(False)
        
        return_layers = {return_layer: '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        """supports both NestedTensor and torch.Tensor
        """
        out = self.body(tensor_list)
        return out['0']

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_layer: str,
                 frozen_bn: bool,
                 dilation: bool):
        
        if frozen_bn:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True)
            
        # load the SwAV pre-training model from the url instead of supervised pre-training model
        if name == 'resnet50':
            checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(state_dict, strict=False)
            #pass
        if name in ('resnet18', 'resnet34'):
            num_channels = 512
        else:
            if return_layer == 'layer3':
                num_channels = 1024
            else:
                num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_layer)


"""
    ************************************
    exemplar featre extractor
    ************************************
"""
class DirectPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, repeat_times=1, use_scale_embedding=True, scale_number=20):
        super().__init__()
        self.repeat_times = repeat_times
        self.use_scale_embedding = use_scale_embedding
        self.patch2query = nn.Linear(input_dim, hidden_dim) # align the patch feature dim to query patch dim.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
        if self.use_scale_embedding:
            self.scale_embedding = nn.Embedding(scale_number, hidden_dim)
    
    def forward(self, patch_feature, scale_index):
        bs, batch_num_patches = scale_index.shape
        patch_feature = self.avgpool(patch_feature).flatten(1) # bs X patchnumber X feature_dim

        patch_feature = self.patch2query(patch_feature) \
            .view(bs, batch_num_patches, -1) \
            .repeat_interleave(self.repeat_times, dim=1) \
            .permute(1, 0, 2) \
            .contiguous() 
        
        if self.use_scale_embedding:
            scale_embedding = self.scale_embedding(scale_index) # bs X number_query X dim
            patch_feature = patch_feature + scale_embedding.permute(1, 0, 2)
        
        return patch_feature

"""
    ************************************
    refiner
    ************************************
"""
class SelfSimilarityModule(nn.Module):
    def __init__(self, hidden_dim, proj_dim, layer_number):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(SelfSimilarityLayer(hidden_dim=hidden_dim, proj_dim=proj_dim)) for i in range(layer_number)])
    def forward(self, features, patches):
        for layer in self.layers:
            features, patches = layer(features, patches)
        return features, patches

'''
Layer in self similarity module
'''
class SelfSimilarityLayer(nn.Module):
    def __init__(self, hidden_dim, proj_dim, dropout_rate=0.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        
        self.query_conv = nn.Linear(hidden_dim, proj_dim)
        self.key_conv = nn.Linear(hidden_dim, proj_dim)
        self.value_conv = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.post_conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                         nn.ReLU())
        
        self.softmax  = nn.Softmax(dim=-1)
            
    
    def forward(self, features, patches):
        """
            inputs :
                x : input feature maps (B X C X W X H)
                patches: feature vectors of exemplar patches (query_number X B X C)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = features.size()
        query_number = patches.shape[0]
        features = features.view(m_batchsize, -1, width*height).permute(0, 2, 1) # B X N X C
        appended_features = torch.cat((features, patches.permute(1, 0, 2)), dim=1) # B X (N+query_number) X C
        proj_value = self.value_conv(features).view(m_batchsize,-1,width*height) # B X C X (N+query_number)
        
        proj_query = self.query_conv(appended_features)
        proj_key = self.key_conv(appended_features).permute(0, 2, 1) # B X C X (N + query_number)
        proj_value = self.value_conv(appended_features) # B X (N+query_number) X C
        
        energy =  torch.bmm(proj_query, proj_key) # B X (N+query_number) X (N+query_number)
        attention = self.softmax(energy) 

        out = torch.bmm(proj_value.permute(0, 2, 1), attention.permute(0,2,1)) # B X C X (N+query_number)
        out = self.gamma * self.dropout(out) + appended_features.permute(0,2,1)
        #out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1) # shape of B X (N+query_number) X dim
        #self.out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1) # shape of B X (N+query_number) X dim
        
        out_feat, out_patch = out[:, :, :-1*query_number], out[:,:,-1*query_number:]
        out_feat = out_feat.reshape(m_batchsize, C, width, height) # B X C X H X W
        out_patch = out_patch.permute(2, 0, 1) # query_number * B * dim
        
        return self.post_conv(out_feat), out_patch

"""
    ************************************
    matcher
    ************************************
"""
class DynamicSimilarityMatcher(nn.Module):
    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim, activation='tanh', pool='mean', use_bias=False):
        super().__init__()
        self.query_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.key_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.dynamic_pattern_conv = nn.Sequential(nn.Linear(in_features=proj_dim, out_features=dynamic_proj_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=dynamic_proj_dim, out_features=proj_dim))
        
        self.softmax  = nn.Softmax(dim=-1)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise NotImplementedError
            
    def forward(self, features, patches):
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # hw * bs * dim
        
        proj_feat = self.query_conv(features)
        patches_feat = self.key_conv(patches)
        patches_ca = self.activation(self.dynamic_pattern_conv(patches_feat))
        
        proj_feat = proj_feat.permute(1, 0, 2)
        patches_feat = (patches_feat * (patches_ca + 1)).permute(1, 2, 0)  # bs * c * exemplar_number        
        energy = torch.bmm(proj_feat, patches_feat)                        # bs * hw * exemplar_number

        corr = energy.mean(dim=-1, keepdim=True)
        out = features.permute(1,0,2)  # hw * bs * c
        out = torch.cat((out, corr), dim=-1)
        
        out = out.permute(1,0,2)
        return out.permute(1, 2, 0).view(bs, c+1, h, w), energy
    
"""
    ************************************
    regression-based counter
    ************************************
"""
class DensityX16(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(counter_dim, 196, 7, padding=3)
        self.conv2 = nn.Conv2d(196, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 1)
        self.conv5 = nn.Conv2d(32, 1, 1)
        
    def forward(self, features):

        x = self.conv1(features)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv5(x)
        x = F.relu(x)

        return x
    
class BMNet(nn.Module):
    def __init__(self, backbone, EPF_extractor, refiner, matcher, counter, hidden_dim):

        super().__init__()
        self.EPF_extractor = EPF_extractor
        self.refiner = refiner
        self.matcher = matcher
        self.counter = counter

        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def encoder(self, samples: torch.Tensor, patches: torch.Tensor, is_train: bool):
        # Stage 1: extract features for query images and exemplars
        scale_embedding, patches = patches['scale_embedding'], patches['patches']
        features = self.backbone(samples)
        features = self.input_proj(features)
        
        patches = patches.flatten(0, 1) 
        patch_feature = self.backbone(patches) # obtain feature maps for exemplar patches
        patch_feature = self.EPF_extractor(patch_feature, scale_embedding) # compress the feature maps into vectors and inject scale embeddings
        
        # Stage 2: enhance feature representation, e.g., the self similarity module.
        refined_feature, patch_feature = self.refiner(features, patch_feature)
        # Stage 3: generate similarity map by densely measuring similarity. 
        counting_feature, corr_map = self.matcher(refined_feature, patch_feature)

        return counting_feature

    def g(self, counting_feature):
        return self.counter(counting_feature)
        
    def forward(self, samples: torch.Tensor, patches: torch.Tensor, is_train: bool):
        """ The forward expects samples containing query images and corresponding exemplar patches.
            samples is a stack of query images, of shape [batch_size X 3 X H X W]
            patches is a torch Tensor, of shape [batch_size x num_patches x 3 x 128 x 128]
            The size of patches are small than samples

            It returns a dict with the following elements:
               - "density_map": Shape= [batch_size x 1 X h_query X w_query]
               - "patch_feature": Features vectors for exemplars, not available during testing.
                                  They are used to compute similarity loss. 
                                Shape= [exemplar_number x bs X hidden_dim]
               - "img_feature": Feature maps for query images, not available during testing.
                                Shape= [batch_size x hidden_dim X h_query X w_query]
            
        """ 
        counting_feature = self.encoder(samples, patches, is_train)
        # Stage 4: predicting density map 
        density_map = self.g(counting_feature)
        
        if not is_train:
            return density_map
        else:
            raise NotImplementedError("It shouldn't be in train mode.")

def make_bmnet(cfg):
    # vision backbone
    backbone = Backbone(cfg.MODEL.backbone, False, cfg.MODEL.backbone_layer, cfg.MODEL.fix_bn, cfg.MODEL.dilation)
    # exemplar feature extractor
    input_dim = 1024 if cfg.MODEL.backbone_layer == 'layer3' else 2048
    epf_extractor = DirectPooling(input_dim=input_dim,
                             hidden_dim=cfg.MODEL.hidden_dim,
                             repeat_times=cfg.MODEL.repeat_times,
                             use_scale_embedding=cfg.MODEL.ep_scale_embedding,
                             scale_number=cfg.MODEL.ep_scale_number)
    # refiner
    refiner = SelfSimilarityModule(hidden_dim=cfg.MODEL.hidden_dim,
                                    proj_dim=cfg.MODEL.refiner_proj_dim,
                                    layer_number=cfg.MODEL.refiner_layers)
    # matcher
    matcher = DynamicSimilarityMatcher(hidden_dim=cfg.MODEL.hidden_dim,
                                        proj_dim=cfg.MODEL.matcher_proj_dim,
                                        dynamic_proj_dim=cfg.MODEL.dynamic_proj_dim,
                                        use_bias=cfg.MODEL.use_bias)
    # counter
    counter = DensityX16(counter_dim=cfg.MODEL.counter_dim)

    model = BMNet(backbone, epf_extractor, refiner, matcher, counter, cfg.MODEL.hidden_dim)

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)
    sys.path.append(parent_path)

    if os.path.isfile(cfg.VAL.resume):
            checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
    else:
        print('model state dict not found.')

    return model


if __name__ == '__main__':
    import os
    import argparse
    from config import cfg

    parser = argparse.ArgumentParser(
        description="BMNet+"
    )
    parser.add_argument(
        "--cfg",
        default="config/bmnet+_fsc147.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.VAL.resume = "bmnet+_resnet_fsc147.pth"

    model = make_bmnet(cfg)