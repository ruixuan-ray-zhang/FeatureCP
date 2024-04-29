import torch
import torch.nn as nn
from torch.autograd import Variable

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, bn=False):
        super(Encoder, self).__init__()
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)

        return x

class Head(nn.Module):
    def __init__(self, bn=False):
        super(Head, self).__init__()
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))

    def forward(self, x):
        return self.fuse(x)

class MCNN(nn.Module):
    def __init__(self, encoder, head):
        super(MCNN, self).__init__()
        self.encoder = encoder
        self.g = head

    def forward(self, im_data):
        features = self.encoder(im_data)
        output = self.g(features)

        return output
    
def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False, volatile = True)
    if is_cuda:
        v = v.cuda()
    return v

class CrowdCounter(nn.Module):
    def __init__(self, encoder, head, is_training):
        super(CrowdCounter, self).__init__()        
        self.DME = MCNN(encoder, head)        
        self.loss_fn = nn.MSELoss()
        self.is_training = is_training
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self,  im_data, gt_data=None):        
        im_data = np_to_variable(im_data, is_cuda=True, is_training=self.is_training)                
        density_map = self.DME(im_data)
        
        if self.is_training:                        
            gt_data = np_to_variable(gt_data, is_cuda=True, is_training=self.is_training)            
            self.loss_mse = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss 

def make_mcnn():
    encoder = Encoder()
    head = Head()
    is_training = False
    model = CrowdCounter(encoder, head, is_training)
    model.cuda()

    return model


if __name__ == "__main__":
    model = make_mcnn()

    import h5py
    import numpy as np
    fname = 'mcnn_shtechB_1356_ruixuan(mae19.1_mse32.8).h5'
    h5f = h5py.File(fname, mode='r')
    for k, v in model.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[k]))         
        v.copy_(param)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

