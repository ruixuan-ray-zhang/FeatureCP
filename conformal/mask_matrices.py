import numpy as np

def split_into_blocks(matrix, a, b):
    """
    Splits the matrix into blocks of size (2, h/a, w/b).

    Args:
    matrix (numpy.ndarray): The input matrix of shape (2, h, w).
    a (int): The number of vertical blocks.
    b (int): The number of horizontal blocks.

    Returns:
    list of numpy.ndarray: List of blocks.
    """
    _, h, w = matrix.shape
    if h % a != 0 or w % b != 0:
        raise ValueError("The dimensions of the matrix are not divisible by a and b respectively.")
    
    block_h = h // a
    block_w = w // b
    blocks = []
    
    for i in range(a):
        for j in range(b):
            start_h = i * block_h
            end_h = start_h + block_h
            start_w = j * block_w
            end_w = start_w + block_w
            block = matrix[:, start_h:end_h, start_w:end_w]
            blocks.append(block)
            
    return blocks

def pad_block_to_original(block, original_shape, block_position, a, b):
    """
    Pads the given block to the original matrix size, placing the block in the correct location.

    Args:
    block (numpy.ndarray): The block to pad.
    original_shape (tuple): The shape of the original matrix (2, h, w).
    block_position (tuple): The position (i, j) of the block in the grid.
    a (int): The number of vertical blocks.
    b (int): The number of horizontal blocks.

    Returns:
    numpy.ndarray: The padded matrix.
    """
    padded_matrix = np.zeros(original_shape, dtype=block.dtype)
    i, j = block_position
    block_h = original_shape[1] // a
    block_w = original_shape[2] // b
    start_h = i * block_h
    start_w = j * block_w
    padded_matrix[:, start_h:start_h + block_h, start_w:start_w + block_w] = block
    
    return padded_matrix

def main(matrix, a, b):
    blocks = split_into_blocks(matrix, a, b)
    padded_matrices = []
    # Pad each block back to the original size and print
    for idx, block in enumerate(blocks):
        block_position = (idx // b, idx % b)  # Calculate block position from index
        padded_matrix = pad_block_to_original(block, matrix.shape, block_position, a, b)
        padded_matrices.append(padded_matrix)
    return padded_matrices

if __name__ == "__main__":
    # Example usage
    sample_matrix = np.random.randint(10, size=(2, 9, 6))
    a, b = 3, 3
    padded_matrices = main(sample_matrix, a, b)
    print("Original matrix:")
    print(sample_matrix)
    print("Padded matrices:")
    for idx, padded_matrix in enumerate(padded_matrices):
        print(f"Padded matrix {idx+1}:\n{padded_matrix}\n")