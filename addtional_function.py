import numpy as np

def downsample_to_average(matrix, new_shape):
    """
    Downsample a 2D matrix to a smaller shape by averaging non-overlapping blocks.

    Parameters:
        matrix (numpy.ndarray): The original 2D matrix (e.g., 512x512).
        new_shape (tuple): The desired shape (e.g., (64, 64)).

    Returns:
        numpy.ndarray: The downsampled matrix with the desired shape.
    """
    old_shape = matrix.shape
    if old_shape[0] % new_shape[0] != 0 or old_shape[1] % new_shape[1] != 0:
        raise ValueError("The original matrix dimensions must be divisible by the new shape dimensions.")
    
    block_size = (old_shape[0] // new_shape[0], old_shape[1] // new_shape[1])
    
    # Reshape and calculate block means
    return matrix.reshape(new_shape[0], block_size[0], new_shape[1], block_size[1]).mean(axis=(1, 3))



def upsample_matrix(matrix, new_shape, method='nearest'):
    """
    Upsample a 2D matrix to a larger shape using interpolation.

    Parameters:
        matrix (numpy.ndarray): The input 2D matrix (e.g., 64x64).
        new_shape (tuple): The desired shape (e.g., (512, 512)).
        method (str): Interpolation method. Options are 'nearest' or 'linear'.

    Returns:
        numpy.ndarray: The upsampled matrix with the desired shape.
    """
    from scipy.ndimage import zoom
    
    if method not in ['nearest', 'linear']:
        raise ValueError("Method must be 'nearest' or 'linear'")
    
    scale_factors = (new_shape[0] / matrix.shape[0], new_shape[1] / matrix.shape[1])
    
    if method == 'nearest':
        return zoom(matrix, scale_factors, order=0)  # Nearest-neighbor interpolation
    elif method == 'linear':
        return zoom(matrix, scale_factors, order=1)  # Linear interpolation

def downsample_to_average_3d(matrix, new_shape):
    """
    Downsample a 3D matrix to a smaller shape by averaging non-overlapping blocks.

    Parameters:
        matrix (numpy.ndarray): The original 3D matrix (e.g., (n, 512, 512)).
        new_shape (tuple): The desired 2D shape for each 2D slice (e.g., (64, 64)).

    Returns:
        numpy.ndarray: The downsampled 3D matrix with shape (n, new_shape[0], new_shape[1]).
    """
    if len(matrix.shape) != 3:
        raise ValueError("Input matrix must be 3D with shape (n, height, width).")
    
    n, old_height, old_width = matrix.shape
    if old_height % new_shape[0] != 0 or old_width % new_shape[1] != 0:
        raise ValueError("The height and width of the matrix must be divisible by the new shape dimensions.")
    
    block_size = (old_height // new_shape[0], old_width // new_shape[1])
    
    # Reshape each 2D slice and calculate block means
    return matrix.reshape(n, new_shape[0], block_size[0], new_shape[1], block_size[1]).mean(axis=(2, 4))


if __name__ == '__main__':
  # Example usage1:
  original_matrix = np.random.rand(512, 512)  # Example 512x512 matrix
  new_shape = (64, 64)
  downsampled_matrix = downsample_to_average(original_matrix, new_shape)
  
  print("Original Matrix Shape:", original_matrix.shape)
  print("Downsampled Matrix Shape:", downsampled_matrix.shape)
  
  
  # Example usage2:
  original_matrix = np.random.rand(512, 512)
  new_shape = (64, 64)
  
  # Downsample
  downsampled_matrix = downsample_to_average(original_matrix, new_shape)
  
  # Upsample
  upsampled_matrix = upsample_matrix(downsampled_matrix, original_matrix.shape, method='linear')
  
  # Compare matrices
  print("Original Matrix Shape:", original_matrix.shape)
  print("Downsampled Matrix Shape:", downsampled_matrix.shape)
  print("Upsampled Matrix Shape:", upsampled_matrix.shape)
  
  print(np.mean(original_matrix))
  print(np.mean(upsampled_matrix-original_matrix))
  
  
  # Example usage3:
  original_matrix = np.random.rand(10, 512, 512)  # Example 3D matrix with 10 slices
  new_shape = (64, 64)
  
  downsampled_matrix = downsample_to_average_3d(original_matrix, new_shape)
  
  print("Original Matrix Shape:", original_matrix.shape)
  print("Downsampled Matrix Shape:", downsampled_matrix.shape)
  








