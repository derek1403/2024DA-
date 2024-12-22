import numpy as np

def add_gaussian_noise(matrix, mu=0, std=1):
    """
    Add Gaussian noise to a 2D matrix.

    Parameters:
        matrix (numpy.ndarray): The original 2D matrix.
        mu (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: The matrix with added noise.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix must be a numpy ndarray.")
    noise = np.random.normal(mu, std, matrix.shape)
    return matrix + noise


def generate_random_ABC():
    """
    Generate random values A, B, C such that:
        A + B + C = 1
        A > 0.6
        A, B, C >= 0

    Returns:
        tuple: A, B, C
    """
    while True:
        A = np.random.uniform(0.6, 1)  # Ensure A > 0.6
        remaining = 1 - A
        B = np.random.uniform(0, remaining)  # Ensure B >= 0 and B <= remaining
        C = remaining - B  # Ensure C >= 0
        if C >= 0:  # Check the constraint
            return A, B, C




if __name__ == '__main__':
  # Example usage 1 :
  original_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  mu = 0
  std = 0.1
  noisy_matrix = add_gaussian_noise(original_matrix, mu, std)
  
  print("Original Matrix:")
  print(original_matrix)
  print("\nNoisy Matrix:")
  print(noisy_matrix)

  # Example usage 2 :
  A, B, C = generate_random_ABC()
  print(f"A = {A:.3f}, B = {B:.3f}, C = {C:.3f}")
  print(A+B+C)

