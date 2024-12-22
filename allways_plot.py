
import matplotlib.pyplot as plt
import numpy as np

def plot_matrices(matrices, colormap='viridis'):
    """
    Plots multiple 2D matrices as subplots with a shared colorbar.

    Parameters:
        matrices (list of np.ndarray): List of 2D matrices to plot.
        colormap (str): Colormap to use for the plots.
    """
    n = len(matrices)
    if n == 0:
        print("No matrices to plot.")
        return

    # Determine subplot layout (rows x cols)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()  # Flatten in case of 2D axes array

    # Determine global min and max for colorbar
    vmin = min(mat.min() for mat in matrices)
    vmax = max(mat.max() for mat in matrices)

    # Plot each matrix
    for i, mat in enumerate(matrices):
        ax = axes[i]
        cax = ax.imshow(mat, cmap=colormap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Matrix {i+1}")
        ax.axis('off')  # Turn off axis ticks

    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add a shared colorbar
    plt.colorbar(cax, ax=axes[:n], orientation='vertical', shrink=0.8)

    # Show the plot

    plt.show()


import numpy as np

def localization(Pf, L):
    """
    Applies localization to the matrix Pf based on the provided distance scale L.

    Parameters:
        Pf (np.ndarray): The input covariance matrix (2D square matrix).
        L (float): The localization distance scale.

    Returns:
        np.ndarray: The localized covariance matrix.
    """
    # Ensure Pf is a square matrix
    if Pf.shape[0] != Pf.shape[1]:
        raise ValueError("Pf must be a square matrix.")
    
    n = Pf.shape[0]  # Dimension of the square matrix
    C_Bloc = np.zeros_like(Pf)  # Initialize the localization matrix
    
    # Compute the localization function f_Bloc(i, j) for each element
    for i in range(n):
        for j in range(n):
            d = abs(i - j)  # Define the distance metric (can adjust for 2D or spatial grids)
            C_Bloc[i, j] = np.exp(- (d ** 2) / (2 * L ** 2))
    
    # Apply the localization to Pf
    Pf_localized = C_Bloc * Pf  # Element-wise multiplication
    
    return Pf_localized

# Example usage
if __name__ == "__main__":
    # Create an example covariance matrix (random)
    Pf = np.random.rand(5, 5)  # 5x5 covariance matrix
    Pf = (Pf + Pf.T) / 2  # Symmetrize it to make it realistic
    
    # Apply localization with a given L
    L = 2.0
    Pf_localized = localization(Pf, L)
    
    print("Original Pf:")
    print(Pf)
    print("\nLocalized Pf:")
    print(Pf_localized)



# Example usage
if __name__ == "__main__":
    # Generate example 2D matrices
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10) * 2
    C = np.random.rand(10, 10) - 0.5
    matrices = [A, B, C]

    # Plot the matrices
    plot_matrices(matrices)

