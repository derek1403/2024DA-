import numpy as np


def observation_operator(The_Number_of_small_grid=512 , The_Number_of_big_grid=64) :
  print('observation_operator')
  grid512 , grid64 = The_Number_of_small_grid , The_Number_of_big_grid # 512 , 64
  div = int(grid512/grid64) # 8
  H = np.zeros((grid64**2,grid512**2)) # np.zeros((4096,262144))
  
  dotime = 0
  for xi in range(grid512):
      for yi in range(grid512):
          A = np.zeros((grid64,grid64))
          A[int(yi/div),int(xi/div)] = 1
          H[:,dotime] = A.reshape(-1)
          dotime += 1
      #print(xi)
  print('observation_operator done')
  return H


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  H = observation_operator()
  # Visualize a portion of the full H (e.g. the first 128 rows and 128 columns)
  H_subset = H[:256, :256]
  
  plt.figure(figsize=(10, 10))
  plt.imshow(H_subset, cmap="viridis", aspect="auto")
  plt.colorbar(label="H values")
  plt.title("Subset of H Matrix Visualization")
  plt.xlabel("Small Grid Index")
  plt.ylabel("Large Grid Index")
  plt.show()
  
  # To aggregate and visualize the entire H, you can use block-wise sum
  H_aggregated = H.reshape(64, 64, 512, 512).sum(axis=(1, 3))
  
  plt.figure(figsize=(10, 10))
  plt.imshow(H_aggregated, cmap="viridis", aspect="auto")
  plt.colorbar(label="Aggregated H values")
  plt.title("Aggregated H Matrix Visualization")
  plt.xlabel("Small Grid Blocks")
  plt.ylabel("Large Grid Blocks")
  plt.show()
  
  
  ##testing
  G = np.array([[1,2],[3,4],[5,6]])
  G1=G.reshape(-1)
  print(G1)
  print(G[0,1])
  plt.figure(figsize=(10, 10))
  plt.imshow(G)
  plt.colorbar(label="Aggregated G values")
  plt.show()
