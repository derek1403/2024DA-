This experiment aims to simulate vortex dynamics using the vorticity equation and an initial condition field, and to explore the relationship between vortex dynamics and deep learning models.

First, we apply the vorticity equation to simulate vortex evolution, starting from an initial vorticity field. The grid size is set to N=640, with a total of num_steps = 5000 time steps and a time step interval of dt=0.01 seconds for stability. By utilizing the Fast Fourier Transform (FFT), we compute the vorticity distribution at each time step and decompose the system's frequency components. This allows us to track the evolution of the vorticity field over time.

Next, we employ a U-Net model, where the central 512?512 region of the vorticity field at time t_n is used as the input, and the vorticity field at a future time t_{n+k} is used as the output. The time interval k between t_n and t_{n+k} can be adjusted, allowing us to flexibly determine how far ahead the U-Net model should forecast. This approach enables us to investigate whether U-Net can effectively capture the nonlinear behavior and complexity of vortex dynamics.

The objective of this experiment is to verify whether U-Net can successfully learn the key features of vortex dynamics and maintain predictive accuracy over multiple time steps. This study not only enhances our understanding of vortex dynamics but also helps establish the effectiveness of deep learning models in forecasting the evolution of vorticity fields.

Additionally, using U-Net for prediction offers the benefit of reducing computational costs. Traditional numerical simulations require significant computational resources, especially for high-resolution and long-duration simulations. In contrast, once trained, the U-Net model can quickly predict future states, significantly improving computational efficiency and reducing runtime.


# Rotation Description

This document describes the categories of `rotation` and their behavior in various scenarios, divided into two main types: `normal` and `special`. Each type includes specific examples with detailed explanations.

## (a) Normal

### 1. Left to right, positive-negative-positive-negative, colume ellipse
- **Explanation**: There are 4 colume ellipses, where the vorticity values from left to right follow the sequence of positive, negative, positive, and negative.
- Timestep: 5000

### 2. Two positive circles, one strong and one weak, close distance
- **Explanation**: There are 2 circles with positive vorticity values. One circle has a higher positive value than the other, and they are located close to each other.
- Timestep: 5000

### 3. Two positive circles, one strong and one weak, far distance
- **Explanation**: There are 2 circles with positive vorticity values. One circle has a higher positive value than the other, but they are far apart from each other.
- Timestep: 5000

### 4. Two positive circles, similar strength, close distance
- **Explanation**: Two circles with similar positive vorticity values are positioned close to each other.
- Timestep: 5000

### 5. Two positive circles, similar strength, far distance
- **Explanation**: Two circles with similar positive vorticity values are positioned far apart from each other.
- Timestep: 5000

### 6. Multiple positive circles
- **Explanation**: Multiple circles with positive vorticity values are present, varying in size and strength.
- Timestep: 5000

### 7. Multiple tilted ellipses aligned in a strip
- **Explanation**: Several tilted ellipses are arranged in a row, forming a strip-like pattern.
- Timestep: 2250

### 8. Two negative circles, one strong and one weak, far distance
- **Explanation**: Two circles with negative vorticity values are present. One has a stronger negative value than the other, and they are far apart from each other.
- Timestep: 5000

### 9. Two negative circles, similar strength, close distance
- **Explanation**: Two circles with similar negative vorticity values are positioned close to each other.
- Timestep: 5000

## (b) Special

### 10. Fractal shape
- **Explanation**: The fractal shape's specific parameters can be found in the code file **rotation.py**.
- Timestep: 5000

### 11. Fractal shape complex - after diffusion
- **Explanation**: A complex of fractal shapes where negative values are embedded within the positive fractal structure. After diffusion, the values are balanced and no longer too extreme.
- Timestep: 5000

### 12. Tai Chi shape[extreme] - after diffusion
- **Explanation**: A Tai Chi shape with positive and negative vorticity values, processed through diffusion to create smoother transitions between regions.
- Timestep: 2000

### 13. Tai Chi shape[resemblance] - after diffusion
- **Explanation**: A Tai Chi shape with positive vorticity values , processed through diffusion to create smoother transitions between regions.
- Timestep: 5000



#####
Note:
When using Fast Fourier Transform (FFT) for numerical simulations, one may encounter limitations related to the discretization of the frequency domain. As the grid resolution increases, the computational load rises significantly, which can lead to numerical instability or aliasing if not properly managed. Additionally, when employing simplified vorticity equations, there is a risk of oversimplification that can overlook critical dynamics in the fluid flow. These simplifications may fail to capture important interactions between vortices, resulting in inaccuracies in the simulation outcomes. Consequently, both FFT and simplified vorticity equations must be carefully calibrated to ensure stable and reliable results.



######
For the detailed changelog, see [Changelog_rotation.md](./Changelog_rotation.md).







