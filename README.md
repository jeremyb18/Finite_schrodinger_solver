# PyWave: 1D Quantum Wave Function Solver

PyWave is a Python package for simulating and visualizing the time evolution of a one-dimensional quantum mechanical wave function. It provides tools to define a quantum system—including a custom initial wave function and potential energy landscape—and then solves the time-dependent Schrödinger equation to animate the probability density over time.

## Key Features

-   **Time Evolution Simulation**: Solves the Schrödinger equation to model the behavior of a wave function over time.
-   **Eigenstate Analysis**: Decomposes an initial wave function into its constituent energy eigenstates for detailed analysis of the system's stationary states.
-   **Customizable Potentials**: Easily define arbitrary potential energy functions to simulate various quantum scenarios.
-   **Comparative Visualization**: Animate and compare the evolution of several wave functions simultaneously to study the effects of different parameters.
-   **Interactive Animations**: Generates clear, interactive animations of the wave function's probability density, suitable for analysis and educational purposes.

## Dependencies

To run this program, you will need the following Python libraries:

-   `numpy`
-   `scipy`
-   `matplotlib`
-   `IPython`

You can install these dependencies using pip:

```bash
pip install numpy scipy matplotlib ipython
```

## Usage Examples

Here are a few examples demonstrating how to use PyWave to simulate different quantum systems.

### 1. Simulating a Wave Packet

This example simulates a simple Gaussian wave packet in a zero-potential environment. It shows the fundamental steps of defining an initial state, creating a wave function evolution object, and animating the result.

```python
import import_ipynb
# Assuming pywave.py is in the same directory or in the python path
from pywave import *

# 1. Define an initial state and potential
initial_wave_function = lambda X: np.sqrt(gaussian(X,0.4,0.05)/2) + 1j*np.sqrt(gaussian(X,0.4,0.05)/2)

# Example: A simple cosine potential with magnitude
potential = lambda X: (np.cos(2*π*X)+1)*5
potential_magnitude = 3e-37

# Bonus: Create background to view in animation
def Background_Potential(V,n,L = 1):
    x,Δx = np.linspace(0, L, n,endpoint=False, retstep=True)
    def func(t):
        return x,V(x)
    return func

# 2. Set system number of points
num_points = 100

# 4. Create the wave function object that evolves in time
background = Potential(V,100)
wave = WAVE_1D_FUNCTION(100,initial_wave_function,potential,potential_magnitude)

# 5. Animate the time evolution
animate_multiple(background,wave) # Display in a Jupyter Notebook

```

### 2. Eigenstate Analysis

PyWave can also be used to find the stationary states (eigenstates) of a given system. This is useful for understanding the energy quantization and the fundamental modes of the system.

```python
import import_ipynb
# Assuming pywave.py is in the same directory or in the python path
from pywave import *

# 1. Define an initial state and potential
initial_wave_function = lambda X: np.sqrt(gaussian(X,0.4,0.05)/2) + 1j*np.sqrt(gaussian(X,0.4,0.05)/2)

# Example: A simple cosine potential with magnitude
potential = lambda X: (np.cos(2*π*X)+1)*5
potential_magnitude = 3e-37

# 2. Set system number of points
num_points = 100

# 3. Perform the eigenstate decomposition
eigenvalues, eigenvectors, coefficients, x_coords = WAVE_1D_EIG(
    num_points, 
    initial_wave_function, 
    potential, 
    potential_magnitude,
    CORRECT_UNITS = True
)

mV = 3e-37
Es,Evects,As,x = WAVE_1D_EIG(50,f,V,mV,CORRECT_UNITS = True)


# 4. Display a detailed analysis of the energy levels
EIG_ANALYSIS(eigenvalues, eigenvectors, coefficients, x_coords)
```
