import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

# Constants

phi_0 = 10
gamma = 5/3
KK = 1

x_min = -2
x_max = 0

y_min = -2
y_max = 2

A = 1
B = 1


phi = lambda x, y: B * (y**2) - A * (x**2)

grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


plt.show()



iter_num = 100

psi = np.zeros_like(grid[0])
phi = phi(grid[0], grid[1])

for i in range(iter_num):

    levels_phi = np.linspace(np.min(phi), np.max(phi), 10)
    plt.contour(grid[0], grid[1], phi, levels=levels_phi, colors='black')

    levels_psi = np.linspace(np.min(psi), np.max(psi), 1)
    plt.contour(grid[0], grid[1], psi, levels=levels_psi, colors='black')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Potential')
    plt.show()

    print("psi: ", psi.shape)
    psi_grad = np.array(np.gradient(psi))
    psi_grad = np.moveaxis(np.array(psi_grad),0,2)
    print("psi_grad: ", psi_grad.shape)
    psi_grad_squared = np.inner(psi_grad, psi_grad)
    print("psi_grad_squared: ", psi_grad_squared.shape)
    laplacian = sci.ndimage.laplace(psi)
    plt.plot(laplacian)
    plt.show()
    second_term = np.gradient(phi + psi_grad_squared).dot(psi_grad)

    residual = (gamma - 1) * (phi - phi_0 + 0.5 * psi_grad_squared) * laplacian + second_term

    plt.plot(residual)
    plt.show()




