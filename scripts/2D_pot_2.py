import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

# Constants

phi_0 = 0.5
gamma = 5/3
KK = 1

x_min = -2
x_max = 0

y_min = -2
y_max = 2

A = 1
B = 1

delta = 0.01


phi = lambda x, y: B * (y**2) - A * (x**2)

grid = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))


plt.show()



iter_num = 500

phi_0 = np.ones_like(grid[0]) * phi_0

phi = phi(grid[0], grid[1])
psi = phi


plt.imshow(phi, cmap='hot', origin='lower')
plt.colorbar()
plt.title('phi')
plt.show()

fft_psi = np.fft.fft2(phi)
plt.imshow(np.abs(fft_psi), cmap='hot', origin='lower')
plt.colorbar()
plt.title('fft_psi')
plt.show()


for i in range(iter_num):


    psi_grad = np.array(np.gradient(psi))
    psi_grad = np.moveaxis(np.array(psi_grad),0,2)

    x_max_mat = np.where(grid[0] >= x_max)
    psi_grad[x_max_mat] = [0.3, 0]

    psi_grad_squared = psi_grad**2
    psi_grad_squared = np.sum(psi_grad_squared, axis=2)

    laplacian = sci.ndimage.laplace(psi)


    second_term = np.moveaxis(np.array(np.array(np.gradient(phi + 0.5 * psi_grad_squared))),0,2)  * psi_grad
    second_term = np.sum(second_term, axis=2)


    first_term = (gamma - 1) * (phi - phi_0 + 0.5 * psi_grad_squared) * laplacian

    residual = first_term + second_term

    


    if i % 10 == 0:
        print("iter: ", i)
        
        plt.imshow(psi, cmap='hot', origin='lower')
        plt.colorbar()
        plt.title('psi')
        plt.show()
        

    psi = psi - delta * residual


    negitve_phi = np.where(phi_0 - phi < 0)
    psi[negitve_phi] = 0



print("psi ", psi)
plt.imshow(psi, cmap='hot', origin='lower')
plt.colorbar()
plt.title('psi')
plt.show()

print("psi_grad_squared ", psi_grad_squared)
plt.imshow(psi_grad_squared, cmap='hot')
plt.colorbar()
plt.title('psi_grad_squared')
plt.show()




