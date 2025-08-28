import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

# Constants

phi_0 = 10
gamma = 5/3
KK = 1

fig, axs = plt.subplots(1, 2, figsize=(8, 10))


i = 0

rho_arr = np.linspace(0.001, 10, 10000)
physical_M_dot = ((phi_0 * (gamma - 1) * (2/(gamma + 1)))**((gamma + 1)/(2*(gamma - 1)))) * ((gamma*KK)**(-1/(gamma - 1)))

M_dots = np.linspace(0.9, 1.1, 11) * physical_M_dot

print("physical_M_dot = ", physical_M_dot)

color = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange', 'purple', 'brown', 'pink']


for M_dot in M_dots:
    x_space = np.sqrt(- phi_0 + (M_dot**2)/(2*(rho_arr**2)) + ((gamma * KK)/(gamma - 1)) * (rho_arr**(gamma - 1)))
    c_space = (gamma *(rho_arr**(gamma - 1)))**(1/2)
    v_space = M_dot/rho_arr


    max_index = np.argmax(x_space)
    if max_index == 0:
        max_index = np.argmax(-x_space)
    print(max_index)


    new_x_space = [x_space[:max_index]] + [-x_space[max_index:]]
    new_x_space = np.concatenate(new_x_space)

    V_C = v_space/c_space
    V_C2 = (M_dot/np.sqrt(gamma))*(rho_arr**(-(gamma + 1)/2))
    axs[0].plot(new_x_space, V_C, color = color[i])    


    axs[1].plot(new_x_space, rho_arr, color = color[i])
    axs
    i += 1
    
axs[0].set_xlabel('x')
axs[0].set_ylabel('V/C')
axs[0].set_title('Mach number profile')
axs[0].grid()

axs[0].set_xlim(-2, 2)
axs[0].set_ylim(0, 4)


axs[1].set_xlabel('x')
axs[1].set_ylabel('rho')
axs[1].set_title('Density profile')
axs[1].grid()
axs[1].set_xlim(-2, 2)
axs[1].set_ylim(0, 10)
    

plt.tight_layout()
plt.legend(M_dots)
strr = "for Phi_0 = " + str(phi_0) + " and K = " + str(KK) + " and gamma = " + str(gamma)
fig.suptitle(strr)
plt.show()


phi_0 = 10
gamma = 5/3
KK = 1



i = 0

rho_arr = np.linspace(0.01, 50, 10000)
physical_M_dot = ((phi_0 * (gamma - 1) * (2/(gamma + 1)))**((gamma + 1)/(2*(gamma - 1)))) * ((gamma*KK)**(-1/(gamma - 1)))

M_dots = np.linspace(0.9, 1.1, 9) * physical_M_dot

print("physical_M_dot = ", physical_M_dot)

color = ['r', 'g', 'b', 'm', 'k', 'orange', 'purple', 'brown', 'pink']


for M_dot in M_dots:
    x_space = (- phi_0 + (M_dot**2)/(2*(rho_arr**2)) + ((gamma * KK)/(gamma - 1)) * (rho_arr**(gamma - 1)))**(1/2)
    c_space = (gamma *(rho_arr**(gamma - 1)))**(1/2)
    v_space = M_dot/rho_arr


    max_index = np.argmax(-x_space)
    if max_index == 0:
        pass
        #max_index = np.argmax(x_space)
    print(max_index)


    new_x_space = [x_space[:max_index]] + [-x_space[max_index:]]
    new_x_space = np.concatenate(new_x_space)

    V_C = v_space/c_space
    V_C2 = (M_dot/np.sqrt(gamma))*(rho_arr**(-(gamma + 1)/2))
    plt.plot(new_x_space, rho_arr, color = color[i])    
    i += 1


x_space = (- phi_0 + (physical_M_dot**2)/(2*(rho_arr**2)) + ((gamma * KK)/(gamma - 1)) * (rho_arr**(gamma - 1)))**(1/2)
V_C2 = (physical_M_dot/np.sqrt(gamma))*(rho_arr**(-(gamma + 1)/2))
max_index = np.argmax(x_space)
if max_index == 0:
    max_index = np.argmax(-x_space)
print(max_index)
new_x_space = [x_space[:max_index]] + [-x_space[max_index:]]
new_x_space = np.concatenate(new_x_space)
plt.plot(new_x_space, rho_arr, 'k--', linewidth = 3)


plt.xlabel('x')
plt.ylabel(r'$\rho$')
plt.title('density profile')
plt.grid()
plt.plot(np.linspace(-2, 2, 100), np.zeros(100), 'k')
plt.xlim(-3, 3)
plt.ylim(0, 15)





legend = ["M_dot * "+str(i)[:4] for i in np.linspace(0.9, 1.1, 9)]
plt.legend(legend)
strr = "for Phi_0 = " + str(phi_0) + ", K = " + str(KK) + ", gamma = " + str(gamma)[:5]
plt.title(strr)
plt.show()
