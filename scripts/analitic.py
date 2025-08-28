import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

def Mdot(phi0, gamma, K):
    result = phi0**((gamma+1)/(2*(gamma-1))) * gamma * K**(-1/(gamma-1)) * (gamma-1)**((gamma+1)/(2*(gamma-1))) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
    return result

phi0 = 1
KK = 0.5

Mdot53 = Mdot(phi0, 5/3, KK)
Mdot43 = Mdot(phi0, 4/3, KK)

Mdot53 = Mdot53/Mdot53
Mdot43 = Mdot43/Mdot53

plt.plot(np.linspace(1, 2, 100), Mdot53 * np.ones(100),  label='gamma = 5/3')
plt.plot(np.linspace(1, 2, 100), Mdot43 * np.ones(100),  label='gamma = 4/3')

plt.plot(np.linspace(1, 2, 100), Mdot(phi0, np.linspace(1, 2, 100), KK)/Mdot53, label='phi0 = phi0')
plt.loglog()
plt.xlabel('gamma')
plt.ylabel('Mdot')
plt.ylim(0.0001, 1.5)
plt.xlim(1, 2)
plt.legend()
plt.title('Mdot vs gamma')

plt.show()