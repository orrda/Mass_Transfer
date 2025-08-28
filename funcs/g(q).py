import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
G = 6.67430e-11




def get_L1(q):
    mu = q / (1 + q)
    func1 = lambda x: x**5 + (mu - 3)*x**4 + (3 - 2*mu)*x**3 - mu*x**2 + 2*mu*x - mu
    guess1 = (mu/3)**(1/3)
    L1 = fsolve(func1, guess1)[0]
    return 1 - L1

def get_L2(q):
    mu = q / (1 + q)
    func2 = lambda x: x**5 - (mu - 3)*x**4 + (3 - 2*mu)*x**3 - mu*x**2 - 2*mu*x - mu
    guess2 = (mu/3)**(1/3)
    L2 = fsolve(func2, guess2)[0]
    return 1 + L2

def get_g(q):
    L1 = get_L1(q)
    L2 = get_L2(q)
    
    pot1 = - 1/L1 - q/(1 - L1) - q**2 / (2 * (1 + q))
    pot2 = - 1/L2 - q/(L2 - 1) - q**2 / (2 * (1 + q))

    return pot2 - pot1


def get_f(q):
    L1 = get_L1(q)
    L2 = get_L2(q)

    C1 = 




q_powers = np.linspace(-14, 0, 100)
q = 10**q_powers
g_arr = np.array([get_g(qi) for qi in q])



g_1 = get_g(1)
arrd = np.log(g_arr / g_1) / np.log(q)

plt.plot(q, arrd)
plt.xscale('log')
plt.show()


