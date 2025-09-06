import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import fsolve


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


def potential(x, q):
    return -1/x - 1/np.abs(1 - x) - 0.5 * (1+q) * (x - (q/(1+q)))**2

def C1(q):
    L1 = get_L1(q)
    return 1 / (L1**3) + q / ((1 - L1)**3)

def C2(q):
    L2 = get_L2(q)
    return 1 / (L2**3) + q / ((L2 - 1)**3)

def ratio(phi0, q):
    first_term = (phi0 - potential(get_L2(q), q))/(phi0 - potential(get_L1(q), q))
    second_term = C1(q) / C2(q)
    return (first_term**3) * second_term

def ratio_vec(phi_arr, q):
    L1 = get_L1(q)
    L2 = get_L2(q)
    first_term = (phi_arr - potential(L2, q))/(phi_arr - potential(L1, q))
    second_term = C1(q) / C2(q)
    return (first_term**3) * second_term

def D_crit(q):
    L1 = get_L1(q)
    L2 = get_L2(q)
    pot1 = potential(L1, q)
    pot2 = potential(L2, q)
    second_term = C1(q) / C2(q)
    print(f"for q={q}, the second term is: {second_term}")
    func = lambda x: (((x - pot2)/(x - pot1))**3) * second_term
    x = scipy.optimize.minimize_scalar(lambda x: np.abs(func(x) - 1), bounds=(-200,200), method='bounded').x
    print(f"for q={q} the difference to L1 is: {x - pot1}, and for L2 is: {x - pot2}, and the ratio is: {func(x)}")
    return x/pot1



qs = np.logspace(-15 ,15, 100)

D_crits = np.array([D_crit(q) for q in qs])

plt.plot(qs, D_crits, label='D_crit')
plt.xlabel('q')
plt.ylabel('D_crit')
plt.xscale('log')
plt.title('D_crit vs q')
plt.legend()
plt.show()
