import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar

# Constants
G = 6.67430e-11


def potantial(r, omega, M1, M2, R):
    d = (M2 * R) / (M1 + M2)
    return - G * M1 / r - G * M2 / np.abs(R - r) - 0.5 * omega**2 * (r - d)**2

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


def getB1(q):
    L1 = get_L1(q)

    return 1 / ((L1**3) * (1 + q)) + q / (((1 - L1)**3) * (1 + q)) - 1 

def getB2(q):
    L2 = get_L2(q)

    C2 = 1 / ((L2**3) * (1 + q)) + q / (((L2 - 1)**3) * (1 + q))

    return 1 / ((L2**3) * (1 + q)) + q / (((L2 - 1)**3) * (1 + q)) - 1 

def getC1(q):
    L1 = get_L1(q)

    return 1 / (L1**3) + q / ((1 - L1)**3)

def getC2(q):
    L2 = get_L2(q)

    return 1 / (L2**3) + q / ((L2 - 1)**3)

def getA1(q):
    L1 = get_L1(q)

    return - 2 / ((L1**3) * (1 + q)) - 2*q / (((1 - L1)**3) * (1 + q)) - 1 

def getA2(q):
    L2 = get_L2(q)

    return - 2 / ((L2**3) * (1 +q )) - 2*q / (((L2 - 1)**3) * (1 + q)) - 1 


def L1_Massflux(q):
    L1 = get_L1(q)
    C1 = 1 / L1**3 + q / (1 - L1)**3
    B1 = C1 - 1 - q
    return Gamma_coefficent3D(gamma,1) * 4 * (1 + q)/ np.sqrt(B1 * C1)

def L2_Massflux(q):
    L2 = get_L2(q)
    C2 = 1 / L2**3 + q / (L2 - 1)**3
    B2 = C2 - 1 - q
    return Gamma_coefficent3D(gamma,1) * 4 * (1 + q)/ np.sqrt(B2 * C2)


def L1_Massflux_MOD(q):
    coeff = 1
    L1 = get_L1(q)
    C1 = 1 / ((L1**3) * (1 + q)) + q / (((1 - L1)**3) * (1 + q))
    B1 = C1 - 1 
    A1 = - 2*C1 - 1
    avg = (B1 + C1)/2
    factor = 1/np.sqrt(1 + coeff*abs(A1/avg))
    return factor

def L2_Massflux_MOD(q):
    coeff = 1
    L2 = get_L2(q)
    C2 = 1 / ((L2**3) * (1 + q)) + q / (((L2 - 1)**3) * (1 + q))
    B2 = C2 - 1
    avg = (B2 + C2)/2
    A2 = - 2* C2 - 1
    factor = 1/np.sqrt(1 + coeff*abs(A2/avg))
    return factor






def get_g(q):
    L1 = get_L1(q)
    L2 = get_L2(q)
    
    pot1 = - 1/L1 - q/(1 - L1) - 0.5 * (1 + q) * ( L1 - q/(1 + q))**2 
    pot2 = - 1/L2 - q/(L2 - 1) - 0.5 * (1 + q) * ( L2 - q/(1 + q))**2 

    return pot1 - pot2

def get_f(q):
    #B1 = getB1(q)
    #B2 = getB2(q)
    C1 = getC1(q)
    C2 = getC2(q)
    f = C1 / C2
    print("for q =", q, "f =", f)

    return C1 / C2

def get_f_mod(q):
    B1 = getB1(q)
    B2 = getB2(q)
    C1 = getC1(q)
    C2 = getC2(q)

    ratio = L2_Massflux_MOD(q)/L1_Massflux_MOD(q)

    return np.sqrt((B1 * C1) / (B2 * C2)) * ratio


def D_crit(q, alpha):
    f = (get_f(q)) ** (1/alpha)
    return - get_g(q) * f / (f - 1)

def D_crit_numer(q, alpha):
    f = (get_f_mod(q)) ** (1/alpha)
    return - get_g(q) * f / (f - 1)






def get_alphas(gamma):
    return (3 * gamma - 1) / (2 * (gamma - 1)) 

def get_q_crit(alpha):
    result = minimize_scalar(D_crit, bounds=(0, 1), args=(alpha,), method='bounded')
    return result.x






def Gamma_coefficent3D(gamma,k):
    return 4* np.pi * ((gamma - 1) / (3 * gamma - 1)) * (gamma - 1)**((gamma + 1) / (2 * (gamma - 1))) * (gamma * k)**(-1 / (gamma - 1)) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))

def Gamma_coefficent2D(gamma,k):
    return (gamma * k)**(-1 / (gamma - 1)) * (2 * (gamma - 1) / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1))) * (np.math.gamma((gamma + 1) / (2 * (gamma - 1)) + 1) / np.math.gamma((gamma + 1) / (2 * (gamma - 1)) + 1.5))


def get_C_ratio(q):
    L1 = get_L1(q)
    L2 = get_L2(q)
    C1 = 1 / L1**3 + q / (1 - L1)**3
    C2 = 1 / L2**3 + q / (L2 - 1)**3
    return C1/C2









gamma = 5/3
powers = np.linspace(-6, 0, 100)
qs = 10**powers


L1_arr = [1 - get_L1(q) for q in qs]
L2_arr = [get_L2(q) - 1 for q in qs]
mu_arr = [(q/(3*(1 + q)))**(1/3) for q in qs]

plt.plot(qs, L1_arr, label='L1')
plt.plot(qs, L2_arr, label='L2')
plt.plot(qs, mu_arr, label='mu')
plt.legend()
plt.xscale('log')
plt.show()

alpha = 3
D = 0.2
f_arr = np.array([get_f(q) for q in qs])
g_arr = np.array([1+D*get_g(q) for q in qs])

D_crits_numer = (g_arr**3) * f_arr

plt.plot(qs, f_arr, label='f')
plt.plot(qs, g_arr, label='g')
plt.plot(qs, D_crits_numer, label='D_crit_numer')
plt.legend()
plt.xscale('log')
plt.show()

D_crits = [D_crit(q, alpha) for q in qs]

mu_arr = np.array([q/(1+q) for q in qs])
D_crits_anl = (8/3) * (mu_arr/3)**(3/2)


plt.plot(qs, D_crits, label='D_crit')
plt.plot(qs, D_crits_anl, label='D_crit_anl')
plt.grid()
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('q')
plt.ylabel('D_crit')


q_crit = get_q_crit(alpha)
print("q_crit = ", q_crit)
plt.title("gamma = 5/3")
plt.show()


D_crits = np.array(D_crits)

xi_crit = (1 + D_crits)**(-1)

plt.plot(qs, xi_crit, label=r'$\xi_{crit}$')
plt.grid()
plt.xscale('log')
plt.xlabel('q')
plt.ylabel(r'$\xi_{crit}$')
plt.legend()
plt.show()




q_power = -0.1
coeff = np.array([q**q_power for q in qs])
stat = D_crits_numer * coeff

plt.plot(qs, stat)
plt.grid()
plt.xscale('log')
plt.show()


massfluxs1 = []
massfluxs2 = []


for q in qs:

    B1 = getB1(q)
    B2 = getB2(q)
    C1 = getC1(q)
    C2 = getC2(q)

    massflux1 = Gamma_coefficent3D(gamma,1) / np.sqrt(B1 * C1)
    massflux2 = Gamma_coefficent3D(gamma,1) / np.sqrt(B2 * C2)

    massfluxs1.append(massflux1 * 4 * (1 + q))
    massfluxs2.append(massflux2 * 4 * (1 + q))



massfluxs1 = np.array(massfluxs1)
massfluxs2 = np.array(massfluxs2)


Teo_data = np.array([
    [0.000001, 0.27432, 0.28106, 1.08682, 1.11351],
    [0.0001, 0.26229, 0.29356, 1.03914, 1.16303],
    [0.010, 0.21217, 0.35740, 0.84058, 1.41594],
    [0.100, 0.16104, 0.48436, 0.63803, 1.91895],
    [0.300, 0.13879, 0.64014, 0.54987, 2.53614],
    [1.000, 0.12854, 1.01709, 0.50926, 4.02955],
    [3.333, 0.13879, 1.82868, 0.54987, 7.24495],
    [10.000, 0.16104, 3.20845, 0.63803, 12.71137]
])

ques = Teo_data[:, 0]
TeoL15_3 = Teo_data[:, 1]
TeoL25_3 = Teo_data[:, 2]
TeoL1_1 = Teo_data[:, 3]
TeoL2_1 = Teo_data[:, 4]


