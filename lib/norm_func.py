import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
import scipy
from scipy.optimize import curve_fit
import scipy.optimize

# Constants
G = 6.67430 * 10**(-11)  # Gravitational constant in m^3 kg^-1 s^-2
K = 1.6 * 10**8  # Kappa in m^2 kg s^-1


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

    return 1 / L1**3 + q / (1 - L1)**3

def getC2(q):
    L2 = get_L2(q)

    return 1 / L2**3 + q / (L2 - 1)**3

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
    
    pot1 = - 1/L1 - q/(1 - L1) - q**2 / (2 * (1 + q))
    pot2 = - 1/L2 - q/(L2 - 1) - q**2 / (2 * (1 + q))

    return pot1 - pot2

def get_f(q):
    B1 = getB1(q)
    B2 = getB2(q)
    C1 = getC1(q)
    C2 = getC2(q)

    return np.sqrt((B1 * C1) / (B2 * C2))

def get_f_mod(q):
    B1 = getB1(q)
    B2 = getB2(q)
    C1 = getC1(q)
    C2 = getC2(q)

    ratio = L2_Massflux_MOD(q)/L1_Massflux_MOD(q)

    return np.sqrt((B1 * C1) / (B2 * C2)) * ratio


def D_crit(q, alpha):
    f = get_f(q) ** (1/alpha)
    return - get_g(q) * f / (f - 1)

def D_crit_numer(q, alpha):
    f = get_f_mod(q) ** (1/alpha)
    return - get_g(q) * f / (f - 1)






def get_alphas(gamma):
    return (3 * gamma - 1) / (2 * (gamma - 1)) 




def Gamma_coefficent3D(gamma,k):
    return 4* np.pi * ((gamma - 1) / (3 * gamma - 1)) * (gamma - 1)**((gamma + 1) / (2 * (gamma - 1))) * (gamma * k)**(-1 / (gamma - 1)) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))

def Gamma_coefficent2D(gamma,k):
    return (gamma * k)**(-1 / (gamma - 1)) * (2 * (gamma - 1) / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1))) * (np.math.gamma((gamma + 1) / (2 * (gamma - 1)) + 1) / np.math.gamma((gamma + 1) / (2 * (gamma - 1)) + 1.5))


def MS_radius(mass):
    solar_mass = 1.989e30
    solar_radius = 6.9634e8
    return (mass / solar_mass)**(0.8) * solar_radius

def MS_phi_0(M2,a):
    R2 = MS_radius(M2)
    M_SMBH = 4* 10**6 * 1.989e30
    omega = np.sqrt(G * (M_SMBH + M2) / a**3)
    return - G * M_SMBH / np.sqrt(a**2 + R2**2) - G * M2 / R2 - 0.5 * omega**2 * ((a * M_SMBH/(M_SMBH + M2))**2 + R2**2)

def L1_massflux(M2,a):
    gamma = 5/3
    M1 = 4 * 10**6 * 1.989e30
    q = M2 / M1
    phi = MS_phi_0(M2,a)
    L1 = get_L1(q) * a
    omega = np.sqrt(G * (M1 + M2) / a**3)
    phi_L1 = - G*M1/L1 - G*M2/(a - L1) - 0.5 * omega**2 * (L1 - a*M2/(M2 + M1))**2
    phi_0 = phi - phi_L1
    C1 = - G * (M1/(L1 ** 3) + M2 / ((a - L1)**3))
    B1 = C1 - omega**2
    A1 = - 2 * C1 - omega**2
    return Gamma_coefficent3D(gamma,K) * phi_0**((3*gamma - 1)/(2*(gamma - 1))) * (1+abs(A1/B1))**(-0.5) * (abs(B1 * C1))**(-0.5)

def L2_massflux(M2,a):
    gamma = 5/3
    M1 = 4 * 10**6 * 1.989e30
    q = M2 / M1
    phi = MS_phi_0(M2,a)
    L2 = get_L2(q) * a
    omega = np.sqrt(G * (M1 + M2) / a**3)
    phi_L2 = - G*M1/L2 - G*M2/(L2 - a) - 0.5 * omega**2 * (L2 - a*M2/(M2 + M1))**2
    phi_0 = phi - phi_L2
    C2 = - G * (M1/(L2 ** 3) + M2 / ((L2 - a)**3))
    B2 = C2 - omega**2
    A2 = - 2 * C2 - omega**2
    return Gamma_coefficent3D(gamma,K) * phi_0**((3*gamma - 1)/(2*(gamma - 1))) * (1+abs(A2/B2))**(-0.5) * (abs(B2 * C2))**(-0.5)


def analytic_error(M2,a):
    R2 = MS_radius(M2)
    gamma = 5/3
    M1 = 4 * 10**6 * 1.989e30
    q = M2 / M1
    phi = MS_phi_0(M2,a)
    L1 = get_L1(q) * a
    omega = np.sqrt(G * (M1 + M2) / a**3)
    phi_L1 = - G*M1/L1 - G*M2/(a - L1) - 0.5 * omega**2 * (L1 - a*M2/(M2 + M1))**2
    phi_0 = phi - phi_L1
    if phi_0 <= 0:
        phi_0 = 0
    C1 = - G * (M1/(L1 ** 3) + M2 / ((a - L1)**3))
    B1 = C1 - omega**2
    L1_radius = np.sqrt(2 * phi_0 / abs(B1))
    error = L1_radius/R2
    return error


def a_start_L1(M2):
    func = lambda x: L1_massflux(M2,x)
    a = scipy.optimize.root_scalar(func, bracket=(0,10**15)).root
    return a

def a_start_L2(M2):
    func = lambda x: L2_massflux(M2,x)
    a = scipy.optimize.root_scalar(func, bracket=(0,10**15)).root
    return a

def a_start_error(M2,thershold):
    func = lambda x: analytic_error(M2,x) - thershold
    a = scipy.optimize.root_scalar(func, bracket=(0,10**15)).root
    return a

def a_Ls_equal(M2):
    func = lambda x: L2_massflux(M2,x)/L1_massflux(M2,x) - 1
    a = scipy.optimize.root_scalar(func, bracket=(0,10**15)).root
    return a 




light_speed = 3e8  # Speed of light in meters per second
gamma = 5/3
solar_mass = 1.989e30
SMBH_mass = solar_mass * 4 * 10**6
schwarzchild_radius = 2 * G * SMBH_mass / (light_speed**2)
print("schwarzchild_radius is - ", schwarzchild_radius)



masses = solar_mass * 10**np.linspace(0,6,100)
a_start_L1s = np.array([a_start_L1(m) for m in masses])
print(a_start_L1s)
a_start_L2s = np.array([a_start_L2(m) for m in masses])
a_ds = np.array([1 - a_start_L2s[i]/a_start_L1s[i] for i in range(len(masses))])
a_Ls_equals = np.array([a_Ls_equal(m) for m in masses])
a_start_errors = np.array([a_start_error(m,0.01) for m in masses])


a_c1 = a_start_L1s[0]
a_array = np.linspace(0.54*a_c1,a_c1,1000)
phis = np.array([MS_phi_0(solar_mass,a) for a in a_array])
plt.plot(a_array/a_c1,phis)
plt.grid()
plt.xlabel("a/a_c [xi]")
plt.ylabel("phi")
plt.show()




for i in range(5):
    mass1 = masses[i*7]
    a_c1 = a_start_L1s[i*7]
    a_mid1 = a_Ls_equals[i*7]

    a_array = np.linspace(a_mid1,a_c1,1000)
    L1_massfluxes = np.array([L1_massflux(mass1,a) for a in a_array])/mass1
    L2_massfluxes = np.array([L2_massflux(mass1,a) for a in a_array])/mass1


    plt.plot(a_array/a_c1,L2_massfluxes/L1_massfluxes, label = str(mass1/solar_mass)[:7] + " solar mass")

plt.legend()
plt.grid()
plt.xlabel("a/a_c [xi]")
plt.ylabel("L2_massflux/L1_massflux")
plt.show()


equal_massfluxes = np.array([2*L2_massflux(m,a_Ls_equal(m)) for m in masses])
plt.plot(masses/SMBH_mass,equal_massfluxes/masses, ".")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.xlabel("mass ratio [q]")
plt.ylabel("total massflux/mass [1/s]")
plt.title("total massflux when L1 and L2 massfluxes are equal")
plt.show()

der = np.diff(np.log(equal_massfluxes))
plt.plot(masses[1:]/SMBH_mass,der, ".")
plt.xscale("log")
plt.grid()
plt.xlabel("mass ratio [q]")
plt.ylabel("derivative of total massflux")
plt.title("derivative of total massflux when L1 and L2 massfluxes are equal")
plt.show()



plt.plot(masses/solar_mass,a_start_L1s/schwarzchild_radius, color = "blue")
plt.plot(masses/solar_mass,a_start_L2s/schwarzchild_radius, color = "red")
plt.plot(masses/solar_mass,a_Ls_equals/schwarzchild_radius, color = "yellow")
#plt.plot(masses/solar_mass,a_start_errors/schwarzchild_radius, color = "pink")
plt.xlabel("masses")
plt.ylabel("a")
plt.xscale("log")
plt.yscale("log")
plt.show()




qs = masses / (4 * 10**6 * solar_mass)
alpha = get_alphas(5/3)
D_crits = [D_crit(q, alpha) for q in qs]
D_crits_numers = np.array([D_crit_numer(q, alpha) for q in qs])

plt.plot(qs,a_start_L1s/a_Ls_equals, color = "blue")
plt.plot(qs,D_crits_numers, color = "red")
plt.xscale("log")
plt.show()

plt.plot(qs,D_crits_numers/(a_start_L1s/a_Ls_equals), color = "blue")
plt.xscale("log")
plt.show()




