import numpy as np
import scipy
import matplotlib.pyplot as plt



# Constants
G = 6.67430 * 10**(-11)  # Gravitational constant in m^3 kg^-1 s^-2
K = 1.6 * 10**8  # Kappa in m^2 kg s^-1
Solar_Mass = 1.989 * 10**30  # Solar mass in kg

class System:
    def __init__(self, M1, M2, a):
        self.M1 = M1
        self.M2 = M2
        self.a = a

    def potential(self, r):
        return - G * self.M1 / r - G * self.M2 / np.abs(self.a - r) - 0.5 * self.get_omega()**2 * (r - (self.M2 * self.a) / (self.M1 + self.M2))**2


    def a_critical_L1(self):
        if hasattr(self, 'crit_a_L1'):
            return self.crit_a_L1
        radius = self.MS_radius()
        def func(a):
            sys = System(self.M1, self.M2, a)
            L1 = sys.get_L1()
            return a - L1 - radius
        crit_a = scipy.optimize.fsolve(func, self.a)[0]
        self.crit_a_L1 = crit_a
        print(f'Critical a (L1): {crit_a:.2e} m')
        return crit_a
    
    def a_critical_L2(self):
        if hasattr(self, 'crit_a_L2'):
            return self.crit_a_L2
        radius = self.MS_radius()
        def func(a):
            sys = System(self.M1, self.M2, a)
            L2 = sys.get_L2()
            return L2 - a - radius
        crit_a = scipy.optimize.fsolve(func, self.a)[0]
        self.crit_a_L2 = crit_a
        print(f'Critical a (L2): {crit_a:.2e} m')
        return crit_a

    def get_L1(self):
        q = self.M2 / self.M1
        mu = q / (1 + q)
        func1 = lambda x: x**5 + (mu - 3)*x**4 + (3 - 2*mu)*x**3 - mu*x**2 + 2*mu*x - mu
        guess1 = (mu/3)**(1/3)
        L1 = scipy.optimize.fsolve(func1, guess1)[0]
        self.L1 = self.a * (1 - L1)
        return self.L1

    def get_L2(self):
        q = self.M2 / self.M1
        mu = q / (1 + q)
        func2 = lambda x: x**5 - (mu - 3)*x**4 + (3 - 2*mu)*x**3 - mu*x**2 - 2*mu*x - mu
        guess2 = (mu/3)**(1/3)
        L2 = scipy.optimize.fsolve(func2, guess2)[0]
        self.L2 = self.a * (1 + L2)
        return self.L2

    def get_C1(self):
        L1 = self.get_L1()
        return G * self.M1 / (L1**3) + G *self.M2 / ((self.a - L1)**3)

    def get_C2(self):
        L2 = self.get_L2()
        return G * self.M1 / (L2**3) + G * self.M2 / ((L2 - self.a)**3)

    def get_omega(self):
        return np.sqrt(G * (self.M1 + self.M2) / self.a**3)

    def Gamma_coefficent3D(self,gamma):
        return 4* np.pi * ((gamma - 1) / (3 * gamma - 1)) * (gamma - 1)**((gamma + 1) / (2 * (gamma - 1))) * (gamma * K)**(-1 / (gamma - 1)) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))

    def MS_radius(self):
        solar_mass = 1.989e30
        solar_radius = 6.9634e8
        return (self.M2 / solar_mass)**(0.8) * solar_radius
    
    def phi_0_L1(self):
        crit_a = self.a_critical_L1()
        xi = self.a / crit_a
        if xi >= 1:
            return 0
        sys2 = System(self.M1, self.M2, crit_a)
        pot = abs(sys2.potential(sys2.get_L1()))
        return pot * (1/xi - 1)
    

    def phi_0_L2(self):
        crit_a = self.a_critical_L2()
        xi = self.a / crit_a
        if xi >= 1:
            return 0
        sys2 = System(self.M1, self.M2, crit_a)
        pot = abs(sys2.potential(sys2.get_L2()))
        return pot * (1/xi - 1)

    def mass_flux_L1(self):
        phi_0 = self.phi_0_L1()
        if phi_0 < 0:
            return 0
        C1 = self.get_C1()
        omega = self.get_omega()
        gamma_coeff = self.Gamma_coefficent3D(5/3)
        alpha = 3
        return gamma_coeff * (phi_0**alpha) /(C1 * omega)

    def mass_flux_L2(self):
        phi_0 = self.phi_0_L2()
        if phi_0 < 0:
            return 0
        C2 = self.get_C2()
        omega = self.get_omega()
        gamma_coeff = self.Gamma_coefficent3D(5/3)
        alpha = 3
        return gamma_coeff * (phi_0**alpha)/(C2 * omega)
    



