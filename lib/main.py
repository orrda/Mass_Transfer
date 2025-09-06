from system import *

M2 = 1.989 * 10**30  # Solar mass in kg
M1 = 6 * 10**6 * M2  # SMBH mass in kg


"""
M_arr = np.logspace(-1,2,100) * M2

a_critical_values_1 = np.array([a_critical(M1, M2) for M2 in M_arr])
a_critical_values_2 = np.array([a_critical_2(M1, M2) for M2 in M_arr])
xi2 = a_critical_values_2 / a_critical_values_1

plt.plot(M_arr/M1, xi2, ".")

plt.xscale('log')
plt.xlabel('M2/M1')
plt.ylabel('a_critical (m)')
plt.title('Critical Semi-Major Axis vs. Mass Ratio')
plt.grid()
plt.show()
"""



M_arr = np.logspace(0,5,100) * M2
a_guess = 1e+14
a_crit_1_list = []
a_crit_2_list = []

for M in M_arr:
    sys = System(M1, M, a_guess)
    a_crit_1 = sys.a_critical_L1()
    a_crit_2 = sys.a_critical_L2()
    a_crit_1_list.append(a_crit_1)
    a_crit_2_list.append(a_crit_2)
    a_guess = a_crit_1  # Update guess for next iteration

plt.plot(M_arr/M2, a_crit_1_list, ".", label='L1')
plt.plot(M_arr/M2, a_crit_2_list, ".", label='L2')
plt.xscale('log')
plt.xlabel('M2/M1')
plt.ylabel('a_critical (m)')
plt.yscale('log')
plt.title('Critical Semi-Major Axis vs. Mass Ratio')
plt.grid()
plt.legend()
plt.show()

M_arr = np.logspace(0,2,5) * M2

xi_space = np.linspace(0.5,1,100)


for M in M_arr:
    ratios = []
    sys = System(M1, M, 1e+14)
    a_crit = sys.a_critical_L1()
    print(f'For M2 = {M/M2:.2f} M☉, a_critical = {a_crit:.4e} m')
    for xi in xi_space:
        sys.a = xi * a_crit
        L1_flux = sys.mass_flux_L1()
        L2_flux = sys.mass_flux_L2()
        if L1_flux == 0:
            ratio = 0
        else:
            ratio = abs(L2_flux / L1_flux)
        print(f'  At ξ = {xi:.2f}, a = {sys.a:.4e} m, L1 flux = {L1_flux:.2e}, L2 flux = {L2_flux:.2e}, ratio = {ratio:.2e}')
        ratios.append(ratio)

    plt.plot(xi_space, ratios, label=f'M2 = {M/M2:.2f} M☉')

plt.xlabel('ξ (a/a_critical)')
plt.ylabel('L2/L1 Flux Ratios')
plt.title('Flux Ratio vs. ξ for Different Masses')
plt.legend()
plt.grid()
plt.show()




def D_crit(q):
    M1 = 1e+25
    M2 = q*M1
    G = 6.67430e-11
    a = (G * (M1 + M2)/1e+13)**(1/3)
    print("a is - ", a)
    alpha = 3
    sys = System(M1, M2, a)
    f = sys.mass_flux_L2() / sys.mass_flux_L1()
    g = abs(sys.phi_0_L2() - sys.phi_0_L1())
    f = f ** (1/alpha)
    print(f"f={f}, g={g}")
    return g * f / (f - 1)



qs = np.logspace(-6, 0, 100)

D_crit_values = [D_crit(q) for q in qs]
print(D_crit_values)

plt.plot(qs, D_crit_values)
plt.xscale('log')
plt.xlabel('Mass Ratio (q)')
plt.ylabel('D_crit')
plt.title('D_crit vs. Mass Ratio')
plt.grid()
plt.show()
