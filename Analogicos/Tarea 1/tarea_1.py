import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.special import sph_harm
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Constantes
h = 6.62607004e-34  # Constante de Planck [J s]
e = 1.602176634e-19  # Carga del electrón [C]
m = 9.10938356e-31  # Masa del electrón [kg]
epsilon0 = 8.85418782e-12  # Permitividad del vacío [F m^-1]
a0 = 4 * np.pi * epsilon0 * h**2 / (m * e**2)  # Radio de Bohr [m]

# Parte 1: Niveles de energía
n = np.array([1, 2, 3])  # Números cuánticos principales
E = -m * e**4 / (8 * epsilon0**2 * h**2 * (n**2))  # Energía [J]
E_eV = E / e  # Energía [eV]
for i in range(len(n)):
    print(f"Para el nivel n = {i + 1}, su energía es de {E[i]:.2e} J, y {E_eV[i]:.2f} eV")

def psi(n,l,m,r,theta,phi):
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(2 * r / (n * a0))
    harmonic = sph_harm(m, l, theta, phi)
    R = np.sqrt((2 / (n * a0))**3 * np.math.factorial(n - l - 1) /
                (2 * n * np.math.factorial(n + l))) * \
        np.exp(-r / (n * a0)) * (2 * r / (n * a0))**l * laguerre * harmonic
    P = np.abs(R)**2 * 4 * np.pi * r**2 
    return P*a0

# Parte 2: Probabilidad con l, m = 0
l = 0  
m = 0  
r = np.linspace(0, 25 * a0, 1000) 
theta = np.linspace(0, np.pi, 1000)  
phi = np.linspace(0, 2 * np.pi, 1000)  
lim = [5, 15, 25]

# Calculo de la probabilidad radial, graficación y radio de máxima probabilidad

for i in range(len(n)):
    P = psi(n[i], l, m, r, theta, phi)
    plt.plot(r / a0, P)
    plt.xlim([0, lim[i]])
    plt.title(f'Orbital {n[i]}s')
    plt.xlabel('r/a_0')
    plt.ylabel('Probabilidad radial (a0*dP/dr)')
    max_P, index = max(P), np.argmax(P)
    r_max = r[index]
    print(f'Radio de máxima probabilidad para el orbital {n[i]}s: {r_max} m, {r_max / a0:.2f} a_0')
    image_path = os.path.join(current_dir, "P2-"+str(i)+".png")
    plt.savefig(image_path)
    plt.show()

# Parte 3: Grafico en 3D
r_max_list = [1 * a0, 5.23 * a0]
n = [1, 2]
l = [0, 1]
m = [-1, 0, 1]
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

def cart(P, phi, theta, n, l, m):
    x = P * np.sin(phi) * np.cos(theta)
    y = P * np.sin(phi) * np.sin(theta)
    z = P * np.cos(phi)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(x[:, 0], z[:, 0], linewidth=0.5, alpha=0.5, label='Crossection', zorder=1,color='white')
    ax1.set_facecolor('black')
    ax1.set_xlabel('Eje X')
    ax1.set_ylabel('Eje Z')
    if l == 0:
        ax1.set_title('Crossection XZ del orbital ' + str(n) + 's')
    if l == 1:
        if m == 1:
            label = "z"
            ax1.set_title('Crossection XZ del orbital ' + str(n) + 'p' + label)
        if m == 0:
            label = "y"
            ax1.set_title('Crossection XZ del orbital ' + str(n) + 'p' + label)
        if m == -1:
            label = "x"
            ax1.set_title('Crossection XZ del orbital ' + str(n) + 'p' + label)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, z, cmap='magma')
    ax2.set_xlabel('Eje X')
    ax2.set_ylabel('Eje Y')
    ax2.set_zlabel('Eje Z')
    if l == 0:
        ax2.set_title('Probabilidad de encontrar un electrón en el orbital ' + str(n) + 's')
        label = "s"
        m=""
    if l == 1:
        label="p"
        if m == 1:
            m = "z"
            ax2.set_title('Probabilidad de encontrar un electrón en el orbital ' + str(n) + 'p' + m)
        if m == 0:
            m = "y"
            ax2.set_title('Probabilidad de encontrar un electrón en el orbital ' + str(n) + 'p' + m)
        if m == -1:
            m = "x"
            ax2.set_title('Probabilidad de encontrar un electrón en el orbital ' + str(n) + 'p' + m)
    image_path = os.path.join(current_dir, "P3-"+str(n)+str(label)+str(m)+".png")
    plt.savefig(image_path)
    plt.show()

for i in range(len(n)):
    r_max = r_max_list[i]
    for j in range(len(l)):
        if l[j] == 0:
            M = 0
            p = psi(n[i], l[j], M, r_max, theta, phi)
            cart(p, phi, theta, n[i], l[j], M)
        if l[j] == 1:
            for k in range(len(m)):
                if n[i] == 1:
                    pass
                else:
                    p = psi(n[i], l[j], m[k], r_max, theta, phi)
                    cart(p, phi, theta, n[i], l[j], m[k])
