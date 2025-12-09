import numpy as np
from scipy import special
import matplotlib
import tkinter as tk
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm
from numpy import load

# fancy plt conf:
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

# constants:
C = 2.99792458e-10
C_SI = C*1e2
QE = 4.80320427e-10
QE_SI = 1.60218e-19
ME = 9.10938e-28
ME_SI = ME*1e-3
MI = 239332.4*ME#(xenon)
MI_SI = MI*1e-3
pi=3.14
# parameters, [SI]:
E =2000
B =0.02
n =1e17
L=0.04304
Te =10*QE_SI#0.2*QE_SI
Ti = 0.0*QE_SI
v0 = E/B
m = MI/ME
vte = np.sqrt(Te/ME_SI)
vti = np.sqrt(Ti/MI_SI)
l_d = np.sqrt(Te*1e7/(4.*np.pi*n*1e-6*QE**2))*1e-2
wpi = np.sqrt((4.*np.pi*n*1e-6*QE**2)/MI)
wpe = np.sqrt((4.*np.pi*n*1e-6*QE**2)/ME)
wce = QE_SI*B/ME_SI
rho_e=vte/wce
cs = l_d*wpi
k0 = wce/v0
Ln=-0.0049504
vn=0.0#vte**2/(Ln*wce)

# parameters output:
print("----------Parameters, [SI]----------")
print("n = %.1e [m-3]"%n)
print("w_ce = %.2e [1/s]"%wce)
print("w_pi = %.2e [1/s]"%wpi)
print("w_pe = %.2e [1/s]"%wpe)
print("l_d = %.2e [m]"%l_d)
print("v_E = %.1e [m/s]"%v0)
print("c_s = %.1e [m/s]"%cs)
print("k0*l_d = %.2e"%(k0))
print("v_Te = %.2e"%vte)
print("v_Ti = %.2e"%vti)


# normalized parameters:
v0 /= cs
vti /= cs
vte /= cs
wce /= wpi
vn /= cs

print("----------Normalized parameters----------")
print("v0 = %.1e"%v0)
print("vti = %.1e"%vti)
print("vte = %.1e"%vte)
print("w_ce = %.2e"%wce)
print("m = mi/me = %.2e"%m)


kyvec_cyc = np.linspace(0.0,4., 800)
#print(kyvec_cyc)
kyvec = kyvec_cyc*k0*l_d
#kyvec = kyvec_cyc*l_d/rho_e
#kyvec=np.arange(1,300)*2*pi*l_d/L
#kyvec_cyc = kyvec*rho_e/l_d
#kzvec_cyc = np.linspace(0.0001, 0.3, 10)
#kzvec = kzvec_cyc*k0*l_d

kzvec = np.array( [0.01,0.02,0.04,0.09])

#print(kzvec)


#kzvec_cyc = kzvec/l_d/k0

def main():
   
    w1, w2 = CVLR_solve(newsol = 1.0, save=True)
    for i in range(len(kzvec)):
         w3=np.array(np.imag((w1[i])))
         w4=np.array(np.real((w1[i])))
         np.savetxt("lnimagE=20000Te=20,ln=-0.0005.txt", w3, fmt="%s")
         np.savetxt("lnrealE=20000Te=20,ln=-0.0005.txt", w4, fmt="%s")
    plot_im_re_sep(w1)
    #plot_kz_ky(w1)




def Z(x):
    """Plasma dispersion function"""
    
    return np.sqrt(np.pi)*1j*special.wofz(x)

def dZdx(x): 
    """1st derivative of plasma dispersion function"""
    
    return -2.0*(1 + x*Z(x))

def d2Zdx2(x):
    """2nd derivative of plasma dispersion function"""
    
    zval = Z(x)
    return (-2.+4.*x**2)*(-4.+zval-4.*x*zval) 

def gsum(Omega,X,Y,N):
    """Partial sum from Gordeev function"""
    
    term = 0.0
    for i in range(-N,N+1): # [-N,...,0,...,N]
        term += Z((Omega - i)/np.sqrt(2*Y))*special.iv(i,X)
    return term

def g(Omega,X,Y,N):
    """Gordeev function, uses gsum(Omega,X,Y,N) for partial sum.
    N should be less than ~15, as Z(x) starts to take arguments 
    outside its domain."""
    
    return (Omega)/np.sqrt(2*Y)*np.exp(-X)*gsum(Omega,X,Y,N)


def CVLR_solve(newsol = True, save=False):

    if not newsol:
        w1 = np.load("w1.npy")
        w2 = np.load("w2.npy")
        return w1, w2

    # arrays for storing solutions, w_plus and w_minus, respectively:
    w1 = np.zeros((kzvec.size,kyvec.size),dtype=complex)
    w2 = np.zeros((kzvec.size,kyvec.size),dtype=complex)
    
    # Initial values to be used in iteration:
    wr = 0.0
    wi = 0.0

    itera = 0
    maxit = 5000
    tol = 1e-6

    pbar = tqdm(total=len(kzvec))
    for kzi, kz in enumerate(kzvec):
        print("Solving for kz = %.1e"%kz)
        pbar.update(1)
        for kyi, ky in enumerate(kyvec):
            wplus = 0.
            wminus = 0.

            itera = 0
            while itera < 10 or (delta > tol and itera <= maxit):
                itera += 1
                wold = wplus
               
                gval = g((wr+1j*wi-ky*v0)/wce, ky**2*m/wce**2, kz**2*m/wce**2, 15)
                gin = np.imag(gval) 
                grn = np.real(gval)
                hn = 1. + ky**2 + kz**2 + grn
                wr = 1./np.sqrt(2)*np.sqrt(ky**2+kz**2)/np.sqrt(hn**2 + gin**2)*np.sqrt(hn + np.sqrt(hn**2+gin**2))
                wi = 1./np.sqrt(2)*np.sqrt(ky**2+kz**2)/np.sqrt(hn**2 + gin**2)*np.sqrt(-hn + np.sqrt(hn**2+gin**2))
                
                wplus = wr + np.sign(-gin)*1j*wi
                wminus = -wr - np.sign(-gin)*1j*wi
                
                delta = np.abs(wplus - wold)
                if np.abs(wplus) > 0:
                    delta = delta/np.abs(wplus)

            #w_new = np.sqrt(3.*kz**2+ky**2)*vti/np.sqrt()
            
            w1[kzi][kyi] = wplus
            w2[kzi][kyi] = wminus
            wr = 0.0
            wi = 0.0
    
    pbar.close()

    if save:
        np.save("w1", w1)
        np.save("w2", w2)
        print("Solutions saved.")

    return w1, w2

def plot_kz_ky(w):

    # save omega imag pic:
    plt.figure(0)

    plt.imshow(np.imag(w), origin="lower", aspect="auto", extent=[kyvec_cyc[0], kyvec_cyc[-1], kzvec_cyc[0], kzvec_cyc[-1]], interpolation="bilinear", cmap="jet")

    plt.colorbar()
    plt.xlabel(r"$k_y / k_0$", fontsize=20)
    plt.ylabel(r"$k_z / k_0$", fontsize=20)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    filename = "omega_im_kz_ky"
    filepath = os.path.join("results", filename)

    plt.savefig(filepath+".pdf", format='pdf', dpi=300)
    plt.savefig(filepath+".png", format='png', dpi=150)

    os.system("sxiv "+filepath+".png")
    
    # save omega imag pic:
    plt.figure(1)

    plt.imshow(np.real(w), origin="lower", aspect="auto", extent=[kyvec_cyc[0], kyvec_cyc[-1], kzvec_cyc[0], kzvec_cyc[-1]], interpolation="bilinear", cmap="jet")

    plt.colorbar()
    plt.xlabel(r"$k_y / k_0$", fontsize=20)
    plt.ylabel(r"$k_z / k_0$", fontsize=20)
    #plt.ylabel(r"$\gamma / \omega_{pi}$", fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    filename = "omega_re_kz_ky"
    filepath = os.path.join("results", filename)

    plt.savefig(filepath+".pdf", format='pdf', dpi=300)
    plt.savefig(filepath+".png", format='png', dpi=150)

    os.system("sxiv "+filepath+".png")

def plot_im_re_sep(w):

    # save omega real pic:
    plt.figure(2)
    for i in range(len(kzvec)):
        plt.plot(kyvec_cyc, np.imag(w[i]), lw=2.0, label=r"$k_z \lambda_D = %.5f$" % kzvec[i])
        #plt.plot(kyvec_cyc/(k0*rho_e), np.imag(w[i])*wpi, lw=2.0, label=r"$k_z \lambda_D = %.5f$"%kzvec[i])
        #plt.plot(kyvec_cyc, np.real(w[i]), lw=2.0, label=r"$k_z \lambda_D = %.5f$"%kzvec[i])
    plt.gca().set_ylim(bottom=0.0)
    #plt.xlabel(r"$k_y/k_0$", fontsize=20)
    plt.xlabel(r"$k/k_0$", fontsize=20)
    plt.ylabel(r"$\gamma/\omega_{pi} $", fontsize=20)
    #plt.ylabel(r"$\omega_r / \omega_{pi}$", fontsize=20)

    plt.grid()
    #plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    filename_re = "omega_im-x"
    filepath = os.path.join("results", filename_re)

    plt.savefig(filepath+".pdf", format='pdf')#, dpi=300)
    plt.savefig(filepath+".png", format='png')#, dpi=150)

    # save omega imag pic:
    #plt.figure()
   # for i in range(len(kzvec)):
       # plt.plot(kyvec_cyc*L/(2*np.pi*rho_e), np.real(w[i])*wpi, lw=2.0, label=r"$k_z \lambda_D = %.5f$"%kzvec[i])

    #plt.gca().set_ylim(bottom=0.0)
    #plt.xlabel(r"$k_y \rho_e$" , fontsize=20)

   # plt.ylabel(r"$\omega $", fontsize=20)

    plt.grid()
    #plt.legend(fontsize=12)

    filename_im = "omega_re-x"
    filepath = os.path.join("results", filename_im)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    plt.savefig(filepath+".pdf", format='pdf')#, dpi=300)
    plt.savefig(filepath+".png", format='png')#, dpi=150)

    # open pics:
    os.system("sxiv "+os.path.join("results", filename_im)+".png")
    os.system("sxiv "+os.path.join("results", filename_re)+".png")


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


if __name__ == "__main__":
    main()
####################################################################
#kyvec_cyc = np.linspace(0.001, 17., 800)
# kyvec = kyvec_cyc*k0*l_d
# kyvec = kyvec_cyc*l_d/rho_e
kyvec = np.arange(0, 300) * 2 *np.pi * l_d / L
kyvec_cyc = kyvec * rho_e / l_d

# kzvec_cyc = np.linspace(0.0001, 0.3, 10)
# kzvec = kzvec_cyc*k0*l_d

kzvec = np.array([0.00001])


# print(kzvec)


# kzvec_cyc = kzvec/l_d/k0

def main():
    w1, w2 = CVLR_solve(newsol=1.0, save=True)
    for i in range(len(kzvec)):
        w3 = np.array(np.imag((w1[i])) * wpi)
        w4 = np.array(np.real((w1[i])) * wpi)
        np.savetxt("lnimagE=20000Te=20,ln=-0.0005.txt", w3, fmt="%0.5e")
        np.savetxt("lnrealE=20000Te=20,ln=-0.0005.txt", w4, fmt="%0.5e")
    plot_im_re_sep(w1)
    # plot_kz_ky(w1)


def Z(x):
    """Plasma dispersion function"""

    return np.sqrt(np.pi) * 1j * special.wofz(x)


def dZdx(x):
    """1st derivative of plasma dispersion function"""

    return -2.0 * (1 + x * Z(x))


def d2Zdx2(x):
    """2nd derivative of plasma dispersion function"""

    zval = Z(x)
    return (-2. + 4. * x ** 2) * (-4. + zval - 4. * x * zval)


def gsum(Omega, X, Y, N):
    """Partial sum from Gordeev function"""

    term = 0.0
    for i in range(-N, N + 1):  # [-N,...,0,...,N]
        term += Z((Omega - i) / np.sqrt(2 * Y)) * special.iv(i, X)
    return term


def g(Omega, X, Y, N):
    """Gordeev function, uses gsum(Omega,X,Y,N) for partial sum.
    N should be less than ~15, as Z(x) starts to take arguments
    outside its domain."""

    return (Omega) / np.sqrt(2 * Y) * np.exp(-X) * gsum(Omega, X, Y, N)


def CVLR_solve(newsol=True, save=False):
    if not newsol:
        w1 = np.load("w1.npy")
        w2 = np.load("w2.npy")
        return w1, w2

    # arrays for storing solutions, w_plus and w_minus, respectively:
    w1 = np.zeros((kzvec.size, kyvec.size), dtype=complex)
    w2 = np.zeros((kzvec.size, kyvec.size), dtype=complex)

    # Initial values to be used in iteration:
    wr = 0.0
    wi = 0.0

    itera = 0
    maxit = 5000
    tol = 1e-6

    pbar = tqdm(total=len(kzvec))
    for kzi, kz in enumerate(kzvec):
        print("Solving for kz = %.1e" % kz)
        pbar.update(1)
        for kyi, ky in enumerate(kyvec):
            wplus = 0.
            wminus = 0.

            itera = 0
            while itera < 10 or (delta > tol and itera <= maxit):
                itera += 1
                wold = wplus

                gval = g((wr + 1j * wi - ky * v0) / wce, ky ** 2 * m / wce ** 2, kz ** 2 * m / wce ** 2, 60)
                gin = np.imag(gval)
                grn = np.real(gval)
                hn = 1. + ky ** 2 + kz ** 2 + grn
                wr = 1. / np.sqrt(2) * np.sqrt(ky ** 2 + kz ** 2) / np.sqrt(hn ** 2 + gin ** 2) * np.sqrt(
                    hn + np.sqrt(hn ** 2 + gin ** 2))
                wi = 1. / np.sqrt(2) * np.sqrt(ky ** 2 + kz ** 2) / np.sqrt(hn ** 2 + gin ** 2) * np.sqrt(
                    -hn + np.sqrt(hn ** 2 + gin ** 2))

                wplus = wr + np.sign(-gin) * 1j * wi
                wminus = -wr - np.sign(-gin) * 1j * wi

                delta = np.abs(wplus - wold)
                if np.abs(wplus) > 0:
                    delta = delta / np.abs(wplus)

            # w_new = np.sqrt(3.*kz**2+ky**2)*vti/np.sqrt()

            w1[kzi][kyi] = wplus
            w2[kzi][kyi] = wminus
            wr = 0.0
            wi = 0.0

    pbar.close()

    if save:
        np.save("w1", w1)
        np.save("w2", w2)
        print("Solutions saved.")

    return w1, w2


def plot_kz_ky(w):
    # save omega imag pic:
    plt.figure(0)

    plt.imshow(np.imag(w), origin="lower", aspect="auto",
               extent=[kyvec_cyc[0], kyvec_cyc[-1], kzvec_cyc[0], kzvec_cyc[-1]], interpolation="bilinear", cmap="jet")

    plt.colorbar()
    plt.xlabel(r"$k_y / k_0$", fontsize=20)
    plt.ylabel(r"$k_z / k_0$", fontsize=20)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    filename = "omega_im_kz_ky"
    filepath = os.path.join("results", filename)

    plt.savefig(filepath + ".pdf", format='pdf', dpi=300)
    plt.savefig(filepath + ".png", format='png', dpi=150)

    os.system("sxiv " + filepath + ".png")

    # save omega imag pic:
    plt.figure(1)

    plt.imshow(np.real(w), origin="lower", aspect="auto",
               extent=[kyvec_cyc[0], kyvec_cyc[-1], kzvec_cyc[0], kzvec_cyc[-1]], interpolation="bilinear", cmap="jet")

    plt.colorbar()
    plt.xlabel(r"$k_y / k_0$", fontsize=20)
    plt.ylabel(r"$k_z / k_0$", fontsize=20)
    # plt.ylabel(r"$\gamma / \omega_{pi}$", fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    filename = "omega_re_kz_ky"
    filepath = os.path.join("results", filename)

    plt.savefig(filepath + ".pdf", format='pdf', dpi=300)
    plt.savefig(filepath + ".png", format='png', dpi=150)

    os.system("sxiv " + filepath + ".png")


def plot_im_re_sep(w):
     #save omega real pic:
    #plt.figure(2)
    for i in range(len(kzvec)):
        plt.plot(kyvec_cyc / (k0 * rho_e), np.imag(w[i]), '.',color='red', lw=2.0,label=r'$\gamma$-theory')

                 #label=r"$k_z \lambda_D = %.5f$" % kzvec[i])
        # plt.plot(kyvec_cyc, np.real(w[i]), lw=2.0, label=r"$k_z \lambda_D = %.5f$"%kzvec[i])
    #plt.gca().set_ylim(bottom=0.0)
    #plt.xlabel(r"$m$", fontsize=20)

    #plt.ylabel(r"$\gamma, [1/s]$", fontsize=20)
    # plt.ylabel(r"$\omega_r / \omega_{pi}$", fontsize=20)

    plt.grid()
   #plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    filename_re = "omega_im-x"
    filepath = os.path.join("results", filename_re)

    plt.savefig(filepath + ".pdf", format='pdf')  # , dpi=300)
    plt.savefig(filepath + ".png", format='png')  # , dpi=150)

    # save omega imag pic:
    #plt.figure()
    #for i in range(len(kzvec)):
    #    plt.plot(kyvec_cyc * L / (2 *np.pi * rho_e), np.real(w[i]) * wpi, lw=2.0,
    #             label=r"$k_z \lambda_D = %.5f$" % kzvec[i])

    #plt.gca().set_ylim(bottom=0.0)
    #plt.xlabel(r"$k_y \rho_e$", fontsize=20)

    #plt.ylabel(r"$\omega$", fontsize=20)

    plt.grid()
    #plt.legend(fontsize=12)

    filename_im = "omega_re-x"
    filepath = os.path.join("results", filename_im)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    plt.savefig(filepath + ".pdf", format='pdf')  # , dpi=300)
    plt.savefig(filepath + ".png", format='png')  # , dpi=150)

    # open pics:
    os.system("sxiv " + os.path.join("results", filename_im) + ".png")
    os.system("sxiv " + os.path.join("results", filename_re) + ".png")


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x == 0 else x for x in values]


if __name__ == "__main__":
    main()

#DataVlasov=load('m_Gammas.npy')
#mode=DataVlasov[:,0]
#growth=DataVlasov[:,1]
#plt.plot(2*mode*np.pi/(k0*L),growth*1e9,'g *')
mode=np.array([29,30,31,32,50,51,52,53,54,55,56,75,76,77,78,79,100,101,102,103,125,126,127,128,150,151,152,175,176,200,201,225,226,250,275])
mode=np.array([29,30,31,32,50,51,52,53,54,55,56,75,76,77,78,79])
#growth10000=np.array([0.584,0.439,0.949,0.362,0.158,0.141,0.174,0.335,0.408,0.547,0.101,0.102,0.124,0.358,0.334,0.571,0.0724,0.209,0.389,0.470,0.156,0.504,0.523,0.283,0.311,0.405,0.380,0.960,0.318,0.554,0.336,0.161,0.192,0.174,0.0621])*1e8
growth10000=np.array([0.584,0.439,0.949,0.362,0.158,0.141,0.174,0.335,0.408,0.547,0.101,0.102,0.124,0.358,0.334,0.571])*1e8
#growth100000=np.array([0.510,0.640,0.527,0.312,0.571,0.723,0.734,0.475,0.290,0.539,0.366,0.519,0.386,0.731,0.474,0.606,0.267,0.236,0.534,0.577,0.261,0.356,0.660,0.292,0.694,0.388,0.391,1.24,0.410,0.595,0.338,0.189,0.136,0.135,0.0631])*1e8
growth_ioncold=np.array([0.618,0.456,1.28,0.351,0.382,0.529,0.546,0.434,0.819,0.864,1.30,0.121,0.149,0.351,0.695,0.594])*1e8
growth_ioncoldgraham=np.array([0.415,0.393,0.958,0.387,0.168,0.112,0.247,0.242,0.492,0.558,1.05,0.367,0.259,0.703,0.691,0.577])*1e8
#plt.xlim(0,140)

#plt.plot(mode,growth100000,'m s',markersize=5,label='$\gamma$-100000ppc')
#plt.plot(mode,growth_ioncoldgraham,'s', color='purple',markersize=5,label='$\gamma$-10000ppc-$T_{ion}0.02eV$')
#plt.plot(mode,growth10000,'r *',label='$\gamma$-10000ppc-$T_{ion}0.2eV$')
plt.legend()
#plt.ylim(0,1.06e8)
