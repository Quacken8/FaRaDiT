#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Set of functions that return opacities for a bunch of situations
"""
import matplotlib
import numpy as np

def ZHU_unvec(rho = 0, T = 0, return_mode = False): #numpy-friendly function actually defined lower
    """Zhu etal. (2012), http://www.physics.unlv.edu/~zhzhu/Opacity.html"""

    xlp = np.log10(rho*T*8.314472*1.e7/2.4)
    xlt = np.log10(T)

    def kappa_ZHU1(rho,T):  return 1.5*(xlt -1.16331)-0.736364					# water ice grains
    def kappa_ZHU2(rho,T):  return -3.53154212*xlt +8.767726-(7.24786-8.767726)*(xlp  -5.)/16.	# water ice evaporation
    def kappa_ZHU3(rho,T):  return 1.5*(xlt -2.30713)+0.62 						# non-water grains
    def kappa_ZHU4(rho,T):  return -5.832*xlt +17.7							# graphite corrosion
    def kappa_ZHU5(rho,T):  return 2.129*xlt -5.9398							# no-graphite grains
    def kappa_ZHU6(rho,T):  return 129.88071-42.98075*xlt +(142.996475-129.88071)*0.1*(xlp  +4)	# grain evaporation
    def kappa_ZHU7(rho,T):  return -15.0125+4.0625*xlt 						# water vapour
    def kappa_ZHU8(rho,T):  return 58.9294-18.4808*xlt +(61.6346-58.9294)*xlp  /4.		# water dissociation
    def kappa_ZHU9(rho,T):  return -12.002+2.90477*xlt +(xlp  -4)/4.*(13.9953-12.002)		# molecules
    def kappa_ZHU10(rho,T): return  -39.4077+10.1935*xlt +(xlp  -4)/2.*(40.1719-39.4077)	# H scattering
    def kappa_ZHU11(rho,T): return  17.5935-3.3647*xlt +(xlp  -6)/2.*(17.5935-15.7376)		# bound-free, free-free
    def kappa_ZHU12(rho,T): return  -0.48								# e- scattering
    def kappa_ZHU13(rho,T): return  3.586*xlt -16.85							# molecules and H scattering

    xlop = None
    if xlt  < 2.23567+0.01899*(xlp-5.):
        if return_mode:  mode = 1
        xlop =  kappa_ZHU1(rho,T)
    elif xlt  < 2.30713+0.01899*(xlp-5.):
        if return_mode:  mode = 1
        xlop =  kappa_ZHU2(rho,T)
    elif xlt  < (17.7-0.62+1.5*2.30713)/(1.5+5.832):
        if return_mode:  mode = 2
        xlop =  kappa_ZHU3(rho,T)
    elif xlt  < (5.9398+17.7)/(5.832+2.129):
        if return_mode:  mode = 3
        xlop =  kappa_ZHU4(rho,T)
    elif xlt  < (129.88071+5.9398 + (142.996475-129.88071)*0.1*(xlp+4))/(2.129+42.98075):
        if return_mode:  mode = 4
        xlop =  kappa_ZHU5(rho,T)
    elif xlt  < (129.88071+15.0125 + (142.996475-129.88071)*0.1*(xlp+4))/(4.0625+42.98075):
        if return_mode:  mode = 5
        xlop =  kappa_ZHU6(rho,T)
    elif xlt  < 3.28+xlp/4.*0.12:
        if return_mode:  mode = 6
        xlop =  kappa_ZHU7(rho,T)
    elif xlt  < 3.41+0.03328*xlp/4.:
        if return_mode:  mode = 7
        xlop =  kappa_ZHU8(rho,T)
    elif xlt  < 3.76+(xlp-4)/2.*0.03:
        if return_mode:  mode = 8
        xlop =  kappa_ZHU9(rho,T)
    elif xlt  < 4.07+(xlp-4)/2.*0.08:
        if return_mode:  mode = 9
        xlop =  kappa_ZHU10(rho,T)
    elif xlt  < 5.3715+(xlp-6)/2.*0.5594:
        if return_mode:  mode = 10
        xlop =  kappa_ZHU11(rho,T)
        if return_mode:  mode = 11
    else:
        xlop =  kappa_ZHU12(rho,T)
        if return_mode:  mode = 12

    if xlt < 4: mode = 13
    xlop_ = (kappa_ZHU13(rho,T) if xlt <4. else xlop) if xlop < 3.586*xlt-16.85 else xlop

    if return_mode: return mode
    return np.power(10, xlop_)

ZHU = np.vectorize(ZHU_unvec)

def main():
    
    # making sure the plot looks how its supposed to
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    rhos = np.logspace(-10, -3)
    Ts = np.logspace(1, 4)*5

    meshrhos, meshTs = np.meshgrid(rhos, Ts)
    data = ZHU(rho=meshrhos, T = meshTs)
    colors = ["darkred", "red", "orange", "lightorange", "yellow", "lightgreen", "green", "cyan", "lightblue", "blue", "darkblue", "violet", "lightviolet"]
    data_colors = ZHU(rho=meshrhos, T = meshTs, return_mode = True)
    print(data_colors)
    from matplotlib import cm
    colors = cm.jet(data_colors/13)
    logrho = np.log10(meshrhos)
    logT = np.log10(meshTs)
    logdata = np.log10(data)
    ax.plot_surface(logrho, logT, logdata, facecolors = colors)#, cmap = matplotlib.colors.ListedColormap(colors))
    plt.show()
    
    pass

if __name__ == "__main__":
    main()
