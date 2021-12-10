#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FaRaDiT.py
Disk module to handle Fargo a Radmc programs.

References: ???
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from radmc3dPy import image
from scipy.optimize import curve_fit
from scipy import integrate
import os
import time

G = 6.67430e-11     #[SI]  CODATA 2018 
au = 1.495978707e11 #[m]  IAU 2012
M_sun = 1.9884e30   #[kg]  ASA 2018
cm = 1e-2
gram = 1e-3
R_gas = 8.314462618     #[J·K⁻¹·mol⁻¹] CODATA 2018
mu = 0.0024             #Mean molecualr weight in kg·mol⁻¹
planck = 6.62607015e-34 #[J·s] CODATA 2018 
k_B = 1.380649e-23      #[J·K⁻¹] CODATA 2018 
light_speed = 299792458 #[m·s⁻¹] https://www.bipm.org/fr/measurement-units/si-defining-constants ???
steff = 5.670374419e-8  #[W·m¯²K¯⁴] CODATA 2018

__author__ = "Ondrej Janoska"
__version__ = "May 29th 2021"

def is_number(s):
    """Returns True if input is a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def polynomial_smooth_min(a, b, smoothing = 1):
    """
    Returns the smaller value between a and b in such a way that if a = f(x) and b = g(x), smooth_min(f, g)(x) will be smooth. 
    """
    h = np.clip( 0.5+0.5*(b-a)/smoothing, 0.0, 1.0 )
    return a*h+b*(1-h)-smoothing*h*(1-h)

def exponential_smooth_min(a, b, smoothing = 1):
    res = np.exp2(-smoothing*a)+np.exp2(-smoothing*b)
    return -(np.log2(res))/smoothing

def smooth_min(a, b, smoothing = 1):
    """
    For two positive functions a,b ∈ 𝓒ⁿ, returns a function f that's at worst 𝓒ⁿ¯¹ such that f≤min{a,b}

    (wolfram input:
    d/dx((f[x]^n*g[x]^n/(f[x]^n+g[x]^n))^(1/n))//FullSimplify
    )
    """
    a, b = np.power(a, smoothing), np.power(b, smoothing)
    return np.power((a*b)/(a+b), 1.0/smoothing)

def interpolate_linear(x1, x2, v, x0):
    """
    Turns out numpy's interp() is much faster and simpler. I should've seen that coming
    Lineary interpolates v between point 
    (x1[0], x1[0]) where v=v[0] and 
    (x2[1], x2[1]) where v=v[1] and 
    returns v0 at point (x0[0], x0[1])
    
    x0 has to be on the line defined by the two points
    """
    if np.array_equal(x1, x2) and (v[0] != v[1]): raise ValueError("Interpolate error: the same point was put in twice but with different values!")
    if np.array_equal(x1, x2) and (x0 != x1): raise ValueError("Interpolate error: the same point was put in twice, but it's not x0; can't interpolate!")
    if np.array_equal(x1, x2) and (x0 == x1): return v[0]
    slope = v[1]-v[0]
    whole_interval = np.linalg.norm(np.asarray(x2)-np.asarray(x1))
    partial_interval = np.linalg.norm(np.asarray(x0)-np.asarray(x1))
    return v[0] + slope*partial_interval/whole_interval

def interpolate_bilinear(x1, x2, x3, v, x0):
    """
    Lineary interpolates v using three points
    x1, x2, x3 with defined values v[0], v[1] and v[2] respectively
    returns v0 at point (x0[0], x0[1])
    
    x0 has to be in the plane defined by the three points

            x2
            /
           /
          /
         /
   tmp_x0_
       /   — _
      /        x0_
     /             — _
    /                  — _
   x1                      x3
    """
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    v30 = x0-x3
    v12 = x2-x1
    t = np.linalg.norm(np.cross(x1-x3,v12))/np.linalg.norm(np.cross(v30,v12))
    tmp_x0 = x3+t*v30
    tmp_v0 = interpolate_linear(x1, x2, v[0:2], tmp_x0)
    return interpolate_linear(tmp_x0, x3, [tmp_v0, v[2]], x0)

def interpolate_trilinear(x0, x1, x2, x3, v, y):
    """
    
    """
    x0, x1, x2, x3, y = np.asarray(x0), np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(y)
    
    v3y = (y-x3)/np.linalg.norm(y-x3)
    v01 = x1-x0
    v12 = x2-x1
    v23 = x3-x2
    n = np.cross(v01, v12)/np.linalg.norm(np.cross(v01, v12))
    w = x3-x1
    si = -n.dot(v23)/n.dot(v3y)
    tmpx = w+si*v3y+x1
    
    tmpv = interpolate_bilinear(x0, x1, x2, v[0:3], tmpx)
    return interpolate_linear(tmpx, x3, [tmpv, v[3]], y)

def pol_to_cart(pol, rmin=2.8, rmax=14, dr=(14-2.8)/1024, dphi=2*np.pi/1536, fine = 1001, D2 = True):
    """Old and pretty lame of turning 2D grid in polar coordinates into one with cartesian ones. Obsolete"""
    cart = []
    y = -rmax
    dx = 2*rmax/fine
    v = [0,0,0]
    for _ in range(fine):
        x = -rmax
        tmp = []
        for _ in range(fine):
            r = np.sqrt(x*x+y*y)
            if r < rmin or r > rmax:
                tmp.append(0)
                x+= dx
                continue
            r2 = r+dr
            phi = np.arctan(y/x)
            phi2 = phi+dphi
            if phi2 >= 2*np.pi: phi2 = phi2 % 2*np.pi
            
            point1 = [np.floor(r/dr)*np.cos(phi/dphi), np.floor(r/dr)*np.sin(phi/dphi), 0]

            v[0] = pol[int(np.floor(r/dr)),int(np.floor(phi/dphi))]
            point2 = [np.floor(r2/dr)*np.cos(phi/dphi), np.floor(r2/dr)*np.sin(phi/dphi), 0]
            try: v[1] = pol[int(np.floor(r2/dr)),int(np.floor(phi/dphi))]
            except IndexError: v[1] = pol[int(np.floor(r/dr)),int(np.floor(phi/dphi))]
            point3 = [np.floor(r/dr)*np.cos(phi2/dphi), np.floor(r/dr)*np.sin(phi2/dphi), 0]
            v[2] = pol[int(np.floor(r/dr)),int(np.floor(phi2/dphi))]

            tmp.append(interpolate_bilinear(point1, point2, point3, v, [x,y,0]))
            
            x+=dx
        cart.append(tmp)
        y+=dx
    if D2:
       return cart
    return np.ravel(cart)

def f_z(h=0, sigma=0, T = 0, equatorial_r=0, M_star=M_sun, getH = False): #[SI]
    """
    Returns volumetric density. Vertically isothermal model according to Chrenko etal. (2017), A&A, 606, A114.
    --------------------
    Parameters:
    -------------------- 
    h [m]: float, height of the point of interest above the disk equatorial plane
    
    sigma [kg/m²]: float, surface density of the dust at the point in equatorial plane under the point of interest

    T [K]: float, temperature of the dust at the point in equatorial plane under the point of interest

    equatorial_r [m]: float, distance from the star to the point in equatorial plane under the point of interest
            
    M_star [kg]: float, mass of the star at the center

    getH: bool, if True, returns the scale height in [m] instead of density
    """

    H = np.sqrt(R_gas*T*equatorial_r**3/(G*M_star*mu))
    if getH: return H
    return sigma/np.sqrt(2*np.pi)/H*np.exp(-0.5*h*h/H/H)

def index_before(array = None, value = 0):
    """Returns the last index "i" at which monotically ascending "array" is smaller than "value" """
    if array is None:
        raise ValueError("index_around error: No array was passed!")
    i = 0
    while (i<len(array)) and (array[i] <= value):
        i+=1
    return i

def is_file_binary(filename):
    try:
        with open(filename, "r") as f:
            for l in f:
                l.encode("utf-8")
        return False
    except UnicodeDecodeError:
        return True

def black_body(wavelength, T = 4000):
    """
    Idealized blackbody spectrum a.k.a. Planck's law.
    reference: Harmanec, Brož, Stavba a vývoj hvězd
    """
    return 2*planck*light_speed**2/(wavelength**5*(np.exp(planck*light_speed/(wavelength*k_B*T))-1))

def distribute_thet(th):
    """
    Function for better distribution of theta cells such that few cells are used in the midplane (where optical depth is too large) and in the high atmosphere (where there isn't much of anything)
    TBD
    """
    th/=np.pi*0.5
    return (np.power(th, 10)+np.power(th, 1/10))*0.25*np.pi
    

class Disk():
    """Disk class to handle Fargo input/Radmc output."""

    def __init__(self):
        """
        Initialize disk.
        """

        self.density = np.array([]) #PŘIDAT VHODNÁ SYNTETICKÁ POLE
        self.T = np.array([])
        self.par = {
        'Nrad' : 1024,  #Number of radial sectors compatible with fargo input
        'Nsec' : 1536,  #Number of angular sectors
        'Nthet': 0,     #Number of sectors in the theta direction
        'Thetamin' : 0,
        'Thetamax' : np.pi,
        'grid_type' : 100, #grid_type: 0 ⇒ cartesian, 100 ⇒ spherical
        'M_star' : np.array([2.1*M_sun]), #Mass of the star(s) in kg
        'Star_temp': 4000, #surface temperature of the star in K
        }

        self.Rmin = 2.8   #Boundary of the disk in au
        self.Rmax = 14    # —||—­
        self.Phimin = 0
        self.Phimax = 2*np.pi

        self.dtheta = 0
        self.dphi = (self.Phimax - self.Phimin)/self.par["Nsec"]
        self.dr = (self.Rmax-self.Rmin)/self.par["Nrad"]

        self.r_inf = np.linspace(self.Rmin, self.Rmax-self.dr, self.par["Nrad"])
        self.r_med = np.linspace(self.Rmin+0.5*self.dr, self.Rmax-0.5*self.dr, self.par["Nrad"])
        self.r_sup = np.linspace(self.Rmin+self.dr, self.Rmax, self.par["Nrad"])


        self.phi_inf = np.linspace(self.Phimin, self.Phimax-self.dphi, self.par["Nsec"])
        self.phi_med = np.linspace(self.Phimin+0.5*self.dphi, self.Phimax-0.5*self.dphi, self.par["Nsec"])
        self.phi_sup = np.linspace(self.Phimin+self.dphi, self.Phimax, self.par["Nsec"])

        self.theta_inf = None
        self.theta_med = None
        self.theta_sup = None
    
    ##Debugging procedures

    def print_params(self):
        """
        Prints out self.par dictionary containing basic parameters of disk
        """
        for name, value in self.par.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                print(name, '\tIs present as a list of ', len(value), ' values')
            else:
                if is_number(str(value)):
                    print(name, '\t', '{:.2e}'.format(value))
                else:
                    print(name, '\t', value)

    def plot_rho_slice(self, phi = 0, thickness = f_z, fine = 100, maxheight = None, logarithmic_scale=False, note = "", show = True, fig = None):
        """
        Creates 2D colormesh vertical plot of the disk's volumetric density along a ray coming from the star
        --------------------
        Parameters:
        -------------------- 
        phi [deg]: float, angular coordinate of the ray along which the plot is to be drawn

        thickness: function, by which the volumetric density around the disk is determined. It is expected to take parameters of the point above which the density is needed:
            h [m]: float, height of the point of interest above the disk
            equatorial_r [m]: float, distance from the star to the point in equatorial plane under the point of interest
            T [K]: float, temperature of the dust at the point in equatorial plane under the point of interest
            sigma [kg/m²]: float, surface density of the dust at the point in equatorial plane under the point of interest
            M_star [kg]: float, mass of the star at the center

        fine: int, resolution in the vertical direction

        maxheight [au]: total height of the plot

        logarithmic_scale: bool, if True, log scale will be used on the color (z) axis
        
        r_range: array-like, list containing boundaries of desired graph, 
        
        note: string, note to be written under the title,
        
        show: bool, if set to True the plot will be immediatelly shown. Otherwise plt.show() or other method with show = True has to be used later

        fig: matplotlib figure, figure to which draw the plot

        """
        if fig is None: fig = plt.figure()

        index = index_before(array = self.phi_med, value = phi*np.pi/180)
        if self.density.ndim == 2:
            if maxheight is None: maxheight = 2
            sigma_seed = self.density[:,index]
            T_seed = self.T[:,index]
            rho_slice = np.array([[]])
            dz = maxheight/(2*fine+1)
            for i in range(-fine,fine+1):
                stack = [thickness(sigma = sigma_seed[j], T = T_seed[j], h = i*dz*au, equatorial_r = self.r_med[j]*au) for j in range(len(sigma_seed))]
                if (i == -fine): rho_slice = stack
                else: rho_slice = np.vstack((rho_slice, stack))
            equatorial_r = np.insert(self.r_sup, 0, self.r_inf[0])
            z = np.linspace(-fine*dz, fine*dz, 2*fine+2)
            r_mesh, z_mesh = np.meshgrid(equatorial_r,z)
            ax = fig.add_subplot(111)
            if logarithmic_scale: 
                import matplotlib.colors
                ap = ax.pcolormesh(r_mesh,z_mesh,rho_slice, norm=matplotlib.colors.LogNorm(), cmap = "hot")
            else: ap = ax.pcolormesh(r_mesh ,z_mesh, rho_slice, cmap = "hot")

        elif self.density.ndim == 3:
            if maxheight is None: maxheight = 0#len(self.theta_med)
            else: maxheight = (len(self.theta_med)-maxheight//2)
            density = np.append(self.density[:,maxheight:,index], self.density[:,maxheight:,index][:,::-1], axis = 1)
            theta_bound = np.append(self.theta_inf[maxheight:], self.theta_sup[-1])
            theta_bound = np.append(theta_bound, np.pi-self.theta_inf[maxheight:][::-1])

            r_mesh, theta_mesh = np.meshgrid(np.append(self.r_inf, self.r_sup[-1]), theta_bound)
            x_mesh = r_mesh*np.sin(theta_mesh)
            y_mesh = r_mesh*np.cos(theta_mesh)

            ax = fig.add_subplot(111)
            if logarithmic_scale: 
                import matplotlib.colors
                ap = ax.pcolormesh(x_mesh, y_mesh, density.T, norm=matplotlib.colors.LogNorm(), cmap = "hot")
            else: ap = ax.pcolormesh(x_mesh, y_mesh, density.T, cmap = "hot")
        else: raise ValueError("Strange dimension of density array")

        plt.xlabel("r [au]")
        plt.ylabel("z [au]")
        plt.title(f"Gas density at φ = {phi}°\n{note}")
        plt.colorbar(ap, label = "kg·m⁻³")
        if show: plt.show()

    def plot_T_slice(self, phi = 0, fine = 100, maxheight = None, logarithmic_scale=False, note = "", show = True, fig = None):
        """
        Creates 2D colormesh vertical plot of the disk's temperature along a ray coming from the star
        --------------------
        Parameters:
        -------------------- 
        phi [deg]: float, angular coordinate of the ray along which the plot is to be drawn

        fine: int, resolution in the vertical direction

        maxheight [au]: total height of the plot

        logarithmic_scale: bool, if True, log scale will be used on the color (z) axis
        
        r_range: array-like, list containing boundaries of desired graph, 
        
        note: string, note to be written under the title,
        
        show: bool, if set to True the plot will be immediatelly shown. Otherwise plt.show() or other method with show = True has to be used later
        
        fig: matplotlib figure, figure to which draw the plot
        """
        if fig is None: fig = plt.figure()

        index = index_before(array = self.phi_med, value = phi*np.pi/180)

        if self.T.ndim == 3:
            if maxheight is None: maxheight = 0#len(self.theta_med)
            else: maxheight = (len(self.theta_med)-maxheight//2)
            Temp = np.append(self.T[:,maxheight:,index], self.T[:,maxheight:,index][:,::-1], axis = 1)
            theta_bound = np.append(self.theta_inf[maxheight:], self.theta_sup[-1])
            theta_bound = np.append(theta_bound, np.pi-self.theta_inf[maxheight:][::-1])

            r_mesh, theta_mesh = np.meshgrid(np.append(self.r_inf, self.r_sup[-1]), theta_bound)
            x_mesh = r_mesh*np.sin(theta_mesh)
            y_mesh = r_mesh*np.cos(theta_mesh)

            ax = fig.add_subplot(111)
            if logarithmic_scale: 
                import matplotlib.colors
                ap = ax.pcolormesh(x_mesh, y_mesh, Temp.T, norm=matplotlib.colors.LogNorm(), cmap = "hot")
            else: ap = ax.pcolormesh(x_mesh, y_mesh, Temp.T, cmap = "hot")
        else: raise ValueError("Strange dimension of temperature array")

        plt.xlabel("r [au]")
        plt.ylabel("z [au]")
        plt.title(f"Gas temperature at φ = {phi}°\n{note}")
        plt.colorbar(ap, label = "K")
        if show: plt.show()

    def plot_fargo_gastemper(self, Mean = True, note = "", show = True):
        """
        Creates 2D colormesh plot of the disk's equatorial temperature
        --------------------
        Parameters:
        -------------------- 
        mean: bool, if True, the radial gradient is substracted from the temperature array so fine structure can be observed
        
        note: string, note to be written under the title,
        
        show: bool, if set to True the plot will be immediatelly shown. Otherwise plt.show() or other method with show = True has to be used later
        """
        phi = np.insert(self.phi_sup, 0, self.phi_inf[0])
        equatorial_r = np.insert(self.r_sup, 0, self.r_inf[0])
        phi_mesh, r_mesh = np.meshgrid(phi, equatorial_r)
        x = r_mesh*np.cos(phi_mesh)
        y = r_mesh*np.sin(phi_mesh)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        mean = 0
        if Mean:
            mean = np.zeros((self.par["Nsec"]+1, self.par["Nrad"]+1))
            mean = [np.average(self.T, axis=1) for i in range(self.par["Nsec"])]
            mean = np.transpose(mean,(1,0))

        ap = ax.pcolormesh(x,y,self.T-mean)
        plt.title("Gas temperature" + Mean*" with subtracted radial gradient" + "\n"+note)
        plt.colorbar(ap, label = "K")
        if show: plt.show()
        
    def plot_fargo_gasdens(self, logarithmic_scale=False, note="", show = True):
        """
        Creates 2D colormesh plot of the disk's surface density
        --------------------
        Parameters:
        -------------------- 
        logarithmic_scale: bool, if True, log scale will be used on the color (z) axis
        
        note: string, note to be written under the title,
        
        show: bool, if set to True the plot will be immediatelly shown. Otherwise plt.show() or other method with show = True has to be used later
        """
        phi = np.insert(self.phi_sup, 0, self.phi_inf[0])
        equatorial_r = np.insert(self.r_sup, 0, self.r_inf[0])
        phi_mesh, r_mesh = np.meshgrid(phi, equatorial_r)
        x = r_mesh*np.cos(phi_mesh)
        y = r_mesh*np.sin(phi_mesh)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        if self.density.ndim == 2: sigma = self.density
        elif self.density.ndim == 3: sigma = self.density[:,(len(self.theta_med)-1),:]
        if logarithmic_scale: 
            import matplotlib.colors
            ap = ax.pcolormesh(x,y,sigma, norm=matplotlib.colors.LogNorm(), cmap = "hot")
        else: ap = ax.pcolormesh(x,y,sigma, cmap = "hot")
        plt.title(f"Gas density\n{note}")
        plt.xlabel("au")
        plt.ylabel("au")
        plt.colorbar(ap, label = "kg·m⁻²")
        if show: plt.show()

    def plot_density_profile(self, along = "ray", r = 4, phi = 0, logarithmic_scale=False, r_range = None,  theta_range = None, note = "", show = True, fig = None, label = None):
        """
        Plots surface density along a ray from the star
        --------------------
        Parameters:
        -------------------- 
        along: string, determines along which curve to plot the profile. Possibilities are:
            "ray" — line from star through the equatorial plane
            "perpendicular" — line perpendicular to the equatorial plane
            "arc" — curve in 3D spherical coordinates with θ being the variable

        phi [deg]: float, angular coordinate of the ray along which the plot is to be drawn

        r [au]: float, radial coordinate of the ray along which the plot is to be drawn (if arc == "arc" or "perpendicular")
        
        logarithmic_scale: bool, if True, log scale will be used on the y axis
        
        r_range [au]: array-like, list containing boundaries of desired graph (if arc == "arc" or "perpendicular"),

        theta_range [deg]: array-like, list containing boundaries of desired graph (if arc == "arc" or "perpendicular"),
        
        note: string, note to be written under the title,
        
        show: bool, if set to True the plot will be immediatelly shown. Otherwise plt.show() or other method with show = True has to be used later

        fig: pyplot figure, which fig the plot is supposed to be added to
        """

        if fig is None: plt.figure()
        else: plt.figure(fig.number)
        plt.subplot(111)

        phi_index = index_before(self.phi_med, phi/180*np.pi)

        if along.casefold() == "arc":
            if self.density.ndim != 3:
                raise ValueError("Density plot error: Density field is not 3D! Can't plot the third dimension!")
            if theta_range is None:
                start = 0
                end = len(self.theta_med)-1
            else:
                theta_range = np.deg2rad(theta_range)
                start = index_before(array = self.theta_med, value = theta_range[0])
                end = index_before(array = self.theta_med, value = theta_range[1])
            r_index = index_before(array = self.r_med, value = r)
            x = np.rad2deg(self.theta_med[start:end])
            data = self.density[r_index, start:end, phi_index]

            plt.title(f"Density of the disk along a θ arc\n at φ = {phi}° and r = {r} au"+f"\n{note}"*(len(note)>0))
            plt.xlabel("θ [deg]")
        
        elif along.casefold() == "perpendicular":
            from scipy.interpolate import griddata

            r_mesh, theta_mesh = np.meshgrid(self.r_med, self.theta_med)
            souradnice_sph = np.column_stack((r_mesh.ravel(), theta_mesh.ravel()))

            x = souradnice_sph[:,0]*np.sin(souradnice_sph[:,1])
            y = souradnice_sph[:,0]*np.cos(souradnice_sph[:,1])

            points = np.column_stack((x,y))
            temps =  self.density[:, :, phi_index].T.ravel()

            ys = np.linspace(-20, 20, num = 1000)
            xs = r*np.ones(shape = 1000)

            interp = griddata(points, temps, (xs,ys), method = "nearest")

            plt.title(f"Density of the disk along a line perpendcular to the disk\n at φ = {phi}° and r = {r} au"+f"\n{note}"*(len(note)>0))
            x = ys
            data = interp
            plt.xlabel("z [au]")

        elif along.casefold() == "ray":
            if r_range is None:
                start = 0
                end = len(self.density[:,phi_index])-1
            else:
                start = index_before(array = self.r_med, value = r_range[0])
                end = index_before(array = self.r_med, value = r_range[1])
            x = self.r_med[start:end]
            if self.density.ndim == 2:
                data = self.density[start:end,phi_index]
            else:
                theta_midplane = (len(self.theta_med)-1)
                data = self.density[start:end, theta_midplane, phi_index]

            plt.xlabel("r [au]")
            plt.title(f"Equatorial density of the disk at φ = {phi}°"+f"\n{note}"*(len(note)>0))
            
        else: raise ValueError(f"Density plot error: Unknow option '{along}'! Only accepted options are 'ray', 'perpendicular' and 'arc'")

        
        plt.plot(x, data, label = label)
        if logarithmic_scale: plt.yscale('log')
        if self.density.ndim == 3: plt.ylabel("ρ [kg·m⁻³]")
        elif self.density.ndim == 2: plt.ylabel("σ [kg·m⁻²]")
        else: raise ValueError(f"Strange dimension of density {self.density.ndim}, what's going on?")
        
        if show: plt.show()

    def plot_T_profile(self, along = "ray", r = 4, phi = 0, logarithmic_scale = False, r_range = None, theta_range = None, note = "", show = True, fig = None, label = None):
        """
        Plots equatorial temperature along a set curve
        --------------------
        Parameters:
        --------------------
        along: string, determines along which curve to plot the profile. Possibilities are:
            "ray" — line from star through the equatorial plane
            "perpendicular" — line perpendicular to the equatorial plane
            "arc" — curve in 3D spherical coordinates with θ being the variable

        phi [deg]: float, angular coordinate of the ray along which the plot is to be drawn

        r [au]: float, radial coordinate of the ray along which the plot is to be drawn (if arc == "arc" or "perpendicular")
        
        logarithmic_scale: bool, if True, log scale will be used on the y axis
        
        r_range [au]: array-like, list containing boundaries of desired graph (if arc == "arc" or "perpendicular"),

        theta_range [deg]: array-like, list containing boundaries of desired graph (if arc == "arc" or "perpendicular"),
        
        note: string, note to be written under the title,
        
        show: bool, if set to True the plot will be immediatelly shown. Otherwise plt.show() or other method with show = True has to be used later

        fig: pyplot figure, which fig the plot is supposed to be added to
        """

        if fig is None: plt.figure()
        else: plt.figure(fig.number)
        plt.subplot(111)

        phi_index = index_before(self.phi_med, phi/180*np.pi)

        if along.casefold() == "arc":
            if self.T.ndim != 3:
                raise ValueError("T plot error: Temperature field is not 3D! Can't plot the third dimension!")
            if theta_range is None:
                start = 0
                end = len(self.theta_med)-1
            else:
                theta_range = np.deg2rad(theta_range)
                start = index_before(array = self.theta_med, value = theta_range[0])
                end = index_before(array = self.theta_med, value = theta_range[1])
            r_index = index_before(array = self.r_med, value = r)
            x = np.rad2deg(self.theta_med[start:end])
            data = self.T[r_index, start:end, phi_index]

            plt.title(f"Temperature of the disk along a θ arc\n at φ = {phi}° and r = {r} au"+f"\n{note}"*(len(note)>0))
            plt.xlabel("θ [deg]")
        
        elif along.casefold() == "perpendicular":
            from scipy.interpolate import griddata

            r_mesh, theta_mesh = np.meshgrid(self.r_med, self.theta_med)
            souradnice_sph = np.column_stack((r_mesh.ravel(), theta_mesh.ravel()))

            x = souradnice_sph[:,0]*np.sin(souradnice_sph[:,1])
            y = souradnice_sph[:,0]*np.cos(souradnice_sph[:,1])

            points = np.column_stack((x,y))
            temps =  self.T[:, :, phi_index].T.ravel()

            ys = np.linspace(-20, 20, num = 1000)
            xs = r*np.ones(shape = 1000)

            interp = griddata(points, temps, (xs,ys), method = "nearest")

            #plt.scatter(x, y)
            #plt.plot(xs ,ys, color = "b")
            #plt.xlabel("r [au]")
            #plt.ylabel("z [au]")
            #plt.figure()
            plt.title(f"Temperature of the disk along a line perpendcular to the disk\n at φ = {phi}° and r = {r} au"+f"\n{note}"*(len(note)>0))
            x = ys
            data = interp
            #plt.plot(ys, interp)
            plt.xlabel("z [au]")
            #plt.ylabel("T [K]")

        elif along.casefold() == "ray":
            if r_range is None:
                start = 0
                end = len(self.density[:,phi_index])-1
            else:
                start = index_before(array = self.r_med, value = r_range[0])
                end = index_before(array = self.r_med, value = r_range[1])
            x = self.r_med[start:end]
            if self.T.ndim == 2:
                data = self.T[start:end,phi_index]
            else:
                theta_midplane = (len(self.theta_med)-1)
                data = self.T[start:end, theta_midplane, phi_index]

            plt.xlabel("r [au]")
            plt.title(f"Equatorial temperature of the disk at φ = {phi}°"+f"\n{note}"*(len(note)>0))
        else: raise ValueError(f"T plot error: Unknow option '{along}'! Only accepted options are 'ray', 'perpendicular' and 'arc'")

        
        plt.plot(x, data, label = label)
        if logarithmic_scale: plt.yscale('log')
        plt.ylabel("T [K]")
        
        if show: plt.show()

    def plot_grid(self, phi = 0, fig = None, show = True):
        """
        Plots points of cell centers or boundaries
        TBD
        """
        if fig is None: plt.figure()
        else: plt.figure(fig.number)
        
        theta_points = np.append(self.theta_med, np.pi-self.theta_med[::-1])
        r_mesh, theta_mesh = np.meshgrid(self.r_med, theta_points)
        souradnice_sph = np.column_stack((r_mesh.ravel(), theta_mesh.ravel()))
        x = souradnice_sph[:,0]*np.sin(souradnice_sph[:,1])
        y = souradnice_sph[:,0]*np.cos(souradnice_sph[:,1])

        plt.scatter(x,y, s = 0.5, label = "Gridpoint centers")
        plt.legend()
        plt.ylabel("z [au]")
        plt.xlabel("r [au]")
        if show: plt.show()

    def plot_scale_height(self, phi = 0, thickness = f_z, show = True, fig = None):
        """
        Plots the scaleheight of the disk
        TBD
        """
        phi_index = index_before(self.phi_med, phi/180*np.pi)
        if self.density.ndim == 2:
            x = self.r_med
            y = thickness(equatorial_r = x*au, sigma = self.density[:, phi_index], T = self.T[:, phi_index], M_star = self.par["M_star"][0], getH = True)/au
            if fig is None: plt.figure()
            else: plt.figure(fig.number)
            plt.plot(x,y, label = "Scaleheight")
            plt.ylabel("z [au]")
            plt.xlabel("r [au]")
            if show: plt.show()
        elif self.density.ndim == 3:
            raise ValueError("3D density scale height not yet supported, I'm lazy")

    def optical_depth(self, phi = 0, from_star = False, equatorial_r = 4, wav = 870, thickness = f_z, filename = None, message = True):
        """
        Calculates (and returns) optical depth along a line perpendicular to the equatorial plane of the disk at a chosen point or a line from the star through the midplane
        --------------------
        Parameters:
        -------------------- 
        phi [deg]: float, phi coordinate of the point of intersection of the disk and the line

        from_star: bool, if False, the line will be perpendicular to the midplane and cross it at 'equatorial_r' [au]. If True, the line of sight will go from the star through the midplane of the disk and 'equatorial_r' is ignored

        equatorial_r [au]: float, r coordinate of the point of intersection of the disk and the line

        wav [μm]: float, wavelength of the light the optical depth of which is to be calculated

        thickness: function, by which the volumetric density around the disk is determined. It is expected to take parameters of the point above which the density is needed:
            h [m]: float, height of the point of interest above the disk
            equatorial_r [m]: float, distance from the star to the point in equatorial plane under the point of interest
            T [K]: float, temperature of the dust at the point in equatorial plane under the point of interest
            sigma [kg/m²]: float, surface density of the dust at the point in equatorial plane under the point of interest
            M_star [kg]: float, mass of the star at the center

        filename: string, name of dustkappa* file to be loaded to get the opacity of the disk's dust

        message: bool, if True, will also write the value in terminal
        """
        if filename is None:
            import glob
            files = glob.glob("dustkappa*")
            if len(files) == 0: raise IOError("Optical depth error: Found no candidate for dustkappa file in current directory!")
            elif len(files) > 1: raise IOError("Optical depth error: Found multiple candidates for dustkappa file in current directory, please specify which file you'd like to use!")
            else: filename = files[0]
        wavs_kappas = np.loadtxt(filename, skiprows = 2, usecols = (0,1,2)) #v druhém sloupci je kappa_abs. Měl bych použít kappa scat? Idk idk
        kappa = np.interp(wav, (wavs_kappas[0]+wavs_kappas[1])*cm*cm/gram, wavs_kappas[2])
        depth = 0
        phi_index = index_before(array = self.phi_med[:], value = phi/180*np.pi)

        if from_star:
            from scipy.interpolate import UnivariateSpline
            spl = UnivariateSpline(self.r_med, thickness(h = 0, equatorial_r=self.r_med, T = self.T[:,phi_index], sigma = self.density[:,phi_index], M_star = M_sun))
            depth = kappa*spl.integral(self.Rmin, self.Rmax)
            if message: print(f"Optical depth along a line in the midplane at phi = {phi} deg for wavelength {wav} microns is {depth:.5e}")

        else:
            interpolated_T = np.interp(equatorial_r, self.r_med[:], self.T[:, phi_index])
            interpolated_sigma = np.interp(equatorial_r, self.r_med[:], self.density[:, phi_index])

            H = thickness(sigma = interpolated_sigma, T = interpolated_T, equatorial_r = equatorial_r*au, M_star = self.par["M_star"][0], getH=True)

            def integrand(z):
                return kappa*thickness(h = z, sigma = interpolated_sigma, T = interpolated_T, equatorial_r = equatorial_r*au, M_star = self.par["M_star"][0])

            depth, err = integrate.quad(integrand, -10*au, 10*au, points = [-10*H, 0, 10*H])
            if message: print(f"Optical depth for wavelength {wav} microns along a line intersecting midplane at point r = {equatorial_r:.2f} au, phi = {phi} deg is {depth:.5e}")
        return depth

    def integrate_mass(self):
        """
        Returns the mass of the flat disk in kg
        """
        areas = np.matmul(np.array([self.phi_sup-self.phi_inf]).T, [(self.r_sup*self.r_sup-self.r_inf*self.r_inf)*au*au])
        mass = np.sum(self.density*0.5*areas.T)
        return mass

    ##Input procedures

    def fargo_input(self, filename):
        """Read Fargo input file in.par"""
        with open(filename, 'r') as file:
            for line in file:
                if '#' not in line:
                    #'''
                    words = line.split()
                    name =''
                    value = ''
                    try:
                        name = words[0]
                    except Exception: pass
                    try:
                        value = words[1]
                    except Exception: pass
                    if is_number(value): self.par.update({f'{name}' : float(value)})
                    else: self.par.update({f'{name}' : value})

    def fargo_read_field(self, filename):
        """Read 1 Fargo binary file == field;
        makes the array 2D in order processed[r,φ]"""

        loaded = np.fromfile(filename, dtype=np.double)
        processed = np.reshape(loaded, (len(self.r_med), len(self.phi_med)))
        return(processed)

    def fargo_read_fields(self, no=0, Nrad = 1024, Nsec = 1536, Rmin = 2.8, Rmax = 14):
        """
        Reads Fargo file with "no" number, updates internal parameters acoordingly. This procedure expects polar coordinates in the read file with cells written from the inside out, ring after ring.
        --------------------
        Parameters:
        -------------------- 
        no: int, number in fargou output filename
        
        Nrad: int, number of cells in the radial direction in the fargo file
        
        Nsec: int, number of cells in the angular direction in the fargo file

        Rmin [au]: float, inner radius of the disk

        Rmax [au]: float, outer radius of the disk
        """

        self.par["Nrad"] = Nrad
        self.par["Nsec"] = Nsec
        self.Rmin = Rmin
        self.Rmax = Rmax

        self.r_inf = np.linspace(self.Rmin, self.Rmax-self.dr, self.par["Nrad"])
        self.r_med = np.linspace(self.Rmin+0.5*self.dr, self.Rmax-0.5*self.dr, self.par["Nrad"])
        self.r_sup = np.linspace(self.Rmin+self.dr, self.Rmax, self.par["Nrad"])

        self.phi_inf = np.linspace(self.Phimin, self.Phimax-self.dphi, self.par["Nsec"])
        self.phi_med = np.linspace(self.Phimin+0.5*self.dphi, self.Phimax-0.5*self.dphi, self.par["Nsec"])
        self.phi_sup = np.linspace(self.Phimin+self.dphi, self.Phimax, self.par["Nsec"])

        self.density = self.fargo_read_field("gasdens%d.dat" % (no))
        self.T = self.fargo_read_field("gastemper%d.dat" % (no))

        self.density = self.density/au/au*M_sun
        self.T = self.T*G*M_sun/au/R_gas*mu
        self.total_mass = self.integrate_mass()/M_sun

        print(f"Loaded files gasdens{no}.dat and gastemper{no}.dat. Total mass of the disk is {self.total_mass:.4e} M_sun")

    def flat_relax(self, angular = 4, radial = 100, log = True):
        r"""
        Reduces the amount of angular and radial sectors in the disk.
        --------------------
        Parameters:
        -------------------- 
        angular: int, desired number of angular sections in the desired disk. If 0, the angular dimension stays unchanged

        radial: int, desired number of radial sections in the desired disk. If 0, the radial dimension stays unchanged

        log: bool, if true, the new disk will have logarithmic grid in the radial direction
        
         ~ ~ ~ - ,                  ~ ~ ~ - ,       
        |     -    ' ,             |     -    ' ,   
        |    /     _. ,            |    /        ,  
        |   -   _.     ,     ->    |   - averaged , 
        |  / _-    _  — ,          |  /       _  — ,
        | -- _  —       ,          | -  _  —       ,
        |/—_____________,          |/—_____________,

        ~ ~ ~ - ,                  ~ ~ ~ - ,
        ______     ' ,             ______     ' ,     
              ——__     ,                 ——__     ,  
        ————__    \     ,    ->              \     , 
        ____   \    \    ,         ____averaged\    ,
        __  \   \    \   ,             \        \   , 
          |  |   |   |   ,              |       |   , 

        """
        print(f"Relaxing disk...")
        
        Nrad = self.par["Nrad"]
        Nsec = self.par["Nsec"]

        new_r_inf   = self.r_inf[:]
        new_r_med   = self.r_med[:]
        new_r_sup   = self.r_sup[:]
        new_phi_inf = self.phi_inf[:] 
        new_phi_med = self.phi_med[:] 
        new_phi_sup = self.phi_sup[:]
        
        newsigma=self.density[:]
        newT = self.T[:]

        if angular != 0:
            new_phis = np.linspace(self.Phimin, self.Phimax, num = 2*angular+1)
            new_phi_inf = new_phis[0:-1:2]
            new_phi_med = new_phis[1::2]
            new_phi_sup = new_phis[2::2]
            phi_pad = angular - Nsec%angular
            Nsec += phi_pad
            newsigma = np.pad(newsigma, ((0,0),(0,phi_pad)), 'edge') #'constant', constant_values = np.nan)
            newsigma = newsigma.reshape(Nrad, int(Nsec/angular), angular)
            newsigma = np.mean(newsigma, axis = 1)
            newsigma = newsigma.reshape(Nrad, angular)

            newT = np.pad(newT, ((0,0),(0,phi_pad)), 'edge') #'constant', constant_values = np.nan)
            newT = newT.reshape(Nrad, int(Nsec/angular), angular)
            newT = np.mean(newT, axis = 1)
            newT = newT.reshape(Nrad, angular)

            Nsec = angular
            self.phi_inf = new_phi_inf[:]
            self.phi_med = new_phi_med[:]
            self.phi_sup = new_phi_sup[:]
            self.par["Nsec"] = Nsec

        if log:
            new_rs = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), num = 2*Nrad+1)
            new_r_inf = new_rs[0:-1:2]
            new_r_med = new_rs[1::2]
            new_r_sup = new_rs[2::2]
            newsigma = np.array([np.interp(new_r_med, self.r_med, newsigma[:,i]) for i in range(len(self.phi_med))]).T
            newT = np.array([np.interp(new_r_med, self.r_med, newT[:,i]) for i in range(len(self.phi_med))]).T
            self.r_inf[:] = new_r_inf[:]
            self.r_med[:] = new_r_med[:]
            self.r_sup[:] = new_r_sup[:]

        if radial != 0:
            if log:
                new_rs = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), num = 2*radial+1)
                new_r_inf = new_rs[0:-1:2]
                new_r_med = new_rs[1::2]
                new_r_sup = new_rs[2::2]
            else:
                new_rs = np.linspace(self.Rmin, self.Rmax, num = 2*radial+1)
                new_r_inf = new_rs[0:-1:2]
                new_r_med = new_rs[1::2]
                new_r_sup = new_rs[2::2]
            self.r_inf = new_r_inf[:]
            self.r_med = new_r_med[:]
            self.r_sup = new_r_sup[:]
            if radial != 0:
                r_pad = radial - Nrad%radial            
                Nrad += r_pad
                newsigma = np.pad(newsigma, ((0, r_pad),(0,0)), 'edge')#, constant_values = np.nan)
                newsigma = newsigma.T.reshape(radial, Nsec, int(Nrad/radial))
                newsigma = np.mean(newsigma, axis = 2)
                newsigma = newsigma.reshape(Nsec, radial).T

                newT = np.pad(newT, ((0, r_pad),(0,0)), 'edge')#, constant_values = np.nan)
                newT = newT.T.reshape(radial, Nsec, int(Nrad/radial))
                newT = np.mean(newT, axis = 2)
                newT = newT.reshape(Nsec, radial).T
                Nrad = radial
                self.par["Nrad"] = Nrad

        self.density = newsigma
        self.T = newT

        print(f"Flat disk relaxed to "+ f"a log grid "*log + f"with a total of {Nrad} radial and {Nsec} angular sectors" )

    def inner_outer_finish(self, outer_steps = 100, inner_steps = 100, r_newmax = 80, r_newmin = 1.001*1.392000000e+11*cm/au, smoothing = 5, r1 = 0.1, r2 = 15, r3 = 0.1):
        """
        Finishes inner and outer parts of the flat disk using funcions
        sigma(r) = g(r) = exp(-(r1/r)**2.0) and
        T(r) = u(r) = 500
        for the inner part,
        sigma(r) = h(r) = exp(-((r-self.Rmax)/r2)**1.0)
        T(r) = v(r) = exp(-((r-self.Rmax)/r2)**1.0)
        for the outer part. Blends original disk with these functions using the power_smooth_min function.
        --------------------
        Parameters:
        --------------------
            outer_steps: int, number of new cells outside of the outer rim of the disk. If set to 0 no new cells will be added
            
            inner_steps: int, number of new cells inside of the inner rim of the disk. If set to 0 no new cells will be added

            r_newmax: float [au], new outer radius of the disk
            
            r_newmin: float [au], new inner radius of the disk. Must not be smaller than the radius of the star. By default set to be the radius of the Sun. 
            
            smoothing: float, parameter for power_smooth_min function. Determines how quickly the disk transitions to the finishing functions mentioned above 

            r1, r2, r3: float, parameters for finishing functions, see above
        """

        def f(r):   #funciton mimicing the disk's sigma
            return r**(-0.5)
        def g(r):   #function taking care of sigma's fading as it gets to the center
            return np.exp(-(r1/r)**2.0)
        def h(r):   #function taking care of sigma's fading as it gets towards the outer rim
            return np.exp(-((r-self.Rmax)/r2)**1.0)

        def t(r):   #function mimicing the disk's T
            return r**(-0.5)
        def u(r):   #function taking care of temperature's profile as it gets to the center
            return 0*r+500
        def v(r):   #function taking care of temperature's fading as it gets towards the outer rim
            return np.exp(-((r-self.Rmax)/r2)**1.0)


        #Here starts the finishing of the inner part of the disk      
        if inner_steps != 0:
            dr = (self.Rmin-r_newmin)/inner_steps
            self.r_inf = np.insert(self.r_inf, 0, np.linspace(r_newmin,self.Rmin+dr, inner_steps+1)[:-1] )
            r_med_new =  np.linspace(r_newmin+dr/2, self.Rmin-dr/2, inner_steps)
            self.r_med = np.insert(self.r_med, 0, r_med_new)
            self.r_sup = np.insert(self.r_sup, 0, np.linspace(r_newmin+dr,self.Rmin, inner_steps+1)[1:])
            self.par["Nrad"] += outer_steps

            inner_Csigma = self.density[0,:]/f(self.Rmin)
            inner_CT =     self.T[0,:]/t(self.Rmin)

            inner_sigma_ring = np.outer(f(r_med_new), np.transpose(inner_Csigma))
            inner_T_ring =     np.outer(t(r_med_new), np.transpose(inner_CT)    )

            inner_sigma_fading = np.outer(g(r_med_new), 1.3*np.mean(np.transpose(inner_Csigma)))
            inner_T_fading =     np.outer(u(r_med_new), np.ones((1,self.par["Nsec"]))    )

            new_inner_sigma = smooth_min(inner_sigma_ring, inner_sigma_fading, smoothing = smoothing)
            new_inner_T =     smooth_min(inner_T_fading, inner_T_ring,         smoothing = smoothing)
            
            self.density = np.insert(self.density, [0], new_inner_sigma, axis = 0)
            self.T =     np.insert(self.T, [0], new_inner_T, axis = 0)

            self.Rmin = r_newmin

        #Here starts the finishing of the outer part of the disk
        if outer_steps != 0:
            dr = (r_newmax-self.Rmax)/outer_steps
            r_med_new = np.linspace(self.Rmax+dr/2, r_newmax-dr/2, outer_steps)
            self.r_inf = np.append(self.r_inf, np.linspace(self.Rmax, r_newmax+dr, outer_steps+1)[1:] )
            self.r_med = np.append(self.r_med, r_med_new)
            self.r_sup = np.append(self.r_sup, np.linspace(self.Rmax+dr, r_newmax, outer_steps+1)[1:] )
            self.par["Nrad"] += inner_steps

            outer_Csigma = self.density[-1,:]/f(self.Rmax)
            outer_CT =     self.T[-1,:]/t(self.Rmax)

            outer_sigma_ring = np.outer(f(r_med_new), np.transpose(outer_Csigma))
            outer_T_ring =     np.outer(t(r_med_new), np.transpose(outer_CT)    )

            outer_sigma_fading = np.outer(h(r_med_new), np.mean(np.transpose(outer_Csigma)) )
            outer_T_fading =     np.outer(v(r_med_new), np.mean(outer_CT)*np.ones((1,self.par["Nsec"])))

            new_outer_sigma = smooth_min(outer_sigma_ring, outer_sigma_fading, smoothing = smoothing)
            new_outer_T     = smooth_min(outer_T_ring,     outer_T_fading,     smoothing = smoothing)

            self.density = np.append(self.density, new_outer_sigma, axis = 0)
            self.T =     np.append(self.T,     new_outer_T,     axis = 0)

            self.Rmax = r_newmax
        print(f"The disk was finished with {inner_steps} new inner and {outer_steps} new outer radial sectors")

    def gas_to_dust(self):
        """
        Modifies the disk's density according to its temperature to better reflect dust density with its typical evaporations and stuffs
        Brinstiel et. al. 2012
        """
        simple_modifier = .01
        self.density*= simple_modifier

    ##radmc3D procedures
    ##ALL RADMC3D FILES NEED TO BE IN CGS, conversion happens at the end of each procedure tho

    def radmc_write_dust_density(self, binary = True, thickness = f_z, nthet = 100, buffer_size = 4096, note = "", theta_min = np.pi*0.25):
        """
        Write dust_density.(b)inp file for radmc3D by thickening flat disk. The procedure will print progress percentage every 5 minutes starting after the first minute.
        --------------------
        Parameters:
        --------------------
        binary: bool, if set True, binary file with .binp file extension. Otherwise creates a text file with .inp file extension

        thickness: function, by which the volumetric density around the disk is determined. It is expected to take parameters of the point above which the density is needed:
            h [m]: float, height of the point of interest above the disk
            equatorial_r [m]: float, distance from the star to the point in equatorial plane under the point of interest
            T [K]: float, temperature of the dust at the point in equatorial plane under the point of interest
            sigma [kg/m²]: float, surface density of the dust at the point in equatorial plane under the point of interest
            M_star [kg]: float, mass of the star at the center
        
        nthet: int, number of cells in theta direction to be made above the disk. The same amount is generated beneath the disk as well, mirrored by radmc itself

        buffer_size: int, max number of floats to be stored in RAM at any given point

        note: string, added as a second line of fig title

        theta_min: float, lower theta bound. Useful if you don't need a bunch of cells in the „polar“ directions
        """

        print(f"Density notice: writing a dust_density."+binary*f"b"+f"inp for radmc3d, a total of {len(self.r_med)}×{nthet}×{len(self.phi_med)} = {np.size(self.density)*nthet} cells...")
        if len(self.density) == 0:
            raise ValueError("Write dust error: list of densities is empty!")
        start_time = time.time()
        if self.density.ndim == 3:
            print("Disk density is 3D; writing it down...")
            nrspecs = 1  # number of dust species
            nrcells = np.size(self.density)
            
            if binary:
                outfile = open("dust_density.binp"+note,"wb")
                np.array([1,8,nrcells,nrspecs]).tofile(outfile)
                (self.density*cm*cm*cm/gram).tofile(outfile)
            else:
                outfile = open("dust_density.inp"+note,"w")
                np.array([1,8,nrcells,nrspecs]).tofile(outfile)
                np.savetxt(outfile, self.density*cm*cm*cm/gram, delimiter = "\n")
            outfile.close()
            return 0
        nrspecs = 1  # number of dust species
        nrcells = np.size(self.density)*nthet

        #Update to theta boundaries is necessary
        self.par["Nthet"] = nthet
        self.par["Thetamin"] = theta_min
        self.par["Thetamax"] = np.pi/2
        thetas = np.power(np.linspace(self.par["Thetamin"], self.par["Thetamax"]**3, num = nthet+1),1/3)

        self.theta_inf = thetas[:-1]#np.linspace(self.par["Thetamin"], self.par["Thetamax"]-dtheta, self.par["Nthet"])
        self.theta_sup = thetas[1:]#np.linspace(self.par["Thetamin"]+dtheta, self.par["Thetamax"], self.par["Nthet"])
        self.theta_med = 0.5*(self.theta_inf+self.theta_sup)#np.linspace(self.par["Thetamin"]+0.5*dtheta, self.par["Thetamax"]-0.5*dtheta, self.par["Nthet"])

        total_mass = 0

        buffer_size = int(buffer_size/40)    #there are total of 40 arrays of size "buffer_size" used at once later, so the number has to be adjusted to correctly represent the amount of mermory the arrays take up
        if binary:
            outfile = open("dust_density.binp"+note,"wb")
            np.array([1,8,nrcells,nrspecs]).tofile(outfile) #Necessary parameters for radmc3d: 1 (always present), 8(# of bytes),…
            timer = time.time()+60
            for phi_index in range(len(self.phi_med)):
                for theta_index, theta in enumerate(self.theta_med):
                    cont = False
                    first_index = 0
                    last_index = buffer_size
                    while not cont:
                        if last_index > len(self.r_med)-1:                                                  #              .·z3
                            last_index = len(self.r_med)                                                    #          .·      ·
                            cont = True                                                                     #      .·           .
                        rs = self.r_med[first_index:last_index]                                             #   .·               .
                        equatorial_rs = np.sin(theta)*rs                                                    #z4·       o<—z       . <— elementary volume from the side
                        equatorial_r1s = np.sin(self.theta_sup[theta_index])*rs                             #  ·                  ·
                        equatorial_r2s = np.sin(self.theta_sup[theta_index])*rs                             #    ·                . 
                        equatorial_r3s = np.sin(self.theta_inf[theta_index])*rs                             #     .          ____.z2
                        equatorial_r4s = np.sin(self.theta_inf[theta_index])*rs                             #       z1..———̅ 
                        zs = np.cos(theta)*rs                                                               #Average density from points z through z4 is saved       
                        z1s = self.r_inf[first_index:last_index]*np.cos(self.theta_sup[theta_index])        
                        z2s = self.r_sup[first_index:last_index]*np.cos(self.theta_sup[theta_index])        
                        z3s = self.r_sup[first_index:last_index]*np.cos(self.theta_inf[theta_index])        
                        z4s = self.r_inf[first_index:last_index]*np.cos(self.theta_inf[theta_index])
                        interpolated_Ts =  np.interp(equatorial_rs,  self.r_med, self.T[:,phi_index])   
                        interpolated_T1s = np.interp(equatorial_r1s, self.r_med, self.T[:,phi_index])   
                        interpolated_T2s = np.interp(equatorial_r2s, self.r_med, self.T[:,phi_index])        
                        interpolated_T3s = np.interp(equatorial_r3s, self.r_med, self.T[:,phi_index])   
                        interpolated_T4s = np.interp(equatorial_r4s, self.r_med, self.T[:,phi_index])   
                        interpolated_sigmas =  np.interp(equatorial_rs,  self.r_med, self.density[:,phi_index])
                        interpolated_sigma1s = np.interp(equatorial_r1s, self.r_med, self.density[:,phi_index])
                        interpolated_sigma2s = np.interp(equatorial_r2s, self.r_med, self.density[:,phi_index])
                        interpolated_sigma3s = np.interp(equatorial_r3s, self.r_med, self.density[:,phi_index])
                        interpolated_sigma4s = np.interp(equatorial_r4s, self.r_med, self.density[:,phi_index])
                        rhos  = thickness(h =  zs*au, sigma = interpolated_sigmas,  T = interpolated_Ts,  equatorial_r=equatorial_rs*au,  M_star= self.par["M_star"][0])
                        rho1s = thickness(h = z1s*au, sigma = interpolated_sigma1s, T = interpolated_T1s, equatorial_r=equatorial_r1s*au, M_star= self.par["M_star"][0])
                        rho2s = thickness(h = z2s*au, sigma = interpolated_sigma2s, T = interpolated_T2s, equatorial_r=equatorial_r2s*au, M_star= self.par["M_star"][0])
                        if theta_index != 0:
                            rho3s = thickness(h = z3s*au, sigma = interpolated_sigma3s, T = interpolated_T3s, equatorial_r=equatorial_r3s*au, M_star= self.par["M_star"][0])
                            rho4s = thickness(h = z4s*au, sigma = interpolated_sigma4s, T = interpolated_T4s, equatorial_r=equatorial_r4s*au, M_star= self.par["M_star"][0])
                            average_rhos = (rhos+rho1s+rho2s+rho3s+rho4s)/5
                        else: average_rhos = (rhos+rho1s+rho2s)/3

                        elementary_volumes = (self.phi_sup[phi_index]-self.phi_inf[phi_index])*(self.r_sup[first_index:last_index]**3-self.r_inf[first_index:last_index]**3)/3*(np.cos(self.theta_inf[theta_index])-np.cos(self.theta_sup[theta_index]))*au*au*au
                        total_mass += np.sum(elementary_volumes*average_rhos)

                        (average_rhos*cm*cm*cm/gram).tofile(outfile)
                        if cont: break
                        first_index, last_index = last_index, last_index + buffer_size

                if time.time()>timer:
                    print(f"{phi_index/(len(self.phi_med)-1)*100:.2f} % done")
                    timer += 300

        else:
            outfile = open("dust_density.inp" + note,"w")
            np.array([1,nrcells,nrspecs]).tofile(outfile, sep = "\n") #Necessary parameters for radmc3d: 1 (always present)…
            outfile.write("\n")
            timer = time.time()+60
            for phi_index in range(len(self.phi_med)):
                for theta_index, theta in enumerate(self.theta_med):
                    cont = False
                    first_index = 0
                    last_index = buffer_size
                    while not cont:
                        if last_index > len(self.r_med)-1:                                                  #              .·z3
                            last_index = len(self.r_med)                                                    #          .·      ·
                            cont = True                                                                     #      .·           .
                        rs = self.r_med[first_index:last_index]                                             #   .·               .
                        equatorial_rs = np.sin(theta)*rs                                                    #z4·       o<—z       . <— elementary volume from the side
                        equatorial_r1s = np.sin(self.theta_sup[theta_index])*rs                             #  ·                  ·
                        equatorial_r2s = np.sin(self.theta_sup[theta_index])*rs                             #    ·                . 
                        equatorial_r3s = np.sin(self.theta_inf[theta_index])*rs                             #     .          ____.z2
                        equatorial_r4s = np.sin(self.theta_inf[theta_index])*rs                             #       z1..———̅ 
                        zs = np.cos(theta)*rs                                                               #Average density from points z through z4 is used       
                        z1s = self.r_inf[first_index:last_index]*np.cos(self.theta_sup[theta_index])        
                        z2s = self.r_sup[first_index:last_index]*np.cos(self.theta_sup[theta_index])        
                        z3s = self.r_sup[first_index:last_index]*np.cos(self.theta_inf[theta_index])        
                        z4s = self.r_inf[first_index:last_index]*np.cos(self.theta_inf[theta_index])
                        interpolated_Ts =  np.interp(equatorial_rs,  self.r_med, self.T[:,phi_index])   
                        interpolated_T1s = np.interp(equatorial_r1s, self.r_med, self.T[:,phi_index])   
                        interpolated_T2s = np.interp(equatorial_r2s, self.r_med, self.T[:,phi_index])        
                        interpolated_T3s = np.interp(equatorial_r3s, self.r_med, self.T[:,phi_index])   
                        interpolated_T4s = np.interp(equatorial_r4s, self.r_med, self.T[:,phi_index])   
                        interpolated_sigmas =  np.interp(equatorial_rs,  self.r_med, self.density[:,phi_index])
                        interpolated_sigma1s = np.interp(equatorial_r1s, self.r_med, self.density[:,phi_index])
                        interpolated_sigma2s = np.interp(equatorial_r2s, self.r_med, self.density[:,phi_index])
                        interpolated_sigma3s = np.interp(equatorial_r3s, self.r_med, self.density[:,phi_index])
                        interpolated_sigma4s = np.interp(equatorial_r4s, self.r_med, self.density[:,phi_index])
                        rhos  = thickness(h =  zs*au, sigma = interpolated_sigmas,  T = interpolated_Ts,  equatorial_r=equatorial_rs*au,  M_star= self.par["M_star"][0])
                        rho1s = thickness(h = z1s*au, sigma = interpolated_sigma1s, T = interpolated_T1s, equatorial_r=equatorial_r1s*au, M_star= self.par["M_star"][0])
                        rho2s = thickness(h = z2s*au, sigma = interpolated_sigma2s, T = interpolated_T2s, equatorial_r=equatorial_r2s*au, M_star= self.par["M_star"][0])
                        if theta_index != 0:
                            rho3s = thickness(h = z3s*au, sigma = interpolated_sigma3s, T = interpolated_T3s, equatorial_r=equatorial_r3s*au, M_star= self.par["M_star"][0])
                            rho4s = thickness(h = z4s*au, sigma = interpolated_sigma4s, T = interpolated_T4s, equatorial_r=equatorial_r4s*au, M_star= self.par["M_star"][0])
                            average_rhos = (rhos+rho1s+rho2s+rho3s+rho4s)/5
                        else: average_rhos = (rhos+rho1s+rho2s)/3

                        elementary_volumes = (self.phi_sup[phi_index]-self.phi_inf[phi_index])*(self.r_sup[first_index:last_index]**3-self.r_inf[first_index:last_index]**3)/3*(np.cos(self.theta_inf[theta_index])-np.cos(self.theta_sup[theta_index]))*au*au*au
                        total_mass += np.sum(elementary_volumes*average_rhos)

                        (average_rhos*cm*cm*cm/gram).tofile(outfile, sep = "\n", format="%9e")
                        outfile.write("\n")

                        first_index, last_index = last_index, last_index + buffer_size

                if time.time()>timer:
                    print(f"{phi_index/(len(self.phi_med)-1)*100:.2f} % done")
                    timer += 300

        outfile.close()
        flat_mass = self.integrate_mass()
        print(f"Density notice: dust_density."+binary*f"b"+f"inp file was created using '{thickness.__name__}' function to make the disk thick. The total mass of the disk is now {total_mass/M_sun:.4e} M_sun and it took {(time.time()-start_time)/60:.2f} min")
        if total_mass/flat_mass > 4: print(f"Your new disk is {total_mass/flat_mass:.1f} more massive than its flat version. This could be caused by too rough of a discretisation in the theta direction. Consider using a higher 'steps' parameter when calling radmc_write_dust_density.")

    def radmc_grid(self, style = 0): #, grid_type = 100, style = 0, x1min = 2.8, x1max=14, x2min = 0, x2max = np.pi, x3min = 0, x3max = 2*np.pi):
        """
        Creates amr_grid.inp file for the radmc3d program. 
        --------------------
        Parameters:
        --------------------
        style: int, 0, 1 or 2, determining the style of the grid: 0 for regular, 1 for layered, 2 for oct-tree. SO FAR ONLY REGULAR GRID SUPPORTED
        """ 
        grid_type=100                   #makes the grid spherical as opposed to cartesian (grid type 0)
        if style != 0: raise ValueError("Grid error: Non-uniform grid style not yet supported!")

        nx = [len(self.r_med),len(self.theta_med), len(self.phi_med)] #number of cells in each direction (x, y, z, or r, θ, φ, or r, φ, z) gets imported
        if 0 in nx:
            raise ValueError("Grid error: wrong dimension of density array. Perhaps it wasn't thickened yet?")

        with open("amr_grid.inp", "w") as outfile:
            outfile.write(f"1\n{style}\n{grid_type}\n0\n1 1 1\n{nx[0]} {nx[1]} {nx[2]}\n")

            outfile.write(f"{self.r_inf[0]*au/cm:.9e}\n")
            for rboundary in self.r_sup:
                outfile.write(f"{rboundary*au/cm:.9e}\n")

            outfile.write(f"{self.theta_inf[0]:.9e}\n")
            for thetaboundary in self.theta_sup:
                outfile.write(f"{thetaboundary:.9e}\n")

            outfile.write(f"{self.phi_inf[0]:.9e}\n")
            for phiboundary in self.phi_sup:
                outfile.write(f"{phiboundary:.9e}\n")

        print("Grid notice: amr_grid.inp file was created.")

    def radmc_stars(self, coordinates=[0,0,0], radii=[1.392000000e+11], fluxes=None):
        """Creates stars.inp file for radmc3d program
        --------------------
        Parameters:
        --------------------
        coordinates [cm]: array-like, positions of stars, expected in [x₁,y₁,z₁,x₂,y₂,z₂,…] format, that is cartesian coordinates

        masses [g]: array-like, masses of stars, expected in [m₁, m₂, …] format
        
        radii [cm]: array-like, masses of stars, expected in [r₁, r₂, …] format

        fluxes [erg·cm¯²·s⁻¹·Hz¯¹] OR [-K]: array-like, fluxes of stars (as seen from 1 parasec = 3.08572×10¹⁸ cm) in wavelengths λ, expected in [F₁(λ₁), F₁(λ₂), …, F₁(λₙ), F₂(λ₁), …] format. If a blackbody radiation is to be assumed, then a negative value of star's surface temperature is to be entered. For instance if star 2 has surface temperature of 5000 K, then fluxes will look like [F₁(λ₁), F₁(λ₂), … ,F₁(λₙ), -5000, F₃(λ₁), …]
        """
        masses = self.par["M_star"]/gram

        if fluxes is None: fluxes = [-self.par["Star_temp"]]

        if len(coordinates)%3 != 0: raise ValueError("Stars error: Number of coordinates isn't a multiple of 3!")
        if len(coordinates) != 3*len(masses): raise ValueError("Stars error: List of star masses isn't the same length as that of their coordinates!")
        if len(coordinates) != 3*len(radii): raise ValueError("Stars error: List of star radii isn't the same length as that of their coordinates!")
        if len(masses) != len(radii): raise ValueError("Stars error: List of star radii isn't the same length as that of their masses!")

        nlambda = 0
        try:
            with open("wavelength_micron.inp","r") as wavelength:
                nlambda = wavelength.readline()
        except FileNotFoundError:
            raise Exception("Stars error: wavelength_micron.inp file not found. It is essential for generating stars.inp!")

        with open("stars.inp", "w") as outfile:
            outfile.write(f"2\n{len(coordinates)//3} {nlambda}")
            for i in range(len(masses)):
                outfile.write(f"{radii[i]:.9e} {masses[i]:.9e} {coordinates[i]:.9e} {coordinates[i+1]:.9e} {coordinates[i+2]:.9e}\n")
            outfile.write("\n")
            done_with_lambda = False
            with open("wavelength_micron.inp","r") as wavelength:
                for line in wavelength:
                    if done_with_lambda:
                        if is_number(line): outfile.write(f"{float(line):.9e}\n")
                    else: done_with_lambda = True
            outfile.write("\n")
            for value in fluxes:
                outfile.write(f"{value:.9e}\n")

        print("Stars notice: stars.inp file was created.")

    def radmc_wavelength(self,array=None, smallest = None, largest = None, step = None):
        """Creates wavelength_micron.inp file for radmc3d program"""
        #Supports input array of wavelengths, uniform distribution or, if no arguments are put in, will use predetermined wavelengths
        #Wavelength is in microns 

        if ((smallest is None) ^ (step is None)) or ((step is None) ^ (largest is None)):
            raise ValueError("Wavelength error: Smallest, largest AND step wavelength have to be passed!")
        if (array != None and smallest != None):
            raise ValueError("Wavelength error: Choose either the array method or uniform method, not both!")

        with open("wavelength_micron.inp","w") as outfile:
            if (array != None):
                outfile.write(f"{len(array)}")
                for i in range (0, len(array)):
                    outfile.write(f"\n{array[i]:.9e}")
            if (step != None):
                outfile.write(np.floor((largest-smallest)/step))
                lambd = smallest
                while lambd < largest:
                    outfile.write(f"\n{lambd:.9e}")
                    lambd+=step
            else: 
                predetirmed = """                    99
                    1.000000000e-01
                    1.250576960e-01
                    1.563942733e-01
                    1.955830748e-01
                    2.445916871e-01
                    3.058807285e-01
                    3.825273915e-01
                    4.783799423e-01
                    5.982509339e-01
                    7.481588342e-01
                    9.356302004e-01
                    1.170077572e+00
                    1.463272052e+00
                    1.829934315e+00
                    2.288473692e+00
                    2.861912473e+00
                    3.579041800e+00
                    4.475867213e+00
                    5.597416412e+00
                    7.000000000e+00
                    7.180503189e+00
                    7.365660863e+00
                    7.555593045e+00
                    7.750422850e+00
                    7.950276569e+00
                    8.155283751e+00
                    8.365577282e+00
                    8.581293478e+00
                    8.802572169e+00
                    9.029556789e+00
                    9.262394474e+00
                    9.501236151e+00
                    9.746236639e+00
                    9.997554752e+00
                    1.025535340e+01
                    1.051979968e+01
                    1.079106502e+01
                    1.106932526e+01
                    1.135476076e+01
                    1.164755654e+01
                    1.194790242e+01
                    1.225599306e+01
                    1.257202817e+01
                    1.289621263e+01
                    1.322875656e+01
                    1.356987552e+01
                    1.391979063e+01
                    1.427872872e+01
                    1.464692244e+01
                    1.502461047e+01
                    1.541203763e+01
                    1.580945504e+01
                    1.621712034e+01
                    1.663529775e+01
                    1.706425837e+01
                    1.750428023e+01
                    1.795564857e+01
                    1.841865598e+01
                    1.889360257e+01
                    1.938079621e+01
                    1.988055271e+01
                    2.039319602e+01
                    2.091905843e+01
                    2.145848083e+01
                    2.201181286e+01
                    2.257941320e+01
                    2.316164978e+01
                    2.375890002e+01
                    2.437155105e+01
                    2.500000000e+01
                    3.073733534e+01
                    3.779135136e+01
                    4.646421759e+01
                    5.712744950e+01
                    7.023782290e+01
                    8.635694065e+01
                    1.061752898e+02
                    1.305418194e+02
                    1.605003072e+02
                    1.973340706e+02
                    2.426209401e+02
                    2.983008479e+02
                    3.667589278e+02
                    4.509276862e+02
                    5.544126202e+02
                    6.816466650e+02
                    8.380800851e+02
                    1.030413945e+03
                    1.266887158e+03
                    1.557629417e+03
                    1.915095109e+03
                    2.354596824e+03
                    2.894961287e+03
                    3.559335835e+03
                    4.376179966e+03
                    5.380484445e+03
                    6.615270188e+03
                    8.133431126e+03
                    1.000000000e+04
                    """
                print(predetirmed.replace("                    ", ""), file = outfile)

    def radmc_write_opacity(self):
        """
        Rn just writes dustopac.inp and dustkappa_silicate.inp file. Kappas are in cm^2/g and lambdas in microns as per radmc's unit standard
        """

        silicate_ncols = "2"   #zeroth col is lambdas, always present. Then there are 2 cols: kappa_abs, kappa_scat (optional). Third col would be anisotropic g factor
        silicate_nlam = "400"  #number of wavelengths in this file. Doesn't have to be the same as in wavelength file
        silicate_kappas_header=[silicate_ncols, silicate_nlam]
        silicate_kappas = """    0.999992E-01  0.103173E+05  0.268585E+05
            0.102927E+00  0.108810E+05  0.277522E+05
            0.105940E+00  0.114659E+05  0.286342E+05
            0.109041E+00  0.120737E+05  0.295038E+05
            0.112233E+00  0.127084E+05  0.303582E+05
            0.115519E+00  0.133746E+05  0.311906E+05
            0.118901E+00  0.140758E+05  0.319903E+05
            0.122381E+00  0.148124E+05  0.327439E+05
            0.125964E+00  0.155804E+05  0.334388E+05
            0.129652E+00  0.163725E+05  0.340652E+05
            0.133447E+00  0.171797E+05  0.346178E+05
            0.137354E+00  0.179938E+05  0.350955E+05
            0.141375E+00  0.188093E+05  0.354996E+05
            0.145514E+00  0.196246E+05  0.358319E+05
            0.149774E+00  0.204421E+05  0.360928E+05
            0.154158E+00  0.212660E+05  0.362802E+05
            0.158671E+00  0.221011E+05  0.363901E+05
            0.163316E+00  0.229507E+05  0.364170E+05
            0.168097E+00  0.238149E+05  0.363553E+05
            0.173018E+00  0.246899E+05  0.362002E+05
            0.178083E+00  0.255677E+05  0.359493E+05
            0.183297E+00  0.264365E+05  0.356031E+05
            0.188663E+00  0.272820E+05  0.351658E+05
            0.194186E+00  0.280883E+05  0.346453E+05
            0.199870E+00  0.288394E+05  0.340535E+05
            0.205721E+00  0.289959E+05  0.333749E+05
            0.211744E+00  0.290515E+05  0.326783E+05
            0.217943E+00  0.290627E+05  0.320037E+05
            0.224323E+00  0.289972E+05  0.318489E+05
            0.230890E+00  0.289190E+05  0.319579E+05
            0.237649E+00  0.288975E+05  0.320907E+05
            0.244606E+00  0.288703E+05  0.324325E+05
            0.251767E+00  0.288675E+05  0.328730E+05
            0.259138E+00  0.289013E+05  0.332773E+05
            0.266724E+00  0.288705E+05  0.336559E+05
            0.274532E+00  0.287498E+05  0.339460E+05
            0.282569E+00  0.284789E+05  0.341415E+05
            0.290841E+00  0.280067E+05  0.342608E+05
            0.299355E+00  0.273569E+05  0.343273E+05
            0.308119E+00  0.265089E+05  0.344009E+05
            0.317139E+00  0.255776E+05  0.345087E+05
            0.326423E+00  0.245892E+05  0.346862E+05
            0.335979E+00  0.236394E+05  0.349484E+05
            0.345815E+00  0.227742E+05  0.352181E+05
            0.355938E+00  0.220134E+05  0.354719E+05
            0.366358E+00  0.213544E+05  0.355896E+05
            0.377084E+00  0.207228E+05  0.355084E+05
            0.388123E+00  0.200516E+05  0.350181E+05
            0.399485E+00  0.192083E+05  0.341451E+05
            0.411180E+00  0.181899E+05  0.327347E+05
            0.423217E+00  0.169268E+05  0.309794E+05
            0.435606E+00  0.155200E+05  0.289183E+05
            0.448359E+00  0.139730E+05  0.268183E+05
            0.461484E+00  0.125526E+05  0.246917E+05
            0.474994E+00  0.111900E+05  0.226870E+05
            0.488899E+00  0.990212E+04  0.208287E+05
            0.503212E+00  0.874924E+04  0.190963E+05
            0.517943E+00  0.783345E+04  0.174533E+05
            0.533106E+00  0.700109E+04  0.159278E+05
            0.548712E+00  0.624243E+04  0.145053E+05
            0.564776E+00  0.565248E+04  0.131816E+05
            0.581310E+00  0.511966E+04  0.119471E+05
            0.598327E+00  0.462765E+04  0.107979E+05
            0.615843E+00  0.423828E+04  0.973738E+04
            0.633872E+00  0.388285E+04  0.875851E+04
            0.652428E+00  0.355825E+04  0.785816E+04
            0.671528E+00  0.330220E+04  0.703276E+04
            0.691187E+00  0.306218E+04  0.628116E+04
            0.711421E+00  0.286012E+04  0.560478E+04
            0.732248E+00  0.268778E+04  0.499702E+04
            0.753684E+00  0.253103E+04  0.444944E+04
            0.775748E+00  0.240949E+04  0.395919E+04
            0.798458E+00  0.229488E+04  0.351939E+04
            0.821833E+00  0.222101E+04  0.313268E+04
            0.845892E+00  0.215353E+04  0.278706E+04
            0.870655E+00  0.208970E+04  0.247818E+04
            0.896143E+00  0.202932E+04  0.220250E+04
            0.922378E+00  0.199025E+04  0.196258E+04
            0.949380E+00  0.195619E+04  0.174928E+04
            0.977173E+00  0.192381E+04  0.155889E+04
            0.100578E+01  0.189118E+04  0.139020E+04
            0.103522E+01  0.185282E+04  0.124387E+04
            0.106553E+01  0.181621E+04  0.111294E+04
            0.109672E+01  0.178124E+04  0.995808E+03
            0.112883E+01  0.172172E+04  0.892476E+03
            0.116187E+01  0.166203E+04  0.800066E+03
            0.119589E+01  0.160511E+04  0.717290E+03
            0.123090E+01  0.152547E+04  0.643286E+03
            0.126693E+01  0.144623E+04  0.577002E+03
            0.130402E+01  0.136865E+04  0.517504E+03
            0.134220E+01  0.127801E+04  0.463359E+03
            0.138149E+01  0.119158E+04  0.414942E+03
            0.142193E+01  0.110472E+04  0.371249E+03
            0.146356E+01  0.101812E+04  0.331907E+03
            0.150640E+01  0.935389E+03  0.296705E+03
            0.155050E+01  0.856069E+03  0.264860E+03
            0.159589E+01  0.780228E+03  0.236472E+03
            0.164261E+01  0.709960E+03  0.210921E+03
            0.169070E+01  0.642901E+03  0.188139E+03
            0.174020E+01  0.581644E+03  0.167737E+03
            0.179114E+01  0.523604E+03  0.149546E+03
            0.184357E+01  0.472837E+03  0.133210E+03
            0.189754E+01  0.425087E+03  0.118647E+03
            0.195309E+01  0.380454E+03  0.105659E+03
            0.201027E+01  0.338224E+03  0.940673E+02
            0.206912E+01  0.300392E+03  0.836141E+02
            0.212969E+01  0.263954E+03  0.743265E+02
            0.219204E+01  0.228838E+03  0.660738E+02
            0.225621E+01  0.201317E+03  0.586418E+02
            0.232226E+01  0.175641E+03  0.520324E+02
            0.239025E+01  0.150835E+03  0.461669E+02
            0.246022E+01  0.136808E+03  0.410137E+02
            0.253224E+01  0.124868E+03  0.364433E+02
            0.260637E+01  0.113631E+03  0.323798E+02
            0.268267E+01  0.105936E+03  0.287456E+02
            0.276121E+01  0.984960E+02  0.255183E+02
            0.284204E+01  0.924681E+02  0.226512E+02
            0.292524E+01  0.877138E+02  0.201043E+02
            0.301088E+01  0.831345E+02  0.178404E+02
            0.309902E+01  0.788365E+02  0.158165E+02
            0.318974E+01  0.746750E+02  0.140206E+02
            0.328312E+01  0.706440E+02  0.124272E+02
            0.337924E+01  0.667381E+02  0.110136E+02
            0.347816E+01  0.629517E+02  0.975953E+01
            0.357998E+01  0.596976E+02  0.865465E+01
            0.368479E+01  0.566571E+02  0.767589E+01
            0.379266E+01  0.537091E+02  0.680706E+01
            0.390369E+01  0.508499E+02  0.603587E+01
            0.401797E+01  0.481977E+02  0.535043E+01
            0.413559E+01  0.462758E+02  0.473749E+01
            0.425666E+01  0.444116E+02  0.419400E+01
            0.438127E+01  0.426029E+02  0.371214E+01
            0.450954E+01  0.409266E+02  0.328375E+01
            0.464155E+01  0.402866E+02  0.289009E+01
            0.477743E+01  0.396712E+02  0.254244E+01
            0.491729E+01  0.390799E+02  0.223552E+01
            0.506124E+01  0.387081E+02  0.196411E+01
            0.520941E+01  0.386217E+02  0.172404E+01
            0.536191E+01  0.385488E+02  0.151240E+01
            0.551888E+01  0.383861E+02  0.132506E+01
            0.568045E+01  0.374800E+02  0.115470E+01
            0.584674E+01  0.366019E+02  0.100517E+01
            0.601790E+01  0.357540E+02  0.873800E+00
            0.619407E+01  0.349600E+02  0.757081E+00
            0.637540E+01  0.341905E+02  0.654984E+00
            0.656204E+01  0.338508E+02  0.563301E+00
            0.675414E+01  0.343580E+02  0.478966E+00
            0.695187E+01  0.348835E+02  0.405952E+00
            0.715539E+01  0.362907E+02  0.339480E+00
            0.736486E+01  0.379995E+02  0.281487E+00
            0.758046E+01  0.424716E+02  0.227665E+00
            0.780238E+01  0.518081E+02  0.175917E+00
            0.803079E+01  0.660613E+02  0.131711E+00
            0.826589E+01  0.144086E+03  0.868412E-01
            0.850787E+01  0.356048E+03  0.535963E-01
            0.875694E+01  0.684046E+03  0.435544E-01
            0.901330E+01  0.119514E+04  0.704892E-01
            0.927716E+01  0.178213E+04  0.138667E+00
            0.954875E+01  0.222483E+04  0.222010E+00
            0.982828E+01  0.236892E+04  0.284773E+00
            0.101160E+02  0.220500E+04  0.303835E+00
            0.104121E+02  0.193925E+04  0.294890E+00
            0.107170E+02  0.161733E+04  0.270614E+00
            0.110307E+02  0.130645E+04  0.237101E+00
            0.113536E+02  0.103239E+04  0.204324E+00
            0.116860E+02  0.763043E+03  0.169980E+00
            0.120281E+02  0.538850E+03  0.135856E+00
            0.123802E+02  0.396974E+03  0.103297E+00
            0.127426E+02  0.321406E+03  0.770989E-01
            0.131157E+02  0.307409E+03  0.565465E-01
            0.134996E+02  0.356843E+03  0.424531E-01
            0.138948E+02  0.421833E+03  0.346364E-01
            0.143016E+02  0.481202E+03  0.283038E-01
            0.147203E+02  0.549349E+03  0.226834E-01
            0.151512E+02  0.670128E+03  0.190361E-01
            0.155948E+02  0.888753E+03  0.199354E-01
            0.160513E+02  0.115639E+04  0.257492E-01
            0.165212E+02  0.138192E+04  0.347566E-01
            0.170049E+02  0.147495E+04  0.413287E-01
            0.175027E+02  0.145481E+04  0.444572E-01
            0.180151E+02  0.135701E+04  0.438780E-01
            0.185424E+02  0.123020E+04  0.407796E-01
            0.190853E+02  0.111886E+04  0.371025E-01
            0.196440E+02  0.101801E+04  0.334490E-01
            0.202191E+02  0.934515E+03  0.299806E-01
            0.208110E+02  0.859964E+03  0.267162E-01
            0.214202E+02  0.802793E+03  0.239215E-01
            0.220473E+02  0.754298E+03  0.214790E-01
            0.226927E+02  0.710366E+03  0.194809E-01
            0.233570E+02  0.667348E+03  0.177034E-01
            0.240408E+02  0.624729E+03  0.161021E-01
            0.247446E+02  0.573651E+03  0.145342E-01
            0.254690E+02  0.526093E+03  0.130511E-01
            0.262146E+02  0.481732E+03  0.116824E-01
            0.269820E+02  0.440206E+03  0.104417E-01
            0.277719E+02  0.397929E+03  0.929141E-02
            0.285849E+02  0.360654E+03  0.827110E-02
            0.294217E+02  0.325896E+03  0.732809E-02
            0.302830E+02  0.294123E+03  0.645885E-02
            0.311696E+02  0.267217E+03  0.568407E-02
            0.320820E+02  0.241931E+03  0.501032E-02
            0.330212E+02  0.222737E+03  0.441654E-02
            0.339879E+02  0.204086E+03  0.389623E-02
            0.349829E+02  0.190166E+03  0.343530E-02
            0.360070E+02  0.176553E+03  0.302957E-02
            0.370611E+02  0.163189E+03  0.268650E-02
            0.381461E+02  0.150576E+03  0.238189E-02
            0.392628E+02  0.140209E+03  0.210320E-02
            0.404122E+02  0.131092E+03  0.186025E-02
            0.415953E+02  0.124097E+03  0.165017E-02
            0.428130E+02  0.117273E+03  0.146381E-02
            0.440663E+02  0.110617E+03  0.129849E-02
            0.453563E+02  0.104433E+03  0.115276E-02
            0.466841E+02  0.992243E+02  0.102554E-02
            0.480508E+02  0.941666E+02  0.912385E-03
            0.494575E+02  0.892554E+02  0.811748E-03
            0.509053E+02  0.837287E+02  0.722040E-03
            0.523956E+02  0.779156E+02  0.642226E-03
            0.539295E+02  0.722821E+02  0.571338E-03
            0.555082E+02  0.673061E+02  0.508316E-03
            0.571332E+02  0.634901E+02  0.452179E-03
            0.588058E+02  0.597815E+02  0.402256E-03
            0.605273E+02  0.561585E+02  0.357793E-03
            0.622992E+02  0.525916E+02  0.318117E-03
            0.641230E+02  0.491211E+02  0.282854E-03
            0.660002E+02  0.461499E+02  0.251580E-03
            0.679324E+02  0.436153E+02  0.223820E-03
            0.699211E+02  0.411498E+02  0.199124E-03
            0.719680E+02  0.389189E+02  0.177143E-03
            0.740748E+02  0.367548E+02  0.157587E-03
            0.762433E+02  0.347357E+02  0.140208E-03
            0.784754E+02  0.328361E+02  0.124756E-03
            0.807727E+02  0.310585E+02  0.111019E-03
            0.831373E+02  0.294688E+02  0.988134E-04
            0.855711E+02  0.279228E+02  0.879496E-04
            0.880762E+02  0.264192E+02  0.782798E-04
            0.906546E+02  0.249570E+02  0.696729E-04
            0.933085E+02  0.235347E+02  0.620124E-04
            0.960401E+02  0.221514E+02  0.551941E-04
            0.988516E+02  0.208059E+02  0.491256E-04
            0.101746E+03  0.197591E+02  0.437412E-04
            0.104724E+03  0.189139E+02  0.389566E-04
            0.107790E+03  0.180926E+02  0.346952E-04
            0.110945E+03  0.172943E+02  0.308998E-04
            0.114193E+03  0.165186E+02  0.275194E-04
            0.117536E+03  0.157646E+02  0.245087E-04
            0.120977E+03  0.150318E+02  0.218273E-04
            0.124519E+03  0.143197E+02  0.194391E-04
            0.128164E+03  0.136275E+02  0.173122E-04
            0.131916E+03  0.129547E+02  0.154179E-04
            0.135778E+03  0.123009E+02  0.137309E-04
            0.139753E+03  0.116653E+02  0.122284E-04
            0.143844E+03  0.110476E+02  0.108903E-04
            0.148055E+03  0.104472E+02  0.969859E-05
            0.152389E+03  0.986356E+01  0.863729E-05
            0.156850E+03  0.929628E+01  0.769214E-05
            0.161442E+03  0.874487E+01  0.685041E-05
            0.166168E+03  0.820887E+01  0.610081E-05
            0.171033E+03  0.768785E+01  0.543324E-05
            0.176040E+03  0.718139E+01  0.483874E-05
            0.181193E+03  0.668907E+01  0.430931E-05
            0.186498E+03  0.621050E+01  0.383784E-05
            0.191957E+03  0.574528E+01  0.341797E-05
            0.197577E+03  0.529305E+01  0.304407E-05
            0.203361E+03  0.498768E+01  0.271163E-05
            0.209314E+03  0.478773E+01  0.241583E-05
            0.215442E+03  0.459346E+01  0.215230E-05
            0.221749E+03  0.440470E+01  0.191751E-05
            0.228240E+03  0.422130E+01  0.170834E-05
            0.234922E+03  0.404311E+01  0.152198E-05
            0.241799E+03  0.386998E+01  0.135594E-05
            0.248878E+03  0.370176E+01  0.120802E-05
            0.256164E+03  0.353831E+01  0.107624E-05
            0.263663E+03  0.337951E+01  0.958827E-06
            0.271382E+03  0.322521E+01  0.854224E-06
            0.279326E+03  0.307528E+01  0.761033E-06
            0.287503E+03  0.292961E+01  0.678007E-06
            0.295920E+03  0.278807E+01  0.604039E-06
            0.304583E+03  0.265055E+01  0.538140E-06
            0.313500E+03  0.251692E+01  0.479430E-06
            0.322677E+03  0.238709E+01  0.427125E-06
            0.332124E+03  0.226093E+01  0.380526E-06
            0.341846E+03  0.213836E+01  0.339011E-06
            0.351854E+03  0.201925E+01  0.302025E-06
            0.362154E+03  0.190352E+01  0.269074E-06
            0.372756E+03  0.179108E+01  0.239718E-06
            0.383669E+03  0.168182E+01  0.213565E-06
            0.394900E+03  0.157565E+01  0.190266E-06
            0.406461E+03  0.147249E+01  0.169508E-06
            0.418360E+03  0.137226E+01  0.151016E-06
            0.430608E+03  0.127486E+01  0.134541E-06
            0.443214E+03  0.118022E+01  0.119863E-06
            0.456188E+03  0.108827E+01  0.106787E-06
            0.469543E+03  0.998916E+00  0.951377E-07
            0.483289E+03  0.912095E+00  0.847594E-07
            0.497437E+03  0.827734E+00  0.755135E-07
            0.512000E+03  0.781318E+00  0.672821E-07
            0.526988E+03  0.737506E+00  0.599479E-07
            0.542416E+03  0.696150E+00  0.534132E-07
            0.558295E+03  0.657113E+00  0.475908E-07
            0.574639E+03  0.620265E+00  0.424031E-07
            0.591461E+03  0.585484E+00  0.377809E-07
            0.608776E+03  0.552652E+00  0.336626E-07
            0.626598E+03  0.521662E+00  0.299931E-07
            0.644941E+03  0.492410E+00  0.267237E-07
            0.663822E+03  0.464798E+00  0.238107E-07
            0.683255E+03  0.438734E+00  0.212151E-07
            0.703257E+03  0.414132E+00  0.189026E-07
            0.723845E+03  0.390909E+00  0.168421E-07
            0.745035E+03  0.368989E+00  0.150062E-07
            0.766846E+03  0.348298E+00  0.133704E-07
            0.789295E+03  0.328767E+00  0.119130E-07
            0.812402E+03  0.310331E+00  0.106144E-07
            0.836185E+03  0.292929E+00  0.945734E-08
            0.860664E+03  0.276503E+00  0.842643E-08
            0.885860E+03  0.260998E+00  0.750790E-08
            0.911793E+03  0.246363E+00  0.668949E-08
            0.938485E+03  0.232548E+00  0.596029E-08
            0.965959E+03  0.219508E+00  0.531058E-08
            0.994238E+03  0.207199E+00  0.473170E-08
            0.102334E+04  0.195580E+00  0.421591E-08
            0.105330E+04  0.184613E+00  0.375635E-08
            0.108414E+04  0.174261E+00  0.334689E-08
            0.111587E+04  0.164489E+00  0.298206E-08
            0.114854E+04  0.155265E+00  0.265699E-08
            0.118217E+04  0.146558E+00  0.236736E-08
            0.121677E+04  0.138340E+00  0.210931E-08
            0.125239E+04  0.130583E+00  0.187938E-08
            0.128906E+04  0.123260E+00  0.167452E-08
            0.132679E+04  0.116348E+00  0.149198E-08
            0.136564E+04  0.109824E+00  0.132935E-08
            0.140561E+04  0.103666E+00  0.118444E-08
            0.144676E+04  0.978526E-01  0.105533E-08
            0.148912E+04  0.923655E-01  0.940292E-09
            0.153271E+04  0.871861E-01  0.837794E-09
            0.157758E+04  0.822971E-01  0.746469E-09
            0.162376E+04  0.776823E-01  0.665100E-09
            0.167130E+04  0.733262E-01  0.592600E-09
            0.172023E+04  0.692144E-01  0.528003E-09
            0.177058E+04  0.653332E-01  0.470447E-09
            0.182242E+04  0.616696E-01  0.419165E-09
            0.187577E+04  0.582115E-01  0.373474E-09
            0.193068E+04  0.549472E-01  0.332763E-09
            0.198720E+04  0.518661E-01  0.296490E-09
            0.204538E+04  0.489576E-01  0.264170E-09
            0.210526E+04  0.462123E-01  0.235374E-09
            0.216689E+04  0.436210E-01  0.209717E-09
            0.223032E+04  0.411749E-01  0.186857E-09
            0.229561E+04  0.388660E-01  0.166488E-09
            0.236282E+04  0.366866E-01  0.148340E-09
            0.243199E+04  0.346294E-01  0.132170E-09
            0.250318E+04  0.326875E-01  0.117762E-09
            0.257646E+04  0.308546E-01  0.104926E-09
            0.265189E+04  0.291244E-01  0.934881E-10
            0.272952E+04  0.274912E-01  0.832973E-10
            0.280943E+04  0.259496E-01  0.742174E-10
            0.289167E+04  0.244945E-01  0.661272E-10
            0.297633E+04  0.231210E-01  0.589190E-10
            0.306346E+04  0.218245E-01  0.524964E-10
            0.315314E+04  0.206006E-01  0.467740E-10
            0.324545E+04  0.194455E-01  0.416753E-10
            0.334046E+04  0.183550E-01  0.371325E-10
            0.343825E+04  0.173258E-01  0.330848E-10
            0.353890E+04  0.163542E-01  0.294784E-10
            0.364250E+04  0.154372E-01  0.262650E-10
            0.374914E+04  0.145715E-01  0.234020E-10
            0.385889E+04  0.137544E-01  0.208510E-10
            0.397186E+04  0.129831E-01  0.185781E-10
            0.408814E+04  0.122551E-01  0.165530E-10
            0.420781E+04  0.115679E-01  0.147486E-10
            0.433100E+04  0.109192E-01  0.131409E-10
            0.445779E+04  0.103069E-01  0.117085E-10
            0.458829E+04  0.972895E-02  0.104322E-10
            0.472261E+04  0.918340E-02  0.929502E-11
            0.486086E+04  0.866844E-02  0.828180E-11
            0.500316E+04  0.818235E-02  0.737903E-11
            0.514963E+04  0.772352E-02  0.657467E-11
            0.530038E+04  0.729043E-02  0.585799E-11
            0.545555E+04  0.688161E-02  0.521944E-11
            0.561526E+04  0.649572E-02  0.465048E-11
            0.577965E+04  0.613147E-02  0.414355E-11
            0.594884E+04  0.578765E-02  0.369188E-11
            0.612299E+04  0.546311E-02  0.328944E-11
            0.630224E+04  0.515676E-02  0.293087E-11
            0.648674E+04  0.486759E-02  0.261139E-11
            0.667664E+04  0.459464E-02  0.232673E-11
            0.687210E+04  0.433700E-02  0.207310E-11
            0.707327E+04  0.409380E-02  0.184712E-11
            0.728034E+04  0.386424E-02  0.164577E-11
            0.749347E+04  0.364755E-02  0.146638E-11
            0.771284E+04  0.344301E-02  0.130653E-11
            0.793863E+04  0.324994E-02  0.116411E-11
            0.817104E+04  0.306770E-02  0.103722E-11
            0.841024E+04  0.289568E-02  0.924153E-12
            0.865645E+04  0.273330E-02  0.823414E-12
            0.890987E+04  0.258003E-02  0.733657E-12
            0.917070E+04  0.243536E-02  0.653684E-12
            0.943917E+04  0.229879E-02  0.582428E-12
            0.971550E+04  0.216989E-02  0.518940E-12
            0.999992E+04  0.204821E-02  0.462372E-12"""
        
        with open("dustkappa_silicate.inp", "w") as outfile:
            print(f"{silicate_kappas_header[0]}\n{silicate_kappas_header[1]}", file = outfile)
            print(silicate_kappas.replace("    ", ""), file = outfile)
        
        nspec = 1   #number of dust species, here only silicate
        opacity_header = [2, nspec]
        dashes = "-------------------------------"
        silicates = [1, 0, "silicate"]  #1/10 for dustkappa/dustkapscatmat; 0/1 for thermal/quantu heating (quantum apparently doesn't even work yet); name used in dustkappa filename
        with open("dustopac.inp", "w") as outfile:
            print(f"{opacity_header[0]}\n{opacity_header[1]}\n{dashes}", file = outfile)
            for thing in silicates:
                print(thing, file = outfile)
            print(dashes, file = outfile)

    def do_all(self, nthet = 300, relax = True, radial_relax = 400, log = True,  angular_relax = 4200, no = 4, npix=500., wav=870, incl=60., phi=0, sizeau=80., buffer = 8192):
        start_time = time.time()
        self.fargo_read_fields(no=no)
        if relax: self.flat_relax(radial = radial_relax, angular = angular_relax, log = log)
        self.inner_outer_finish(r2 = 30, r3 = 0.005)
        self.radmc_write_dust_density(binary = True, nthet = nthet, buffer_size=buffer)
        self.radmc_grid()
        os.system('radmc3d mctherm')
        image.makeImage(npix=npix, wav=wav, incl=incl, phi=phi, sizeau=sizeau)
        im = image.readImage()
        print(f"Everything done, it took {time.time()-start_time:.2f} s")
        image.plotImage(im, au=True, log=True, maxlog=13, saturate=1e-7, cmap=plt.cm.get_cmap("gist_heat"))
        return [(time.time()-start_time)/60/60, radial_relax, 2*nthet, angular_relax]

    def radmc_write_inputs(self, mrw = None, thickness = f_z, nthet = 100, binary = True, buffer_size = 4096, **kwargs):
        """
        Creates all the necessary files for running radmc3d script
        --------------------
        Parameters:
        --------------------
        mrw: bool, determines whether modified random walk should be used. If unspecified, will calculate the optical thickness of the disk and sets the parameter acoordingly

        thickness: function, by which the volumetric density around the disk is determined. It is expected to take parameters of the point above which the density is needed:
            h [m]: float, height of the point of interest above the disk
            equatorial_r [m]: float, distance from the star to the point in equatorial plane under the point of interest
            T [K]: float, temperature of the dust at the point in equatorial plane under the point of interest
            sigma [kg/m²]: float, surface density of the dust at the point in equatorial plane under the point of interest
            M_star [kg]: float, mass of the star at the center
        
        nthet: float, number of cells in theta direction to be made above the disk. The same amount is generated beneath the disk as well

        buffer_size: int, max number of floats to be stored in RAM at any given point

        **kwargs: see radmc3d manual, chapter MAIN INPUT AND OUTPUT FILES OF RADMC-3D, section INPUT: radmc3d.inp. There's like a brazillion of inputs, you can't demand from me to write them all here
        """
        if mrw is None:
            depth = self.optical_depth(from_star=True, wav = 10, message = False, thickness=thickness)
            if depth > 1e7:
                print(f"Optical depth is quite high ({depth:.3e}), setting modified_random_walk to true in radmc3d.inp file...")
                mrw = 1
            else: mrw = 0

        radmc_inp_params = {
            "istar_sphere": 1,  #whether to take star as a sphere (1) or a point (0)
            "itempdecoup": 1,   #all dust species are thermally independet
            "lines_mode": -1,   #default, needs extra files to work properly
            "modified_random_walk": int(mrw),   #if 1 and if photon is stuck for too long in a cell, the probability of where it will exit is calculated and the photon is transported there
            "nphot": 1000000,    #number of photons for thermal siulation
            "nphot_scat": 300000,   #number of photons for image-making
            "nphot_spec": 100000,   #number of photons for spectra-making
            "rto_style": 3,     #1 for output files to be ascii, 3 for binary, 2 for pain (old and as of now obsolete fortran format)
            "scattering_mode_max": 1,   #if 1, radmc3d ignore anisotropic scattering (it's not like I provided it in the opacity files anyway)
            "tgas_eq_tdust": 1, #if 1, gas temperature will be the same as the first dust species' temp
        }

        for kw in kwargs:
            radmc_inp_params[kw] = kwargs[kw]

        with open("radmc3d.inp", "w") as radmcinp:
            for name, value in radmc_inp_params.items():
                print(f"{name} = {value}", file = radmcinp)
        self.gas_to_dust()
        self.radmc_write_dust_density(binary = binary, thickness = thickness, nthet=nthet, buffer_size=buffer_size)
        self.radmc_grid()
        self.radmc_wavelength()
        self.radmc_stars()
        self.radmc_write_opacity()

    def radmc_write_temperature(self, binary=True, note = ""):
        """
        Creates dust_temperature.inp from fargo temperature data for image making.
        """
        nrspecs = 1
        nrcells = np.size(self.T)
        if binary: 
            file = open(note+"dust_temperature.bdat", "wb")
            header = np.array([1, 8, nrcells, nrspecs])
            header.tofile(file)
            np.transpose(self.T, (2,1,0)).tofile(file)
        else:
            file = open(note+"dust_temperature.dat", "w")
            header = np.array([1, nrcells, nrspecs])
            header.tofile(file, sep = "\n")
            file.write("\n")
            np.transpose(self.T, (2,1,0)).tofile(file, sep = "\n")
        a = np.fromfile("dust_temperature.bdat", dtype = np.double, offset = 4*8)
        pass

    def radmc_read_temperature(self, isbinary = None, filename = None):
        """
        Reads radmc3d dust_temperature.inp file. It's usually quite large so watch out
        --------------------
        Parameters:
        -------------------- 
        isbinary: bool, if True, will try and load the file as binary, if False, will try and load as text, if None, will first check each line of the file to check whether it's binary

        filename: string, name of the file to be loaded. If None will look for typical filename: 'dust_temperature*'
        """

        if self.theta_med is None: raise ValueError("Temperature error: Disk hasn't been thickened yet, no knowledge of what shape the temperature is! Load the grid first.")

        if filename is None:
            import glob
            radm_files = glob.glob("dust_temperature*")
            if len(radm_files) > 1: raise IOError("Found multiple radmc3d temperature files")
            #far_files = glob.glob("gastemper*.dat")
            #if len(far_files) > 1: raise IOError("Found multiple fargo temperature files")
            files = radm_files#+far_files
            if len(files) > 1: raise IOError("Found both fargo and radmc3d temperature files. Specify file name")
            filename = files[0]

        if isbinary is None:
            print("Read temperature notice: unknown file type, checking for binary...")
            isbinary = is_file_binary(filename)

        if isbinary: self.T = np.fromfile(filename, dtype = np.double, offset = 4*8)
        else: self.T = np.loadtxt(filename, skiprows = 3)

        self.T = np.reshape(self.T, (len(self.phi_med), len(self.theta_med), len(self.r_med)))
        self.T = np.transpose(self.T, (2,1,0))
        self.T = np.reshape(self.T, (len(self.r_med), len(self.theta_med), len(self.phi_med)))

    def radmc_read_density(self, isbinary = None, filename = None):
        """
        Reads radmc3d dust_density.inp file. It's usually quite large so watch out
        """

        if self.theta_med is None: raise ValueError("Density error: Disk hasn't been thickened yet, no knowledge of what shape the density is! Load the grid first.")

        if filename is None:
            import glob
            radm_files = glob.glob("dust_density*")
            if len(radm_files) > 1: raise IOError("Found multiple radmc3d density files")
            filename = radm_files[0]

        if isbinary is None:
            print("Read density notice: unknown file type, checking for binary...")
            isbinary = is_file_binary(filename)

        if isbinary: self.density = np.fromfile(filename, dtype = np.double, offset = 4*8)
        else: self.density = np.loadtxt(filename, skiprows = 3)

        self.density = np.reshape(self.density, (len(self.phi_med), len(self.theta_med), len(self.r_med)))
        self.density = np.transpose(self.density, (2,1,0))
        self.density = np.reshape(self.density, (len(self.r_med), len(self.theta_med), len(self.phi_med)))*cm*cm*cm/gram

    def radmc_read_grid(self, filename = None):

        if filename is None:
            import glob
            radm_files = glob.glob("amr_grid*")
            if len(radm_files) > 1: raise IOError("Found multiple radmc3d grid files")
            filename = radm_files[0]

        loaded = np.fromfile(filename, sep = " ")
        n_r =    int(loaded[7])
        n_thet = int(loaded[8])
        n_phi =  int(loaded[9])

        self.r_inf = loaded[10:10+n_r]*cm/au
        self.r_sup = loaded[10+1:10+n_r+1]*cm/au
        self.r_med = 0.5*(self.r_inf+self.r_sup)
        self.par["Nrad"] = n_r

        self.theta_inf = loaded[10+n_r+1:  10+n_r+n_thet+1]
        self.theta_sup = loaded[10+n_r+2:10+n_r+n_thet+2]
        self.theta_med = 0.5*(self.theta_inf+self.theta_sup)
        self.par["Nthet"] = n_thet

        self.phi_inf = loaded[10+n_r+n_thet+2:  10+n_r+n_thet+n_phi+2]
        self.phi_sup = loaded[10+n_r+n_thet+3:10+n_r+n_thet+n_phi+3]
        self.phi_med = 0.5*(self.phi_inf+self.phi_sup)
        self.par["Nsec"] = n_phi

        print(f"Updated internal grid, which now consists of {n_r}×{n_thet}×{n_phi} r×θ×φ cells")

    def interpolate_temp(self, cutoff = 0.95):
        """
        Fills in holes in the temperature grid with the interpolation of cells neighboring it in the phi direction
        """
        print("Correcting disk's temperature...")
        total = 0
        for r_index in range(len(self.r_med)):
            for theta_index in range(len(self.theta_med)):
                shifted_left = np.roll(self.T[r_index, theta_index,:], -1)
                shifted_right = np.roll(self.T[r_index, theta_index,:], 1)
                average_of_neighbours = 0.5*(shifted_left+shifted_right)
                to_fix = np.where(self.T[r_index, theta_index,:]<average_of_neighbours*cutoff)[0]   #this [0] is there cuz np.where returns tuple of arrays, but the second element is nothing in this case
                total += len(to_fix)
                for bad_index in to_fix:
                    self.T[r_index, theta_index, bad_index]=average_of_neighbours[bad_index]
        print(f"Altered {total} out of {np.size(self.T)} cells.")
        
    def update_densities(self, evaporation_temp = 20):
        """
        Evaporates silicates by halving its density in each cell that surpases 'evaporation_temp'
        
        """
        self.radmc_read_temperature()
        to_half = np.where(self.T > evaporation_temp, .5, 1)
        self.radmc_read_density()
        self.density() *= to_half


    def add_heating(self):
        """
        Tries to add some temperature as aresult of simplified viscose heating to make up for the defference between fargo and radmc temps
        TBD
        https://arxiv.org/pdf/2005.09132.pdf ??
        As of yet unnecessary¿?
        """
        #kappa = ?? #opacity ??
        #nu = ?? ##viscosity
        
        def additional_T4(r, sigma):
            F = 3*np.pi*sigma*nu
            return 3*G*self.par["M_star"]*F*3*kappa*sigma/(8*np.pi*steff*8*r^3)
        
        self.T = (self.T^4+additional_T4(self.r_med, self.sigma))^.25

    def add_heatsource(self):
        """
        Adds heatsource file for radmc3d
        maybe to be used as a source of viscous heating?
        """

        #self.radmc_read_density()
        outfile = open("heatsource.inp", "w")
        outfile.write(f"1\n{len(self.phi_med)*len(self.theta_med)*len(self.r_med)}\n")
        for phi in self.phi_med:
            for theta in self.theta_med:
                for r in self.r_med:
                    #normuj hustotou!!!!!! (najít spodní limit pro kdy je viskozita dulezita)
                    if theta > 2*np.pi/5:
                        outfile.write("1e-80\n")
                    else: outfile.write("0\n")

        outfile.close()


    def kopecek(self, r = 8, size = 0.3):
        """
        Makes a circular bump/gap in the density 
        TBD
        """
        r_index = index_before(self.r_med, r)
        const = self.density[r_index,0]
        def gaussian(x):
            return np.exp(-(x-r)**2)*const*size
        rho = self.density.T
        rho+=gaussian(self.r_med)
        self.density = rho.T

        const = self.T[r_index,0]
        temp = self.T.T
        temp += gaussian(self.r_med)
        self.T = temp.T

def main():
    """Test program for the Disk class"""
    disk = Disk()

    """casova narocnost"""
    #import cProfile
    #import pstats
    #profile = cProfile.Profile()
    #profile.runcall(disk.do_all)
    #ps = pstats.Stats(profile)
    #ps.sort_stats("cumtime")
    #ps.print_stats()

    nrad, nthet, nphi = 2000, 250, 1000
    disk.fargo_read_fields(no = 4)
    disk.plot_fargo_gasdens()
    disk.inner_outer_finish()
    #disk.plot_fargo_gasdens(show = False)
    #disk.plot_fargo_gastemper(Mean=False)
    disk.flat_relax(angular = nphi, radial = nrad)
    #fig1=plt.figure()
    #disk.plot_T_profile(label="Fargo output", show = False, fig = fig1)
    #disk.kopecek(r = 8, size = 5)
    
    disk.radmc_write_inputs(nthet = nthet, nphot = 4000000, buffer_size = 4096*4)
    #disk.add_heatsource()
    #disk.radmc_read_density()
    #print(disk.optical_depth())
    #input()
    os.system('radmc3d mctherm setthreads 6 countwrite 100000')
    #disk.radmc_read_density()
    #disk.radmc_read_temperature()

    im = image.readImage()
    image.plotImage(im, au=True, log=True, maxlog=6, saturate=1, cmap=plt.cm.gist_heat, pltshow = True)

    """hustota vs teplota"""
    #plt.scatter(disk.density.ravel(), disk.T.ravel(), s = 0.5)
    #plt.xlabel("Volumetric density [kg·m¯³]")
    #plt.ylabel("Temperature [K]")
    #plt.title(f"{nrad} × {nthet} × {nphi} of cells")
    #plt.show()


    #disk.plot_scale_height(fig = fig, show = False)
    #disk.radmc_read_density()
    #disk.radmc_read_temperature()
    #disk.plot_T_slice(fig = fig, show = False)
    #disk.plot_grid(fig = fig, show = False)
    ##plt.ylim((0,1))
    #plt.show()



    #fig = plt.figure()
    #disk.radmc_read_temperature()
    #disk.plot_T_slice()
    #plt.show()
    
    #disk.radmc_read_temperature()
    #disk.plot_T_profile(show=False, fig = fig1, label="Heated ony by the star")
    #plt.legend()
    #plt.show()
    #disk.plot_T_profile()
    #from radmc3dPy import analyze
    #a = analyze.readSpectrum()
    #analyze.plotSpectrum(a, xlg = True, ylg = True, micron = True)
    #x = np.logspace(-1, 4, num = 10000)
    #y = black_body(x*1e-6, T = 4000)/400000000000000000
    #plt.plot(x,y)
    #plt.ylim((1e-14,1e-4))
    #plt.show()
    #disk.radmc_read_grid()
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=0.2, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=10, saturate=1e-5, cmap=plt.cm.gist_heat, pltshow = False)
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=5, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=10, saturate=1e-5, cmap=plt.cm.gist_heat, pltshow = False)
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=10, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=10, saturate=1e-5, cmap=plt.cm.gist_heat, pltshow = False)
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=10.4, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=10, saturate=1e-5, cmap=plt.cm.gist_heat, pltshow = False)
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=11, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=10, saturate=1e-5, cmap=plt.cm.gist_heat, pltshow = False)
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=40, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=10, saturate=1, cmap=plt.cm.gist_heat, pltshow = False)
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=100, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=8, saturate=1, cmap=plt.cm.gist_heat, pltshow = False)
    #fig = plt.figure()
    #image.makeImage(npix=300., wav=1000, incl=37., phi=0., sizeau=30.)
    #im = image.readImage()
    #image.plotImage(im, fig = fig, au=True, log=True, maxlog=6, saturate=1, cmap=plt.cm.gist_heat, pltshow = True)

    #disk.inner_outer_finish(r_newmax=50)
    #disk.density *= M_sun*0.05/disk.integrate_mass()
    #disk.plot_T_profile(show = False, phi = 20)
    #disk.gas_to_dust()
    #disk.flat_relax(radial=10, angular=6, log=True)
    #fig = plt.figure()
    #disk.plot_T_profile(fig = fig, show = False, label = "fargo 2D output")
    #disk.radmc_write_inputs(nthet = 10, nphot = 40000000, rto_style = 3, mrw = True)

    #disk.radmc_read_temperature()
    #disk.plot_T_profile(fig = fig, label = "radmc 3D output", show = False)
    #plt.legend()
    #plt.show()
    #fig = plt.figure()
    #disk.plot_T_profile(along = "perpendicular", fig = fig, show = False,label = "r = 3 [au]", r = 3)
    #disk.plot_T_profile(along = "perpendicular", fig = fig, show = False,label = "r = 5 [au]", r = 5)
    #disk.plot_T_profile(along = "perpendicular", fig = fig, show = False,label = "r = 7 [au]", r = 7)
    #disk.plot_T_profile(along = "perpendicular", fig = fig, show = False,label = "r = 9 [au]", r = 9)
    #plt.yticks(ticks = np.linspace(0,250, num = 11))
    #plt.xticks(ticks = np.linspace(-12, 12, num = 9))
    #plt.title(f"Temperature of the disk at φ = 0°\nHeated with 4·10⁷ photons")
    #plt.legend()
    #plt.show()
    
    #disk.interpolate_temp()
    #disk.plot_T_profile(perpendicular = True, phi = 20, note = "Heated with 4·10⁷ photons, corrected for cold cells")


    
    #disk.plot_sigma_profile(r_range=[0,3.5])
    #disk.optical_depth(from_star=True)
    #input("Press Enter to continue...")
    
    
    

    #disk.radmc_write_dust_density(binary = False, nthet = 5)
    #disk.plot_fargo_gastemper()
    #plt.show()

    
    

## TO DO:
##zkus upravit hustotu na různý kopečky a mezery a tak
#zkus vygrafit u planety teploty
#hermitova interpolace
#vypočítej si hodně zhruba rozšíření čáry
#simulátor almy??
#alma ost?
#přes spectrum plancka (černý těleso ve správný vzdálenosti, intenzita )

##Jansky jednotky
##metalicita pro prach a plyn — pro opacitu neni od věci přenásobit siguḿu 0.01
##  flock 2013?
##udělej jako fci možná r možná T (lepčí), we'll see

#v/p3c validator html


##Dust -> gas


#normuj hustotou!!!!!! (najít spodní limit pro kdy je viskozita dulezita)






if __name__ == "__main__":
    main()