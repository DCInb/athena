import numpy as np
import matplotlib.pyplot as plt
import athena_read as ar
import OrbitAnalysisUtils as ou
import Constants 

from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm.auto import tqdm
#from skimage import measure
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit
from athena_read import athdf
import os
import sys
import subprocess
from typing import List

c=Constants.Constants()

from scipy import optimize

# EOS(radiation pressure and ideal gas pressure)
# P/rho = 4*c.sigmaSB*x**4/3/c.c/rho+c.kB*x/1.25*2/c.mp

def temp_rad_gas(rho,press):
    f = lambda x, rho,epsilon: 4*c.sigmaSB*x**4/3/c.c/rho+c.kB*x/1.25*2/c.mp-epsilon
    fder = lambda x, rho,epsilon: 16*c.sigmaSB*x**3/3/c.c/rho+c.kB/1.25*2/c.mp
    
    x = 1e3*np.ones_like(rho)
    epsilon = press/rho
    
    return optimize.newton(f, x, fprime=fder, args=(rho,epsilon, ), maxiter=200)

#  interp functions
def get_interp_function(d,var,rescale_factor=1,method = 'nearest'): #'
    """
    MM: Use RegularGridInterpolator to pass data to interpolating function for a given variable
    Parameters
    -----------
    d : dict
       athena data dict from read_data
    var: str
       name of variable to be interpolated
       
    Returns
    --------
    var_interp: an interpolating function that can be called with a tuple (phi,theta,r)
    """
    dph = np.gradient(d['x3v'])[0]
    dtheta = np.gradient(d['x2v'])[0]
    two_pi = ( (d['x3v'][-1]-d['x3v'][0]+dph) /(2*np.pi) > 0.99 ) # boolean to determine if spans 2pi in phi
    pi_th = ( (d['x2v'][-1]-d['x2v'][0]+dtheta) /(np.pi) > 0.99 ) # boolean to determine if spans 2pi in phi
    x1v = d['x1v']
    var_shape = d[var].shape
    
    if two_pi:
        x3v = np.append(d['x3v'][0]-dph,d['x3v'])
        x3v = np.append(x3v,x3v[-1]+dph)
        var_data = np.append([d[var][-1]],d[var],axis=0)
        var_data = np.append(var_data,[var_data[0]],axis=0)
        var_shape = var_data.shape
    else:
        x3v = d['x3v']
        var_data = d[var]
        
    # extend in theta
    if pi_th:
        x2v = np.append(d['x2v'][0]-dtheta,d['x2v'])
        x2v = np.append(x2v,x2v[-1]+dtheta)
        var_the0 = var_data[:,0,:].reshape(var_shape[0],1,var_shape[2])
        half = var_shape[0] // 2
        var_the0 = np.concatenate((var_the0[half:], var_the0[:half]), axis=0)
        var_data = np.append(var_the0,var_data,axis=1)
        var_the1 = var_data[:,-1,:].reshape(var_shape[0],1,var_shape[2])
        var_the1 = np.concatenate((var_the1[half:], var_the1[:half]), axis=0)
        var_data = np.append(var_data,var_the1,axis=1)
    else:
        x2v = d['x2v']
        
    var_interp = RegularGridInterpolator((x3v,x2v,x1v),var_data,bounds_error=False,method=method)
    return var_interp

def cart_to_polar(x,y,z):
    """cartesian->polar conversion (matches 0<phi<2pi convention of Athena++)
    Parameters
    x, y, z
    Returns
    r, th, phi
    """
    r = np.sqrt(x**2 + y**2 +z**2)
    th = np.arccos(z/r)
    phi = np.arctan2(y,x)
    phi = np.where(phi>=0,phi,phi+2*np.pi)
    return np.stack((r, th, phi), axis=2)

def mesh_interpolate_at_xyzpoints(d,var,points):
    """
    MM: convience function to interpolate a variable to mesh points
    Parameters
    -----------
    d: athena++ data dict
    var: str variable name in, e.g. "rho"
    points: array of cartesian positions (eg vertices or centroids) (n,n,3) floats (x,y,z)
    """
    var_interp = get_interp_function(d,var)
    rp,thp,php = cart_to_polar(points[:,:,0],points[:,:,1],points[:,:,2])
    return var_interp( (php,thp,rp) )

### integrate along radius from every solid angle
mylevel = None

# considering Hydrogen and ff opacity
def ff_opacity(rho, temp, wavelength):
    # in cgs unit
    opacity = 4.97*rho**2/temp**0.5*wavelength**3*1000
    return opacity

def Rosseland_mean_opacity(rho,temp):
    # in cgs unit
    opacity = 8e22*temp**(-7/2)*rho**2
    return opacity

def opac(rho,temp,X,Z):
    # in cgs units
    # molecules
    kappa_m = 0.1*Z
    
    # negative hydrogen ion
    kappa_nH = 1.1e-25*Z**0.5*rho**0.5*temp**7.7
    
    # electron scattering
    kappa_e = 0.2*(1+X)/(1+2.7e11*rho/temp**2)/(1+(temp/4.5e8)**0.86)
    
    # Kramers formula
    kappa_K = 4e25*(1+X)*(Z+0.001)*rho/temp**3.5
    
    opacity = (kappa_m+1.0/(1.0/kappa_nH+1/(kappa_e+kappa_K)))*rho
    
    return opacity

def run_exe_batch(exe_path, logT_array: np.ndarray, logR_array: np.ndarray) -> np.ndarray:
    """
    Run the .exe with numpy arrays of logT and logR values via stdin and return outputs.
    NOTE: Opacity tables must exist in the current directory. Currently only support 
    "gs98_z0.02_x0.7.data" and "lowT_fa05_gs98_z0.02_x0.7.data".
    
    Args:
        logT_array: ND array of logT values
        logR_array: ND array of logR values (same shape as logT_array)
    
    Returns:
        ND array of output values matching input shape
    """
    # Validation checks
    if logT_array.shape != logR_array.shape:
        raise ValueError("Input arrays must have identical dimensions")
    if logT_array.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Path resolution
    exe_full_path = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[0])) if hasattr(sys, 'argv') else os.getcwd(),
        exe_path
    )
    if not os.path.exists(exe_full_path):
        raise FileNotFoundError(f"Executable not found at {exe_full_path}")

    # Array preprocessing
    original_shape = logT_array.shape
    logT_flat = logT_array.ravel()
    logR_flat = logR_array.ravel()

    # Optimized input generation
    header = f"{logT_array.size}\n"
    body = "\n".join(f"{t:.6f} {r:.6f}" for t, r in zip(logT_flat, logR_flat))
    input_data = header + body + "\n"

    # Process execution with buffering
    with subprocess.Popen(
        [exe_full_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=2**20  # 1MB buffer for large datasets
    ) as proc:
        stdout, stderr = proc.communicate(input=input_data)
    
    if proc.returncode != 0:
        raise RuntimeError(f"Execution failed: {stderr.strip()}")

    # Memory-efficient output parsing
    try:
        outputs = np.fromstring(stdout, sep='\n', dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Output parsing failed: {e}\nRaw output: {stdout[:200]}...")

    if outputs.size != logT_array.size:
        raise RuntimeError(f"Expected {logT_array.size} outputs, got {outputs.size}")
    
    return outputs.reshape(original_shape)

def getLogR(logT,logRho):
    return logRho-3*logT+18

def observer_grid(n, box_half_length = 1.0865e11, method = 'uniform',direction ='z'):
    #n*n*3 rectangle gird
    
    if method == 'uniform':
        obs_grid_xx = box_half_length*np.ones((n,n))
        pos_yy = np.linspace(-box_half_length, box_half_length, n)
        pos_zz = pos_yy
        obs_grid_yy,obs_grid_zz = np.meshgrid(pos_yy, pos_zz)
        
        if direction == 'x':
            obs_grid = np.stack((obs_grid_xx, obs_grid_yy, obs_grid_zz), axis=2)
            
        if direction == 'z':
            obs_grid = np.stack((obs_grid_yy, obs_grid_zz, obs_grid_xx), axis=2)
            
        if direction == 'y':
            obs_grid = np.stack((obs_grid_yy, obs_grid_xx, obs_grid_zz), axis=2)
        if direction == '-y':
            obs_grid = np.stack((obs_grid_yy, -obs_grid_xx, obs_grid_zz), axis=2)
    return obs_grid

def optical_depth(data, wavelength, obs_grid, rescale_factor=1, dx = -1, direction = 'x',opac_model="analytical",plot = True):
    """Marching the obs_grid along given direction to find the photo-spheric position
    
    Parameters:
    -------
    data: athena++ data dict
    wavelength: observer wavelength in nm
    obs_grid: observer grid, (n,n) matrix
    direction: observer direction: 'x','y','z'; all from positive axis to negtive
    opac_model: model option for opacity, options: "analytical", "mesa_exe"
    
    Returns: (n,n,3) array with photo-spheric position in cartesian coordinates, found with coordinate<9*domain_size, not found with
    coordinate=10*domain size
    """
    r0_thersh=0.6
    ## initializition
    n = len(obs_grid)
    op_dep = np.zeros((n,n))
    marching_box_xyz = obs_grid.copy()
    
    # computitional domain size
    domain_size = np.max(obs_grid)
    #print(domain_size)
    cell_size = domain_size/(n-1)*2
    area = cell_size**2
    pho_position = np.ones((n,n,3))*domain_size*10
    
    ## get interpolation function
    rho_star = np.where(data['r0']>r0_thersh,data['rho']*data['r0']/rescale_factor**2,0)
    rho_star_dict = {'rho_star':rho_star}
    data.update(rho_star_dict)
    interp_rho = get_interp_function(data,'rho_star',rescale_factor=rescale_factor)
    # temp = temp_rad_gas(data['rho'],data['press'])
    temp = data['press']/data['rho']*c.mp*1.25/2/c.kB/rescale_factor
    temp_dict = {'temp':temp}
    data.update(temp_dict)
    interp_temp = get_interp_function(data,'temp',rescale_factor=rescale_factor)
    
    if opac_model=="mesa_exe":
        script_dir = os.getcwd()
        exe_path = os.path.join(script_dir, "opacMesa.exe")
        opacity=10**run_exe_batch(exe_path, np.log10(temp),getLogR(np.log10(temp),np.log10(data['rho'])))
        opac_dict = {'opacity':opacity}
        data.update(opac_dict)
        interp_opac = get_interp_function(data,'opacity',rescale_factor=rescale_factor)

    if dx == -1:
        dx = cell_size/10
        
    if direction == 'x':
        marching_pos = np.min(marching_box_xyz[:,:,0])
    if direction == 'y':
        marching_pos = np.min(marching_box_xyz[:,:,1])
    if direction == '-y':
        marching_pos = np.min(marching_box_xyz[:,:,1])
    if direction == 'z':
        marching_pos = np.min(marching_box_xyz[:,:,2])
    
    flag = -1 if marching_pos<0 else 1
    dx= flag*dx
    
        
    # use success matrix to determine if the line of sight is on the star,-1 for no, 1 for yes, 0 for first success
    success= -1*np.ones((n,n)) 
        
    #print('marching_pos', marching_pos)
    while flag*marching_pos>-domain_size:
        # marching half step
        marching_pos = marching_pos-dx/2
        if direction == 'x':
            marching_box_xyz[:,:,0] -= dx/2
        if direction == 'y' or direction == '-y':
            marching_box_xyz[:,:,1] -= dx/2
        if direction == 'z':
            marching_box_xyz[:,:,2] -= dx/2
        
        # get density and temperature to get opacity
        marching_box_rtp = cart_to_polar(marching_box_xyz[:,:,0],marching_box_xyz[:,:,1],marching_box_xyz[:,:,2])
        rho_box = interp_rho((marching_box_rtp[:,:,2],marching_box_rtp[:,:,1],marching_box_rtp[:,:,0]))
        rho_box = np.where(np.isnan(rho_box), 10**10, rho_box)
        #print(rho_box)
        temp_box = interp_temp((marching_box_rtp[:,:,2],marching_box_rtp[:,:,1],marching_box_rtp[:,:,0]))
            
        # integral optical depth
        #op_dep_new = op_dep + ff_opacity(rho_box,temp_box,wavelength)*dx
        #op_dep_new = op_dep + Rosseland_mean_opacity(rho_box,temp_box)*dx
        if opac_model=="analytical":
            op_dep_new = op_dep + opac(rho_box,temp_box,0.7,0.02)*abs(dx)
        elif opac_model=="mesa_exe":
            op_dep_new = op_dep + interp_opac((marching_box_rtp[:,:,2],marching_box_rtp[:,:,1],marching_box_rtp[:,:,0]))*rho_box*abs(dx)
        
        op_dep = np.where(op_dep>=1,op_dep,op_dep_new)
        success = np.where(op_dep>=1,success+1,success)
        
        # once success, add adaptive refinement grid
        #if np.all(op_dep_new_new==op_dep):
        #    pass
            
        # marching half step
        marching_pos = marching_pos-dx/2
        if direction == 'x':
            marching_box_xyz[:,:,0] -= dx/2
        if direction == 'y':
            marching_box_xyz[:,:,1] -= dx/2
        if direction == 'z':
            marching_box_xyz[:,:,2] -= dx/2    

        
            
        # if first success, restore the success coordinates
        pho_position[:,:,0] = np.where(success == 0.,marching_box_xyz[:,:,0],pho_position[:,:,0])
        pho_position[:,:,1] = np.where(success == 0.,marching_box_xyz[:,:,1],pho_position[:,:,1])
        pho_position[:,:,2] = np.where(success == 0.,marching_box_xyz[:,:,2],pho_position[:,:,2])
                
        
        
            
        success = np.where(success>=1,1,success)
            
       
    # using the radius to say the photo-spheric radius at the surface or not(0.9R_S--1.1R_S)
    #pho_radius = (pho_position[:,:,0]**2+pho_position[:,:,1]**2+pho_position[:,:,2]**2)**0.5/c.rsun
    #pho_radius = np.where(pho_radius>10,0,pho_radius)
    # plot
    if plot: 
        pho_radius = (pho_position[:,:,0]**2+pho_position[:,:,1]**2+pho_position[:,:,2]**2)**0.5/c.rsun
        pho_radius = np.where(pho_radius>9.9*(domain_size/c.rsun)*rescale_factor,0,pho_radius)
        if direction =='x':
            plt.imshow(pho_radius)
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Photosphere radius Plot')
            plt.xlabel('$y \ \ [R_\odot]$')
            plt.ylabel('$z \ \ [R_\odot]$')
            plt.show()
        if direction =='y':
            plt.imshow(pho_radius)
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Photosphere radius Plot')
            plt.xlabel('$x \ \ [R_\odot]$')
            plt.ylabel('$z \ \ [R_\odot]$')
            plt.show()
        if direction =='z':
            plt.imshow(pho_radius)
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Photosphere radius Plot')
            plt.xlabel('$x \ \ [R_\odot]$')
            plt.ylabel('$y \ \ [R_\odot]$')
            plt.show()
        
    return pho_position

def Blackbody(wavelength,temp):
    """
    Planck blackbody emission, in cgs units, wavelength in nm
    """
    a = 2*c.h*c.c**2
    b = c.h*c.c/(wavelength*1e-7)/c.kB/temp
    return a/(wavelength*1e-7)**5/(np.exp(b)-1.0)


def flux_pho(data,pho_pos,grid,wavelength,rescale_factor=1,lum_bolo=False, plot = True, direction ='z'):
    """
    Synthetic photometry, Planck blackbody emission flux
    """
    # using the radius to say the photo-spheric radius at the surface or not(0.9R_S--1.1R_S)
    domain_size = np.max(grid)
    pho_radius = (pho_pos[:,:,0]**2+pho_pos[:,:,1]**2+pho_pos[:,:,2]**2)**0.5/c.rsun
    pho_radius = np.where(pho_radius>9*(domain_size/c.rsun)*rescale_factor,0,pho_radius)
    
    ## get interpolation function
    interp_rho = get_interp_function(data,'rho_star',rescale_factor=rescale_factor)
    interp_temp = get_interp_function(data,'temp',rescale_factor=rescale_factor)
    interp_r0 = get_interp_function(data,'r0',rescale_factor=rescale_factor)
    
    # computitional domain size
    domain_size = np.max(grid)
    n = len(grid)
    cell_size = domain_size/(n-1)*2
    area = cell_size**2
    
    # flux intensity 
    pho_pos_rtp = cart_to_polar(pho_pos[:,:,0],pho_pos[:,:,1],pho_pos[:,:,2])
    pho_temp = interp_temp((pho_pos_rtp[:,:,2],pho_pos_rtp[:,:,1],pho_pos_rtp[:,:,0]))
    pho_r0 = interp_r0((pho_pos_rtp[:,:,2],pho_pos_rtp[:,:,1],pho_pos_rtp[:,:,0]))
    
    #print(pho_temp)
    #print(interp_rho((pho_pos_rtp[:,:,2],pho_pos_rtp[:,:,1],pho_pos_rtp[:,:,0])))
    if lum_bolo:
        flux_intensity = pho_temp**4*c.sigmaSB
        flux_intensity = np.where(pho_radius==0,0,flux_intensity)
        flux = flux_intensity*area
    else:
        flux_intensity = Blackbody(wavelength, pho_temp)
        flux_intensity = np.where(pho_radius==0,0,flux_intensity)
        flux = flux_intensity*area
       
    
    #print(np.where(pho_temp>1e15,pho_radius,0))
    
    # plot
    if plot: 
        if direction =='x':
            plt.pcolor(grid[0,:,1]/c.rsun,grid[:,0,2]/c.rsun,flux, cmap='inferno')
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Flux Plot')
            plt.xlabel('$y \ \ [R_\odot]$')
            plt.ylabel('$z \ \ [R_\odot]$')
            plt.show()
        if direction =='y':
            plt.pcolor(grid[0,:,0]/c.rsun,grid[:,0,2]/c.rsun,flux, cmap='inferno')
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Flux Plot')
            plt.xlabel('$x \ \ [R_\odot]$')
            plt.ylabel('$z \ \ [R_\odot]$')
            plt.show()
        if direction =='z':
            plt.pcolor(grid[0,:,0]/c.rsun,grid[:,0,1]/c.rsun,flux, cmap='inferno')
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Flux Plot')
            plt.xlabel('$x \ \ [R_\odot]$')
            plt.ylabel('$y \ \ [R_\odot]$')
            plt.show()

    return flux

def temp_flux(data,pho_pos,grid,wavelength,rescale_factor=1,plot = True, direction ='z'):
    """
    Synthetic photometry, Planck blackbody emission flux
    """
    
    # using the radius to say the photo-spheric radius at the surface or not(0.9R_S--1.1R_S)
    domain_size = np.max(grid)
    pho_radius = (pho_pos[:,:,0]**2+pho_pos[:,:,1]**2+pho_pos[:,:,2]**2)**0.5/c.rsun
    pho_radius = np.where(pho_radius>9*(domain_size/c.rsun)*rescale_factor,0,pho_radius)
    
    ## get interpolation function
    interp_rho = get_interp_function(data,'rho_star',rescale_factor=rescale_factor)
    interp_temp = get_interp_function(data,'temp',rescale_factor=rescale_factor)
    
    # computitional domain size
    domain_size = np.max(grid)
    n = len(grid)
    cell_size = domain_size/(n-1)*2
    area = cell_size**2
    
    # flux intensity 
    pho_pos_rtp = cart_to_polar(pho_pos[:,:,0],pho_pos[:,:,1],pho_pos[:,:,2])
    pho_temp = interp_temp((pho_pos_rtp[:,:,2],pho_pos_rtp[:,:,1],pho_pos_rtp[:,:,0]))
    #pho_temp = interp_temp((pho_pos_rtp[:,:,2],pho_pos_rtp[:,:,1],pho_pos_rtp[:,:,0]))
    #print(pho_temp)
    #print(interp_rho((pho_pos_rtp[:,:,2],pho_pos_rtp[:,:,1],pho_pos_rtp[:,:,0])))
    
    flux_intensity = np.where(pho_radius==0,np.nan,pho_temp)
    flux = flux_intensity
    
    #print(np.where(pho_temp>1e15,pho_radius,0))
    
    # plot
    if plot: 
        if direction =='x':
            plt.pcolor(grid[0,:,1]/c.rsun,grid[:,0,2]/c.rsun,np.log10(flux), cmap='inferno')
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Flux Plot')
            plt.xlabel('$y \ \ [R_\odot]$')
            plt.ylabel('$z \ \ [R_\odot]$')
            plt.show()
        if direction =='y':
            plt.pcolor(grid[0,:,0]/c.rsun,grid[:,0,2]/c.rsun,np.log10(flux), cmap='inferno')
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Flux Plot')
            plt.xlabel('$x \ \ [R_\odot]$')
            plt.ylabel('$z \ \ [R_\odot]$')
            plt.show()
        if direction =='z':
            plt.pcolor(grid[0,:,0]/c.rsun,grid[:,0,1]/c.rsun,np.log10(flux), cmap='inferno')
            plt.colorbar()  # Add a color bar to show the scale
            plt.title('Temperature Plot')
            plt.xlabel('$x \ \ [R_\odot]$')
            plt.ylabel('$y \ \ [R_\odot]$')
            plt.show()
        
    return flux

    
def light_curve(dirname, start_time, end_time, time_incre, wavelength, grid_size, plot = True, direction ='z',opac_model="analytical"):
    """
    Temporal evolution of the integrated luminosity
    """
    
    # find data file by time and loop over files
    start_id = round(start_time/time_incre)-10
    end_id = round(end_time/time_incre)+10
    file_num = end_id-start_id+1
    
    lum_time = np.zeros(file_num)
    time = np.zeros(file_num)
    
    ob_grid=observer_grid(grid_size,direction=direction) # should be fixed
    
    #from multiprocessing import Pool
    
    #with Pool(64) as p:
        #lum_time, time=p.map(integrated_luminosity, id_order,start_id,dirname,orb,mylevel,wavelength,ob_grid,direction)
        
    
   #""" 
    # Non-parrallel
    for id_order in range(file_num):
        # read data
        id_name = id_order+start_id
        datafile = dirname + "PEGM.out1."+str(id_name).zfill(5)+".athdf"
        data = ou.read_data(datafile,orb,gamma=1.66667,level=mylevel,
                       get_energy=False,profile_file=base_dir+'polytrope.dat')
        time[id_order] = data['Time']
        
        pho_pos = optical_depth(data,wavelength,ob_grid,direction=direction,opac_model=opac_model)
        flux_con = flux(data, pho_pos, ob_grid, wavelength,plot=False, direction=direction)
        int_lum = np.sum(flux_con)
        lum_time[id_order] = int_lum
   # """
    
    if plot:
        plt.plot(time,lum_time)
        plt.title('Light Curve')
        plt.xlabel('$Time$')
        plt.ylabel('$Luminosity$')
        plt.show()
        
def light_curve_parallel(id_name, plot_interv, wavelength, time, int_lum,ob_grid, direction,base_dir,opac_model):
    """
    Temporal evolution of the integrated luminosity(parallel execution)
    ----
    Argments: tuple of (id_name, wavelength)
    """
    
    
    orb = ou.read_trackfile(base_dir+"pm_trackfile.dat")
    mylevel = None
    
    datafile = base_dir + "PEGM.out1."+str(id_name*plot_interv).zfill(5)+".athdf"
    data = ou.read_data(datafile,orb,gamma=1.66667,level=mylevel,
                get_energy=False,profile_file=base_dir+'polytrope.dat')
    
    pho_pos = optical_depth(data,wavelength,ob_grid,direction=direction,opac_model=opac_model,plot=False)
    flux_con = flux_pho(data,pho_pos, ob_grid,wavelength,lum_bolo=False,plot = False, direction=direction)

    time[id_name] = data['Time']
    int_lum[id_name] = np.sum(flux_con, dtype=np.float64)
    
    
    
def light_curve_parallel_plot(plot_num, plot_interv, wavelength,base_dir,grid_size,direction,opac_model="analytical"):
    """
    Use multiprocessing.Process to parallel execution
    """
    from multiprocessing import Pool,Array,Process,Manager
    
     
    p_list = []
    manager = Manager()
    time = manager.Array('d',np.arange(plot_num))
    int_lum = manager.Array('d',np.arange(plot_num))
    ob_grid = observer_grid(100, box_half_length =grid_size, direction=direction)

     
    pool = Pool(20)
    for i in np.arange(plot_num):
        pool.apply_async(light_curve_parallel, args=(i,plot_interv,wavelength, time, int_lum, ob_grid,direction,base_dir,opac_model))
    
    pool.close()
    pool.join()

    print('done!')
    return time, int_lum       


     
    
def optical_depth_radial(data, wavelength, rescale_factor=1, dx = -1,opac_model="analytical",plot = True):
    """Marching along the radial direction from outter boundary to find photosphere radii
    
    Parameters:
    -------
    data: athena++ data dict
    wavelength: observer wavelength in nm
    Returns: 
    (n_phi,n_theta) array with photosphere radii('x1v') of each solid angle interval
    (n_phi,n_theta) array with photosphere radii id of each solid angle interval 
    """
    
    ## initializition
    n_theta = len(data['x2v'])
    n_phi   = len(data['x3v'])
    op_dep = np.zeros((n_phi,n_theta))
    pho_radii = np.ones((n_phi,n_theta))*data['x1v'][-1]
    pho_id    = np.ones((n_phi,n_theta))*int(len(data['x1v'])-1)
    op_dep_id = int(len(data['x1v'])-1)
    
    # use success matrix to determine if the line of sight is on the star,-1 for no, 1 for yes, 0 for first success
    success= -1*np.ones((n_phi,n_theta)) 
    
    ## get data
    # correction for r0
    r0_thresh =0.9
    rho_star = np.where(data['r0']>r0_thresh,data['rho']*data['r0'],0)
     
    #rho_star = data['rho']*data['r0']
    # temp = temp_rad_gas(data['rho'],data['press'])
    temp = data['press']/data['rho']*c.mp*1.25/2/c.kB

    if opac_model=="mesa_exe":
        script_dir = os.getcwd()
        exe_path = os.path.join(script_dir, "opacMesa.exe")
        opacity=10**run_exe_batch(exe_path, np.log10(temp),getLogR(np.log10(temp),np.log10(data['rho'])))

    while np.any(success==-1):
        
        rho_box  = rho_star[:,:,op_dep_id]
        temp_box = temp[:,:,op_dep_id]
            
        dx = data['x1f'][op_dep_id+1]-data['x1f'][op_dep_id]
        
        if opac_model=="analytical":
            op_dep_new = op_dep + opac(rho_box,temp_box,0.7,0.02)*dx
        elif opac_model=="mesa_exe":
            op_dep_new = op_dep + opacity[:,:,op_dep_id]*rho_box*dx
        
        op_dep_new = np.nan_to_num(op_dep_new, nan=0, posinf=1000, neginf=-1000)

        op_dep = np.where(op_dep>=1,op_dep,op_dep_new)
        success = np.where(op_dep>=1,success+1,success)
        
        # if first success, restore the success coordinates
        pho_radii = np.where(success==0,data['x1v'][op_dep_id],pho_radii)
        pho_id = np.where(success==0,op_dep_id,pho_id)
        
        """
        if np.any(success==0):
            pho_radii = np.where(success==0,data['x1v'][op_dep_id-1],pho_radii)
            pho_id = np.where(success==0,op_dep_id-1,pho_id)
            
        """
        
        #marching
        op_dep_id -= 1
        success = np.where(success>=1,1,success)
        
            
    # plot
    if plot: 
        plt.imshow(pho_radii.T/c.rsun)
        plt.colorbar()  # Add a color bar to show the scale
        plt.title('Photosphere radius Plot')
        plt.xlabel('$phi$')
        plt.ylabel('$theta$')
        plt.show()
        
    return pho_radii,pho_id

def flux_radial(data,pho_radii,pho_id,wavelength,rescale_factor=1,lum_bolo=False, plot = True, temp_plot = True):
    """
    Synthetic photometry, Planck blackbody emission flux
    F=\sigma*T^4*d\theta*d\phi*sin(theta)*r^2
    """
    
    ## initializition
    n_theta = len(data['x2v'])
    n_phi   = len(data['x3v'])
    rho_star = data['rho']*data['r0']
    # temp = temp_rad_gas(data['rho'],data['press'])
    temp = data['press']/data['rho']*c.mp*1.25/2/c.kB
    pho_temp = np.zeros_like(pho_radii,dtype=float)
    pho_rho = np.zeros_like(pho_radii,dtype=float)
    pho_dr = np.zeros_like(pho_radii,dtype=float)
    flux = np.zeros_like(pho_radii,dtype=float)
    
    for ii in range(n_phi):
        for jj in range(n_theta):
            r_id = int(pho_id[ii,jj])
            pho_temp[ii,jj] = temp[ii,jj,r_id]
            pho_rho[ii,jj] = rho_star[ii,jj,r_id]
            pho_dr[ii,jj] = (data['x1f'][r_id+1]-data['x1f'][r_id])
            
            
            
    area = np.pi/n_theta*2*np.pi/n_phi*np.outer(np.ones(n_phi),np.sin(data['x2v']))*pho_radii**2
    pho_aver_radii = np.sqrt(np.sum(area)/4/np.pi)
    pho_aver_rho = np.sum(pho_rho*area* pho_dr)/np.sum(area* pho_dr)
    
    if lum_bolo:
        flux_intensity = pho_temp**4*c.sigmaSB
        flux = flux_intensity*area
    else:
        flux_intensity = Blackbody(wavelength, pho_temp)
        flux_intensity = np.where(pho_radius==0,0,flux_intensity)
        flux = flux_intensity*area
    pho_aver_temp = (np.sum(flux)/np.sum(area)/c.sigmaSB)**0.25        
    
    # plot
    if plot: 
        plt.imshow(flux.T)
        plt.colorbar()  # Add a color bar to show the scale
        plt.title('Flux Plot')
        plt.xlabel('$phi$')
        plt.ylabel('$theta$')
        plt.show()
        
    if temp_plot:
        plt.imshow(pho_temp.T)
        plt.colorbar()  # Add a color bar to show the scale
        plt.title('Photosphere temperature Plot')
        plt.xlabel('$phi$')
        plt.ylabel('$theta$')
        plt.show()

    return flux,pho_aver_radii,pho_aver_temp,pho_aver_rho
        
def light_curve_radial_parallel(id_name, plot_interv, wavelength, time, int_lum, radii_aver, temp_aver,rho_aver,base_dir,opac_model):
    """
    Temporal evolution of the integrated luminosity(parallel execution)
    ----
    Argments: tuple of (id_name, wavelength)
    """
    
    
    orb = ou.read_trackfile(base_dir+"pm_trackfile.dat")
    mylevel = 0
    
    datafile = base_dir + "PEGM.out1."+str(id_name*plot_interv).zfill(5)+".athdf"
    data = ou.read_data(datafile,orb,gamma=1.66667,level=mylevel,
                get_energy=False,profile_file=base_dir+'polytrope.dat')
    
    op_radii,op_id = optical_depth_radial(data, wavelength,opac_model=opac_model,plot=False)
    flux,radii,temp,rho = flux_radial(data,op_radii,op_id,wavelength,lum_bolo=True,plot = False, temp_plot = False)
    time[id_name] = data['Time']
    int_lum[id_name] = np.sum(flux)
    radii_aver[id_name] = radii
    temp_aver[id_name] = temp
    rho_aver[id_name] = rho
    
    
    
def light_curve_radial_parallel_plot(plot_num, plot_interv, wavelength,base_dir,opac_model='analytical'):
    """
    Use multiprocessing.Process to parallel execution
    """
    from multiprocessing import Pool,Array,Process,Manager
    
     
    p_list = []
    manager = Manager()
    time = manager.Array('d',np.arange(plot_num))
    int_lum = manager.Array('d',np.arange(plot_num))
    radii_aver = manager.Array('d',np.arange(plot_num))
    temp_aver = manager.Array('d',np.arange(plot_num))
    rho_aver = manager.Array('d',np.arange(plot_num))
    
     
    pool = Pool(20)
    for i in np.arange(plot_num):
        pool.apply_async(light_curve_radial_parallel, args=(i,plot_interv,wavelength, time, int_lum, radii_aver, temp_aver,rho_aver,base_dir,opac_model))
    
    pool.close()
    pool.join()

    print('done!')
    return time, int_lum, radii_aver,temp_aver,rho_aver      

