"""
POPPY implementation of a lateral shearing interferometer. 
Units of length are meters.
"""

#import astropy.units as u
import numpy as np
import poppy 

def _sheararray(inputarray,shear,pixelscale,axis='y',wrap=False):
    """

    Parameters
    ----------
    inputarray :
        a 2D numpy array
    shear : 
        a shear in units of pixel scale
    pixel scale : float
         units of length per pixel (i.e. meters per pixel)
    axis : str
        shear along x or y axis? Default y
    wrap : bool
        should sheared values wrap around the array or be set to zero?
    
        
    Returns
    -------
    a newsheared  array 
    
    Example
    -------

    shearing by 0.3 meters:
        
        wavefront_arm.wavefront = sheararray(wavefront_arm.wavefront,0.3,wavefront_arm.pixelscale) #sheared

    """
    assert (inputarray.ndim == 2)
    
    npix_shear=np.int64(np.round(shear/pixelscale))
    #_log.debug("shearing by %3f pixels"%npix_shear)


    if npix_shear == 0:
        sheared= inputarray
    if wrap:
        sheared = np.roll(inputarray,npix_shear,axis=axis)
    else:
        #see http://stackoverflow.com/a/16401173
        if axis ==None: axis = 'x'
        if axis == 'x':
            if npix_shear<0:
                sheared=np.pad(inputarray,((0,0),(0,-npix_shear,)), mode='constant')[:, -npix_shear:]
            if npix_shear>0:
                sheared=np.pad(inputarray,((0,0),(+npix_shear,0)), mode='constant')[:, :-npix_shear]
        if axis == 'y':
            if npix_shear<0:
                sheared=np.pad(inputarray,((0,-npix_shear,),(0,0)), mode='constant')[ -npix_shear:,:]
            if npix_shear>0:
                sheared=np.pad(inputarray,((+npix_shear,0),(0,0)), mode='constant')[:-npix_shear,:]               
    return sheared

def shearing_nuller(wavefront,shear=0.0,bright=False,shear_axis='x'):
        
    """
    Interferes a wavefront with itself via lateral shearing. 
    Defaults to destructive interference.
        
    Parameters
    ----------
    
    wavefront : poppy.Wavefront 
        Wavefront to mask.
    input_aperture : poppy.OpticalElement
        optical element used to define the initial beams
    shear : float
        The lateral shear between interferred wavefronts. Units of meters.
    bright : bool
        Switch to constructive interference if True.
    shear_axis : str
        shear along the x or y axis of the input wavefront

    
    
    """
    wavefront_arm = wavefront.copy()
    wavefront_arm.wavefront = _sheararray(wavefront_arm.wavefront,shear,wavefront_arm.pixelscale,axis=shear_axis) #sheared
    if not bright:
        #the dark output
        wavefront.wavefront = 0.5*wavefront.wavefront + 0.5*(-1.0)*wavefront_arm.wavefront
    else:
        #bright output wavefront
        wavefront.wavefront = 0.5*wavefront.wavefront + 0.5*(1.0)*wavefront_arm.wavefront


    #recenter:
    wavefront.wavefront = _sheararray(wavefront.wavefront,-shear/2.0,wavefront.pixelscale,axis=shear_axis)

def mask_noninterference(wavefront,input_aperture,shear=0.0,shear_axis='x'):
    
    """ 
    Masks regions of a pupil that will not interfere.
        
    Parameters
    ----------
    wavefront : poppy.Wavefront 
        Wavefront to mask.
    input_aperture : poppy.OpticalElement
        optical element used to define the initial beams
    shear : float
        The lateral shear between interferred wavefronts. Units of meters.

    """
    wavefront_ideal = wavefront.copy()
    wavefront_ideal.wavefront=np.ones(wavefront_ideal.shape)

    wavefront_ideal*=input_aperture
    mask = wavefront_ideal.wavefront + _sheararray(wavefront_ideal.wavefront,shear,
                                                          wavefront_ideal.pixelscale,
                                                          axis=shear_axis)
    mask_array=np.zeros(wavefront.shape)
    mask_array[np.where(mask  >1.1)]=1.0
    mask_array[np.where(mask < 1.0)]=0
    mask_array = _sheararray(mask_array,-shear/2.0,wavefront.pixelscale,axis=shear_axis)
    wavefront.wavefront*=mask_array