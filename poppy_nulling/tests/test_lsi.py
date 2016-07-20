
import astropy.units as u
import numpy as np
import poppy 
from .. import lsi
def test_lsi_intensity_mismatch():
    detector = poppy.Detector(pixelscale=.05,fov_arcsec=10)
    aperture = poppy.optics.CircularAperture(radius=.25)
    
    #measure contrast
    #compute dark fringe PSF
    wf_fraunhofer = poppy.Wavefront(diam=0.5, wavelength=675e-9,npix=1024,oversample=4)
    wf_fraunhofer *= aperture
    lsi.shearing_nuller(wf_fraunhofer, shear=.15*u.m,bright=False,shear_axis='x',
                    one_arm_error=.99**0.5) #add a 1% reflectivity mismatch to one arm
    lsi.mask_noninterference(wf_fraunhofer, aperture,shear=0.15*u.m,shear_axis='x')

    wf_fraunhofer.propagateTo(detector)
    dark_psf=wf_fraunhofer.copy()
    #COMPUTE BRIGHT FRINGE PSF:
    wf_fraunhofer = poppy.Wavefront(diam=0.5, wavelength=675e-9,npix=1024,oversample=4)
    wf_fraunhofer *= aperture
    lsi.shearing_nuller(wf_fraunhofer, shear=.15*u.m,bright=True,shear_axis='x',
                    one_arm_error=.99**0.5) #add a 1% intensity mismatch to one arm
    lsi.mask_noninterference(wf_fraunhofer, aperture,shear=0.15*u.m,shear_axis='x')
    wf_fraunhofer.propagateTo(detector)
    bright_psf=wf_fraunhofer.copy()
    lsi_null_depth=dark_psf.intensity.sum()/bright_psf.intensity.sum()
    print("Simulation null depth: %.4e"%lsi_null_depth)
    
    #Check against Analytic Values
    
    e=-0.01
    #fractional deviation from input beam ***intensity*** 
    #whereas the scalar arm mismatch above is multiplied by the wavefront amplitude
    I_1=1
    I_2=1+e
    dI = (I_1-I_2)/((I_1+I_2))
    N_serabyn=(dI)**2/4. #eq (4) Serabyn 2000, Proc SPIE. 
    print("First Order Null Depth: %.4e"%N_serabyn)
    
    N=(2.+e-2.0*(1.+e)**0.5)/(2.+e+2.0*(1.+e)**0.5) #eq. B.7 of Douglas 2016, PhD thesis.
    print("null depth: %.4e"%N)
    
    assert np.round(lsi_null_depth,decimals=15) == np.round(N,decimals=15)

