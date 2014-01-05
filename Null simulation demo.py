# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Simulations of a visible nulling coronagraph

# <markdowncell>

# POPPY uses the Fraunhofer approximation of diffraction.
# 
# How good is the Fraunhofer approx. for propogation of the waves in the interferomer:
# 
# The Fresnel number divides the Fresnel and Fraunhofer diffraction regimes, 
# 
# $ F_n=\frac{a^2}{\lambda L}$
# 
# Fraunhofer is appropriate for small Fresnel numbers.
# 
# 
# 
# References:
# 
# Perrin, Marshall D., Rémi Soummer, Erin M. Elliott, Matthew D. Lallo, and Anand Sivaramakrishnan. 2012. “Simulating Point Spread Functions for the James Webb Space Telescope with WebbPSF.” In Proc. SPIE, 8442:84423D–84423D–11. doi:10.1117/12.925230. http://dx.doi.org/10.1117/12.925230.
# 
# Mendillo, Chris. 2013. “SCATTERING PROPERTIES OF DUST IN ORION AND THE EPSILON ERIDANI EXOPLANETARY SYSTEM.” April. PhD Dissertation, Boston University.
# 

# <codecell>

#import packages:
import null
import nulling_utils
import os
#from poppy_local
import poppy 
import matplotlib.pyplot as plt
import astropy

# <codecell>

#setup optical system
total_counts_sec=3e8
osys = poppy.OpticalSystem(oversample=10)
osys.addPupil(function='Circle', radius=.25,pad_factor = 1.0)# pupil radius in meters
osys.addDetector(pixelscale=0.18, fov_arcsec=23)  # image plane coordinates in arcseconds
#calculated scale to pupil of PICTURE telescope:
dm_meters_pixel=.00048816816816816817

# <markdowncell>

# ##Create an ideal nuller instance and one with an deformable mirror with high frequency error.
# 
# 

# <codecell>

nuller_ideal=null.NullingCoronagraph(osys,intensity_mismatch=.01, display_intermediates=True,normalize='not', shear=0.3)

#keywork phase_mismatch_fits is measurement of a deformable mirror surface figure.
#keywork phase_flat_fits is a low pass filtered version of the phase mismatch file, simulating a flattened, active, closed loop DM.
#pupil mask would add the central obscuration etc, it is left out in this example for simplicity.
nuller=null.NullingCoronagraph(osys,intensity_mismatch=.01, display_intermediates=True,normalize='not', shear=0.3,
                               phase_mismatch_fits='FITS/splinecongriddedcroppedrawdmdata.fits',
                                    phase_mismatch_meters_pixel=dm_meters_pixel,phase_flat_fits='FITS/boxcarred30pix.dm.fits')

# <codecell>

nuller_ideal.null()
nuller.null()

# <headingcell level=2>

# Dark output

# <codecell>

ax=subplot(121)
nuller_ideal.wavefront.display(nrows=2,ax=ax,colorbar=True)
ax=subplot(122)
nuller.wavefront.display(nrows=2,ax=ax,colorbar=True)

# <headingcell level=4>

# Interferometer bright output

# <codecell>

ax=subplot(121)

nuller_ideal.wavefront_bright.display(nrows=2,colorbar=True,ax=ax)
ax=subplot(122)

nuller.wavefront_bright.display(nrows=2,colorbar=True,ax=ax)

# <codecell>

figure(figsize=[4,4])
title("Ideal Nuller, Dark/vs Bright output contrast",fontsize=7)
imshow(log10(nuller_ideal.wavefront.intensity/nuller_ideal.wavefront_bright.intensity))
colorbar()
figure(figsize=[4,4])
title("Nuller with real DM, Dark/vs Bright output contrast",fontsize=7)

imshow(log10(nuller.wavefront.intensity/nuller.wavefront_bright.intensity))
colorbar()

# <markdowncell>

# ##Polychromatic nulling:

# <codecell>

#use map convention to run locally, same syntax applies PiCloud's cloud.map():

waves=tuple(np.arange(15)*0.01e-6+0.6e-6)
weight=1.0/float(len(waves))
#don't have nullwave function return bright output, necc. for PiCloud bandwidth restrictions:
returnBright=False 
#turn off plotting so matplotlib doesn't fill up memory:
nuller.display_intermediates=False
psf=map(nulling_utils.nullwave,
        [nuller for i in range(len(waves))],
        list(waves),
        [weight for i in range(len(waves))],
        [np.array([[0],[0],[0]]) for i in range(len(waves))],
        [total_counts_sec for i in range(len(waves))],
        [returnBright for i in range(len(waves))])

# <codecell>


#add arrays, (different dimensions depending on whether map returns a bright array or not)
if returnBright:
        darkfits1024 = astropy.io.fits.HDUList(astropy.io.fits.PrimaryHDU(np.array(psf)[:,0,:,:].sum(axis=0)))
if not returnBright:
        darkfits1024 = astropy.io.fits.HDUList(astropy.io.fits.PrimaryHDU(np.array(psf).sum(axis=0)))        
#annotate FITS header:
nulling_utils.add_nuller_to_header(darkfits1024[0],nuller)

imshow(log10(darkfits1024[0].data))
title("Oversampled Polychromatic Leakage for nuller with real DM")
colorbar()

# <codecell>


# <codecell>


