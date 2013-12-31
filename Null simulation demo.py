# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import null
import nulling_utils
import os
#from poppy_local
import poppy 
import matplotlib.pyplot as plt
import astropy

total_counts=5e8
secondary_dia=4.40*0.0254# inches*m/inch -> [meters]...
osys = poppy.OpticalSystem(oversample=10)
osys.addPupil(function='Circle', radius=.25,pad_factor = 1.0)          # pupil radius in meters
osys.addDetector(pixelscale=0.18, fov_arcsec=23)  # image plane coordinates in arcseconds
leak=0.005
dm_meters_pixel=secondary_dia/(9.25*24.75)
nuller=null.NullingCoronagraph(osys,intensity_mismatch=.2, display_intermediates=True,normalize='not', shear=0.3,
                               phase_mismatch_fits='FITS/splinecongriddedcroppedrawdmdata.fits',
                                    phase_mismatch_meters_pixel=dm_meters_pixel,phase_flat_fits='FITS/boxcarred30pix.dm.fits')

# <codecell>

nuller.null()

# <headingcell level=2>

# Dark output

# <codecell>

nuller.wavefront.display(nrows=2,colorbar=True)

# <headingcell level=4>

# Interferometer bright output

# <codecell>

nuller.wavefront_bright.display(nrows=2,colorbar=True)

# <codecell>


# <codecell>


