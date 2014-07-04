import poppy
from matplotlib.colors import LogNorm, Normalize  # for log scaling of images, with automatic colorbar support
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib
import logging
import numpy as np
import astropy
import time

_log = logging.getLogger('poppy-null')
print("logging")
_log.setLevel(logging.DEBUG)
_log.debug("logging")

# internal constants for types of plane
_PUPIL = 1
_IMAGE = 2
_DETECTOR = 3 # specialized type of image plane.
_ROTATION = 4 # not a real optic, just a coordinate transform
_typestrs = ['', 'Pupil plane', 'Image plane', 'Detector', 'Rotation']

class NullingCoronagraph(poppy.OpticalSystem):
    print('nulling class')
    """
    based on POPPY Module:SemiAnalyticCoronagraph



    Parameters
    -----------
    ExistingOpticalSystem : OpticalSystem
        An optical system which can be converted into a SemiAnalyticCoronagraph. This
        means it must have exactly 4 planes, in order Pupil, Image, Pupil, Detector.
    oversample : int
        Oversampling factor in intermediate image plane. Default is 8
    wavelength : float
        Default= is0.633e-6,
    normalize: str
        which plane to normalize to, if any, default is 'last' 
    save_intermediates: True/False
        Default False
    display_intermediates:True/False
        Default True
    shear: float
        The shear between the apertures, (or the baseline as a fraction of primary diameter), Default=0.3
    nrows: int,
        for plotting. default is 6.
    intensity_mismatch: float
        constant intensity mismatch between arms. default is zeros
    phase_mismatch_fits: hdulist, filename or False
        array of phase mismatch values. default is False.
    phase_mismatch_meters_pixel: float
        scaling number of meters per pixel for mapping of phase mismatch to the pupil plane. default 1.0/32.0
    phase_flat_fits:HDUlist,filename,False
        a phase flattening array, i.e. a low pass filtered DM phase map, to be subtracted from the phase mismatch, must have same scale. default false
    pupilmask: fits filename, hduList, False
        a mask to cut out the shearing artifacts or stray light, default False
    obscuration_fname: fits filename, False
                input pupil obscuration, i.e. secondary, uses phase_mismatch_meters_pixel scaling, defaults to False.
    verbose:
        Default is True,
    defocus: ThinLens(AnalyticOpticalElement) or False
        defocusing term to apply to the interferred pupil.
    """

    def __init__(self, ExistingOpticalSystem,
        oversample=False,
        wavelength=0.633e-6,
        normalize='last',
        save_intermediates=False, display_intermediates=True,
        shear=0.3,
        nrows=6,
        intensity_mismatch=0.000,
        phase_mismatch_fits=False,
        phase_mismatch_meters_pixel=0, phase_flat_fits=False,
        obscuration_fname=False,
        pupilmask=False,verbose=True,defocus=False,store_pupil=False):
        self.phase_mismatch_fits=phase_mismatch_fits
        self.pupilmask=pupilmask
        self.normalize=normalize
        self.name = "SemiAnalyticCoronagraph for "+ExistingOpticalSystem.name
        self.verbose =verbose
        self.source_offset_r = ExistingOpticalSystem.source_offset_r
        self.source_offset_theta = ExistingOpticalSystem.source_offset_theta
        self.planes = ExistingOpticalSystem.planes
        self.intensity_mismatch=intensity_mismatch
        self.wavelength=wavelength
        self.save_intermediates=save_intermediates
        self.display_intermediates=display_intermediates
        self.phase_mismatch_meters_pixel=phase_mismatch_meters_pixel
        self.shear=shear
        self.defocus=defocus
        self.nullerstatus=False
        self.phase_flat_fits=phase_flat_fits
        self._default_display_size= 20.
        self.obscuration_fname=obscuration_fname
        try:
            self.inputpupil = self.planes[0]
            # self.occulter = self.planes[1]
            #self.lyotplane = self.planes[2]
            self.detector = self.planes[-1]
        except Exception,err:
            print(err)
        self.oversample = oversample
        self.store_pupil=store_pupil

    def null(self,wavelength=0.633e-6,wave_weight=1,flux=1.0,offset_x=0.0,offset_y=0.0,prebuilt_wavefront=False):
        '''
        nulls the Nulling Coronagraph according to the optical system prescription.
        After null runs, the nullstatus is set to True.
        Returns: a tuple of the dark and bright outputs: (self.wavefront, wavefront_bright).
        Flux should be counts/second.
        '''
        if poppy.settings.enable_speed_tests():
            t_start = time.time()
        if prebuilt_wavefront:
            print(prebuilt_wavefront.__class__)
            if prebuilt_wavefront.__class__ ==poppy.poppy_core.Wavefront:
                wavefront=prebuilt_wavefront.copy()
                wavefront.display(what='other',nrows=6,row=1, colorbar=True)
                _log.debug("copying input wavefront, plate scale is:"+str(wavefront.pixelscale))

            else:
                raise _log.error("prebuilt_wavefront is not a wavefront class.")
        else:
            wavefront = self.inputWavefront(wavelength)
            _log.debug("Generated a new input wavefront, plate scale is:"+str(wavefront.pixelscale))


        if  poppy.settings.enable_flux_tests(): _log.debug("Wavefront initialized,  Flux === "+str(wavefront.totalIntensity))

        print(wavefront.wavefront.real.max())

        if self.save_intermediates:
            raise NotImplemented("not yet")
        if self.display_intermediates:
            suptitle = plt.suptitle( "Propagating $\lambda=$ %.3f $\mu$m" % (self.wavelength*1.0e6), size='x-large')

            nrows = 6
            #plt.clf()
            #wavefront.display(what='intensity',nrows=nrows,row=1, colorbar=False)
            wavefront *= self.inputpupil
        if self.defocus:
            wavefront *= self.defocus
        wavefront.normalize()
        wavefront *= np.sqrt(flux)
        _log.debug("Normalized all planes, after the unobscured aperture, then multiplied by incident flux %s",str(flux))
        if  poppy.settings.enable_flux_tests(): _log.debug("Wavefront multiplied by flux,  Flux === "+str(wavefront.totalIntensity))
        if self.obscuration_fname:
            self.obscuration=poppy.FITSOpticalElement(transmission=self.obscuration_fname,
                              pixelscale=self.phase_mismatch_meters_pixel,
                              oversample=self.oversample,opdunits='meters')
            #only apply obscuration if there is not going to be a pupil plane mask later.
            if not self.pupilmask:
                wavefront *= self.obscuration
                wavefront_ideal *=  self.obscuration

        wavefront_ideal = wavefront.copy()
        wavefront_ideal.wavefront=np.ones(wavefront.shape)
        wavefront_ideal *=  self.inputpupil

        if (offset_x!=0) or (offset_y !=0):
            wavefront.tilt(Xangle=offset_x, Yangle=offset_y)
            _log.debug("Tilted wavefront by theta_X=%f, theta_Y=%f arcsec, for target with flux of %f" % (offset_x, offset_y,flux))
        else:
            _log.debug("No Tilt. Target with flux of %f" % (flux))

        def sheararray(inputwavefront,shear):
            sheared = np.roll(inputwavefront,int(round(inputwavefront.shape[0]*shear)))
            return sheared

        wavefront_arm = wavefront.copy()
        wavefront_bright= wavefront.copy()

        if not self.pupilmask:
            mask= wavefront_ideal.wavefront + sheararray(wavefront_ideal.wavefront,self.shear)
            mask_array=np.zeros(wavefront.shape)
            mask_array[np.where(mask  >1.1 )]=1.0
            mask_array[np.where(mask < 1)]=0
            #force area wrapped back over the leading edge to zero:
            mask_array[:,0:int(round(wavefront.wavefront.shape[0]*self.shear))]=0
            self.mask_array = mask_array

        else:
            self.FITSmask=poppy.FITSOpticalElement(transmission=self.pupilmask,planetype=_PUPIL,rotation=-45,oversample=self.oversample)     
            print(self.FITSmask.pixelscale)
            #offset mask onto the sheared array
            self.mask_array = np.roll(self.FITSmask.amplitude,int(round(self.FITSmask.amplitude.shape[0]*self.shear)/2.0))

            #calculate the effect of phase differences between the arms:
        if self.phase_mismatch_fits:
            #this also filters out the dead actuators.
            #let the dead actuators through: not implimented.
            #DM pupil.
            if type(self.phase_mismatch_fits)==astropy.io.fits.hdu.hdulist.HDUList:
                DM_array=poppy.FITSOpticalElement(opd=self.phase_mismatch_fits,pixelscale=self.phase_mismatch_meters_pixel,oversample=self.oversample,opdunits='meters',rotation=225)
            else:
                _log.warn("phase mismatch is not a FITS HDUList, trying to use it as if it's a FITSOpticalElement.")
                DM_array=self.phase_mismatch_fits
                
        #a low passed version to subtract, simulating flattening the DM:
        if self.phase_flat_fits:
            # DM pupil:
            DM_flat=poppy.FITSOpticalElement(opd=self.phase_flat_fits,pixelscale=self.phase_mismatch_meters_pixel,oversample=self.oversample,opdunits='meters',rotation=225)
            DM_array.opd=DM_array.opd-DM_flat.opd
            #center DM on mask:
            DM_array.opd= sheararray(DM_array.opd,-self.shear/2.0)

            #    DM_array.opd=DM_array.opd
        try:
            _log.debug("RMS wavefront error in mismatched arm, (includes beyond mask):"+str(np.mean(np.sqrt(DM_array.opd**2))))
            _log.debug("Mean RMS wavefront error in mismatched arm, (includes beyond mask):"+str(np.mean(np.sqrt(DM_array.opd**2))))
            _log.debug("Mean RMS wavefront error in mismatched arm, (only within mask):"    +str(np.mean(np.sqrt((DM_array.opd*self.mask_array)**2))))
            _log.debug("DM_array plate scale is:"+str(DM_array.pixelscale))
            wavefront_arm *= DM_array

        except Exception, err:
            _log.warn(err)
            _log.warn("is DM_array defined?")
        wavefront_arm.wavefront = sheararray(wavefront_arm.wavefront,self.shear) #sheared
        #interfere the arms, accounting for fractional intensity mismatch between the arms: 
        if self.display_intermediates:
            plt.figure()
            plt.subplot(221)
            plt.title("Wavefront arm OPD [radians]")
            plt.imshow(wavefront_arm.phase)#*wavefront.wavelength/(2.0*np.pi))
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(wavefront_arm.phase*self.mask_array)#*wavefront.wavelength/(2.0*np.pi))
            plt.colorbar()
            plt.figure()
            displaywavefrontarm=wavefront_arm.copy()
            displaywavefrontarm.wavefront=displaywavefrontarm.wavefront*self.mask_array
            displaywavefrontarm.wavefront=sheararray(displaywavefrontarm.wavefront,-self.shear/2.0)
            displaywavefrontarm.display(what='other',nrows=2,row=1, colorbar=True,vmax=wavefront_arm.amplitude.max(),vmin=wavefront_arm.amplitude.min())


        wavefront_combined = 0.5*(1.0 + self.intensity_mismatch)*wavefront.wavefront + 0.5*(-1.0 + self.intensity_mismatch)*wavefront_arm.wavefront
        wavefront_bright.wavefront = 0.5*(1.0 - self.intensity_mismatch)*wavefront.wavefront + 0.5*(1.0 + self.intensity_mismatch)*wavefront_arm.wavefront

        wavefront.wavefront=wavefront_combined

        #plt.imshow(mask_array)
        if self.display_intermediates:
            plt.figure()
            ax=plt.subplot(121)
            wavefront.display(what='phase',nrows=nrows,row=1, colorbar=True,vmax=wavefront.amplitude.max(),vmin=wavefront.amplitude.min(),ax=ax)


        wavefront.wavefront=wavefront.wavefront*self.mask_array
        wavefront_bright.wavefront=wavefront_bright.wavefront*self.mask_array
        #recenter arrays, almost:
        wavefront.wavefront = sheararray(wavefront.wavefront,-self.shear/2.0)

        wavefront_bright.wavefront = sheararray(wavefront_bright.wavefront,-self.shear/2.0)

        if  poppy.settings.enable_flux_tests(): _log.debug("Masked Dark output (wavefront),  Flux === "+str(wavefront.totalIntensity))
        if  poppy.settings.enable_flux_tests(): _log.debug("Masked Bright output, (wavefront_bright),  Flux === "+str(wavefront_bright.totalIntensity))

        if self.store_pupil:  
            self.pupil_plane_dark=wavefront.wavefront.copy()     
            self.pupil_dm_arm=wavefront_arm.wavefront
        '''
	if self.display_intermediates:
		intens = wavefront.intensity.copy()
		phase  = wavefront.phase.copy()
        phase[np.where(intens ==0)] = 0.0
		   
		_log.debug("Mean RMS wavefront error in combined, masked wavefront:"   +str(wavefront.wavelength*(np.mean(np.sqrt(phase**2)))/(2*np.pi)))
		plt.figure()
		ax=plt.subplot(111)
		wavefront.display(what='other',nrows=2,row=1, colorbar=True,vmax=wavefront.amplitude.max(),vmin=wavefront.amplitude.min())
		plt.figure()
		ax2=plt.subplot(111)
		ax2.imshow(np.log10(phase))#wavefront.wavelength*/(2*np.pi))
		ax2.set_title("Phase errors, [$log_{10}$(radians)]")
		plt.colorbar(ax2.images[0])
		plt.figure()
		ax3=plt.subplot(111)
		ax3.set_title("Oversampled Pupil Intensity Map [$log_{10}$(counts)]")
		ax3.imshow(np.log10(wavefront.intensity))#wavefront.wavelength*/(2*np.pi))
		plt.colorbar(ax3.images[0])
		#suptitle.remove() #  does not work due to some matplotlib limitation, so work arount:
		suptitle.set_text('') # clean up before next iteration to avoid ugly overwriting
        '''
        wavefront.propagateTo(self.detector)
        wavefront_bright.propagateTo(self.detector)
        if poppy.settings.enable_flux_tests():
             _log.debug(" Dark output in front of detector (wavefront),  Flux === "+str(wavefront.totalIntensity))
             _log.debug(" Bright output in front of detector (wavefront_bright),  Flux === "+str(wavefront_bright.totalIntensity))
        self.wavefront = wavefront#.wavefront #.asFITS()
        self.wavefront_bright = wavefront_bright#.wavefront #.asFITS()
        self.nullerstatus=True
        if poppy.settings.enable_speed_tests():
            t_stop = time.time()
            deltat=t_stop-t_start
            if self.verbose: _log.info(" nulled in %g " % deltat)
        return(self.wavefront, wavefront_bright)
    # self.pupil_plane_dark =	wavefront.copy()



def downsample_display(input,block=(10,10),
		       save=False,
		       filename='DownsampledOut.fits',
		       vmin=1e-8,vmax=1e1,
		       ax=False,norm='log',add_noise=False):
	'''
	takes a wavefront's intensity, and generates a downsampled fits image for display and saving to disk.
	'''
	print(str(type(input)))
	if str(type(input)) == "<class 'astropy.io.fits.hdu.hdulist.HDUList'>":
		inFITS=input
	else:
		try:
			inFITS=input.asFITS()
		except Exception, err:
			print(err)
			raise ValueError("Type not recognized as wavefront")
	if ax==False:
		plt.figure()
		ax = plt.subplot(111)
	
	cmap = matplotlib.cm.jet
	halffov_x = inFITS[0].header['PIXELSCL']*inFITS[0].data.shape[1]/2
	halffov_y = inFITS[0].header['PIXELSCL']*inFITS[0].data.shape[0]/2
	extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]
	unit="arcsec"
	if norm=="log":
		norm=LogNorm(vmin=vmin,vmax=vmax)
	else:
		norm=Normalize(vmin=vmin,vmax=vmax)
	plt.xlabel(unit)
	downsampled=downsample(inFITS[0].data,block=block)
	titlestring=str(inFITS[0].data.shape)+" array, downsampled by:"+str(block)
	plt.title(titlestring)
	poppy.utils.imshow_with_mouseover(downsampled,ax=ax, interpolation='none',  extent=extent, norm=norm, cmap=cmap)
	plt.colorbar(ax.images[0])
	outFITS = fits.HDUList(fits.PrimaryHDU(data=downsampled,header=inFITS[0].header))
	newpixelscale=inFITS[0].header['PIXELSCL']*block[0]
	outFITS[0].header.update('PIXELSCL', newpixelscale, 'Scale in arcsec/pix (after oversampling and subsequent downsampling)')
	outFITS[0].header.add_history(titlestring)
	try:
		outFITS.writeto(filename)
	except Exception, err:
		print(err)
        return outFITS




