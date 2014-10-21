import poppy
from matplotlib.colors import LogNorm, Normalize  # for log scaling of images, with automatic colorbar support
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib
import logging
import numpy as np
import astropy
import time

_log = logging.getLogger('poppy')
print("logging")
_log.setLevel(logging.DEBUG)
_log.debug("logging")

# internal constants for types of plane
_PUPIL = 1
_IMAGE = 2
_DETECTOR = 3 # specialized type of image plane.
_ROTATION = 4 # not a real optic, just a coordinate transform
_typestrs = ['', 'Pupil plane', 'Image plane', 'Detector', 'Rotation']

poppy.Conf.use_fftw.set(True)
poppy.Conf.enable_speed_tests.set(True)
poppy.Conf.autosave_fftw_wisdom.set(True)
def sheararray(inputarray,shear,pixelscale):
    """
    Inputs:
    inputarray - a numpy array
    shear - a shear in units of pixel scale
    pixel scale - units of length per pixel (i.e. meters per pixel)

    Returns:
    an array rolled by the distance shear
    
    Example:
    wavefront_arm.wavefront = sheararray(wavefront_arm.wavefront,self.shear,wavefront_arm.pixelscale) #sheared

    """
    npix_shear=np.int64(np.round(shear/pixelscale))
    _log.debug("shearing by %3f pixels"%npix_shear)
    sheared = np.roll(inputarray,npix_shear)
    return sheared

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
        The shear between the apertures/pupils, in units of meters, Default=0.3
    nrows: int,
        for plotting. default is 6.
    intensity_mismatch: float
        constant intensity mismatch between arms. default is zeros
    phase_mismatch_fits: hdulist, filename or False
        array of phase mismatch values. default is False.
    phase_mismatch_meters_pixel: float
        scaling number of meters per pixel for mapping of phase mismatch to the pupil plane. 
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

        
        if self.phase_mismatch_fits:
            self.init_wfe_error()
    def init_wfe_error(self):
        '''Read the HDULIst or FITSOptical element self.phase_mismatch_fits and
            transforms it into the internal FITSOpticalElement "self.DM_array"
            (though it doesn't have to be a deformable mirror)
            also if self.phase_flats_fits exists, it will be subtracted off of self.DM_array,
             allowing simulation of adaptive corrections to the input
             phase mismatch fits.
        '''
            
        _log.debug("initializing a phase error in this class")

        #this also filters out the dead actuators.
        #let the dead actuators through: not implimented.
        #DM pupil.
        if (type(self.phase_mismatch_fits)==astropy.io.fits.hdu.hdulist.HDUList) | (type(self.phase_mismatch_fits) == type('string')):
            _log.debug("phase mismatch is an HDUList")
            try:
                self.DM_array = poppy.FITSOpticalElement(opd=self.phase_mismatch_fits, pixelscale=self.phase_mismatch_meters_pixel,oversample=self.oversample,opdunits='meters',rotation=0)
            except Exception,err:
                _log.warn(err)
        elif type(self.phase_mismatch_fits)==poppy.FITSOpticalElement:
            _log.debug("phase mismatch is a FITSOptical Element")
            DM_array = self.phase_mismatch_fits

        else:
            _log.warn("phase mismatch is not a FITS HDUList, trying to use it as if it's a FITSOpticalElement")
            DM_array = self.phase_mismatch_fits
            
            #a low passed version to subtract, simulating flattening the DM:
            if self.phase_flat_fits:
                _log.debug("A phase_flat_fits exists, if compatible, it will be subtracted from self.DM_array")

                # DM pupil:
                if type(self.phase_flat_fits)==astropy.io.fits.hdu.hdulist.HDUList:
                    DM_flat=poppy.FITSOpticalElement(opd=self.phase_flat_fits,pixelscale=self.phase_mismatch_meters_pixel,
                                                     oversample=self.oversample,opdunits='meters',rotation=0 )
                if type(self.phase_flat_fits) == type('string'):
                    _log.debug("phase_flat_fits is a string, trying to open as a fits file")
                    try:
                        DM_flat = poppy.FITSOpticalElement(opd=astropy.io.fits.open(self.phase_flat_fits), pixelscale=self.phase_mismatch_meters_pixel,oversample=self.oversample,opdunits='meters',rotation=0)
                    except Exception,err:
                        _log.warn(err)
                else:
                    try:
                        _log.debug("phase flat is not a FITS HDUList, trying to use it as if it's a FITSOpticalElement.")
                        DM_flat=self.phase_flat_fits
                        phase_error=DM_array.opd-DM_flat.opd
                        self.DM_array=poppy.FITSOpticalElement(opd=astropy.io.fits.HDUList(astropy.io.fits.ImageHDU(phase_error)), pixelscale=self.phase_mismatch_meters_pixel,oversample=self.oversample,opdunits='meters',rotation=0)
                        #self.DM_array.opd= sheararray(self.DM_array.opd,self.shear/2.,self.DM_array.pixelscale)
                    except Exception,err:
                        _log.warn(err)
            else:
                phase_error=DM_array.opd
                self.DM_array=poppy.FITSOpticalElement(opd=astropy.io.fits.HDUList(astropy.io.fits.ImageHDU(phase_error)), pixelscale=self.phase_mismatch_meters_pixel,oversample=self.oversample,opdunits='meters',rotation=0)

            self.DM_array.opd= sheararray(self.DM_array.opd,-self.shear/2.,self.DM_array.pixelscale)
            _log.debug("initialized:"+str(self.DM_array))
                        
    def null(self,wavelength=0.633e-6,wave_weight=1,flux=1.0,offset_x=0.0,offset_y=0.0,prebuilt_wavefront=False):
        '''
        nulls the Nulling Coronagraph according to the optical system prescription.
        After null runs, the nullstatus is set to True.
        Returns: a tuple of the dark and bright outputs: (self.wavefront, wavefront_bright).
        Flux should be counts/second.
        '''
        nrows=6
        if poppy.Conf.enable_speed_tests():
            t_start = time.time()
        if prebuilt_wavefront:
            if prebuilt_wavefront.__class__ ==poppy.poppy_core.Wavefront:
                wavefront=prebuilt_wavefront.copy()
                _log.debug("copying a prebuilt input wavefront:")
                if self.display_intermediates:
                    wavefront.display(what='other',nrows=nrows,row=1, colorbar=True)
            else:
                raise _log.error("prebuilt_wavefront is not a wavefront class.")
        else:
            wavefront = self.inputWavefront(wavelength)
            _log.debug("Generated a new input wavefront, plate scale is:"+str(wavefront.pixelscale))


        if  poppy.Conf.enable_flux_tests(): _log.debug("Wavefront initialized,  Flux === "+str(wavefront.totalIntensity))

        if self.save_intermediates:
            raise NotImplemented("not yet")
        if self.display_intermediates:
            suptitle = plt.suptitle( "Propagating $\lambda=$ %.3f $\mu$m" % (self.wavelength*1.0e6), size='x-large')

            #plt.clf()
            #wavefront.display(what='intensity',nrows=nrows,row=1, colorbar=False)
            wavefront *= self.inputpupil
        if self.defocus:
            wavefront *= self.defocus
        wavefront.normalize()
        wavefront *= np.sqrt(flux)
        _log.debug("Normalized all planes, after the unobscured aperture, then multiplied by flux/aperture %s",str(flux))
        if  poppy.Conf.enable_flux_tests(): _log.debug("Wavefront multiplied by flux/aperture,  total intensity === "+str(wavefront.totalIntensity))
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


        wavefront_arm = wavefront.copy()
        wavefront_bright= wavefront.copy()

        if not self.pupilmask:
            mask= wavefront_ideal.wavefront + sheararray(wavefront_ideal.wavefront,self.shear,wavefront_ideal.pixelscale)
            mask_array=np.zeros(wavefront.shape)
            mask_array[np.where(mask  >1.1 )]=1.0
            mask_array[np.where(mask < 1)]=0
            #force area wrapped back over the leading edge to zero:
            mask_array[:,0:int(round(self.shear/wavefront.pixelscale))]=0
            #recenter:
            self.mask_array = sheararray(mask_array,-self.shear/2.0,wavefront.pixelscale)

        else:

            if (type(self.pupilmask)==astropy.io.fits.hdu.hdulist.HDUList) |   (type(self.pupilmask)==type('string')):
                _log.debug("pupilmask is an HDUList or string" )
                self.FITSmask = poppy.FITSOpticalElement(transmission=self.pupilmask, pixelscale=self.phase_mismatch_meters_pixel,oversample=self.oversample,opdunits='meters',rotation=0)
            elif type(self.pupilmask)==poppy.FITSOpticalElement:
                _log.debug("self.pupilmask  is a FITSOptical Element")
                self.FITSmask = self.pupilmask
        
            else:
                _log.warn("pupilmask is not a FITS HDUList, trying to use it as if it's a FITSOpticalElement")
                self.FITSmask = self.pupilmask
                
            #self.FITSmask=poppy.FITSOpticalElement(transmission=self.pupilmask,planetype=_PUPIL,rotation=0,oversample=self.oversample,pixelscale=self.phase_mismatch_meters_pixel)   
            #print(self.FITSmask.pixelscale)

            #offset mask onto the sheared array
            #self.mask_array = sheararray(self.FITSmask.amplitude,self.shear/2.,self.FITSmask.pixelscale)
            self.mask_array = self.FITSmask.amplitude#,self.shear/2.,self.FITSmask.pixelscale)

            #calculate the effect of phase differences between the arms:



        
        try:
            _log.debug("RMS OPD error mismatch, (includes beyond mask):"+str(np.mean(np.sqrt(self.DM_array.opd**2))))
            _log.debug("Mean RMS OPD error in mismatched arm, (includes beyond mask):"+str(np.mean(np.sqrt(self.DM_array.opd**2))))
            _log.debug("Mean RMS Amplitude error in mismatched arm, (includes beyond mask):"+str(np.mean(np.sqrt(self.DM_array.amplitude**2))))

            #_log.debug("Mean RMS OPD error in mismatched arm, (only within mask):"    +str(np.mean(np.sqrt((self.DM_array.mask_array)**2))))
            _log.debug("DM_array plate scale is:"+str(self.DM_array.pixelscale))
            _log.debug("DM_array shape is:"+str(self.DM_array.amplitude.shape))

            wavefront_arm *= self.DM_array
            _log.debug("RMS phase error [radians] in mismatched arm:"+str(np.mean(np.sqrt(wavefront_arm.phase**2))))

        except Exception, err:
            _log.warn(err)
            _log.warn("is DM_array defined?")
        wavefront_arm.wavefront = sheararray(wavefront_arm.wavefront,self.shear,wavefront_arm.pixelscale) #sheared
        #interfere the arms, accounting for fractional intensity mismatch between the arms: 
        '''if self.display_intermediates:
            plt.figure()
            plt.subplot(121)
            plt.title("Wavefront arm OPD [radians]")
            plt.imshow(wavefront_arm.phase)#*wavefront.wavelength/(2.0*np.pi))
            plt.colorbar()
            plt.subplot(122)
            plt.title("wavefront arm phase * mask")
            plt.imshow(wavefront_arm.phase*self.mask_array)#*wavefront.wavelength/(2.0*np.pi))
            plt.colorbar()
            plt.figure()
            displaywavefrontarm=wavefront_arm.copy()
            displaywavefrontarm.wavefront=displaywavefrontarm.wavefront*self.mask_array
            displaywavefrontarm.wavefront=sheararray(displaywavefrontarm.wavefront,-self.shear,displaywavefrontarm.pixelscale)
            displaywavefrontarm.display(what='other',nrows=nrows,row=1, colorbar=True,vmax=wavefront_arm.amplitude.max(),vmin=wavefront_arm.amplitude.min())
        '''

        wavefront_combined = 0.5*(1.0 + self.intensity_mismatch)*wavefront.wavefront + 0.5*(-1.0 + self.intensity_mismatch)*wavefront_arm.wavefront
        wavefront_bright.wavefront = 0.5*(1.0 - self.intensity_mismatch)*wavefront.wavefront + 0.5*(1.0 + self.intensity_mismatch)*wavefront_arm.wavefront

        wavefront.wavefront=wavefront_combined

        #plt.imshow(mask_array)
        if self.display_intermediates:
            plt.figure()
            ax=plt.subplot(121)
            wavefront.display(what='phase',nrows=nrows,row=2, colorbar=True,vmax=wavefront.phase.max(),vmin=wavefront.phase.min(),ax=ax)
            plt.title("output phase")

        #recenter arrays, almost:
        
        wavefront.wavefront = sheararray(wavefront.wavefront,-self.shear/2.0,wavefront.pixelscale)
        wavefront_bright.wavefront = sheararray(wavefront_bright.wavefront,-self.shear/2.0,wavefront.pixelscale)

        wavefront.wavefront=wavefront.wavefront*self.mask_array
        wavefront_bright.wavefront=wavefront_bright.wavefront*self.mask_array
        


        if  poppy.Conf.enable_flux_tests(): _log.debug("Masked Dark output (wavefront),  Flux === "+str(wavefront.totalIntensity))
        if  poppy.Conf.enable_flux_tests(): _log.debug("Masked Bright output, (wavefront_bright),  Flux === "+str(wavefront_bright.totalIntensity))
        
        #plt.imshow(mask_array)
        if self.display_intermediates:
            plt.figure(figsize=[12,6])
            ax=plt.subplot(111)
            wavefront.display(what='phase',nrows=nrows,row=1, colorbar=True,vmax=wavefront.amplitude.max(),vmin=wavefront.amplitude.min())#,ax=ax)
            ax.set_title("interfered Phase")


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
        if poppy.Conf.enable_flux_tests():
             _log.debug(" Dark output in front of detector (wavefront),  Flux === "+str(wavefront.totalIntensity))
             _log.debug(" Bright output in front of detector (wavefront_bright),  Flux === "+str(wavefront_bright.totalIntensity))
        self.wavefront = wavefront#.wavefront #.asFITS()
        self.wavefront_bright = wavefront_bright#.wavefront #.asFITS()
        self.nullerstatus=True
        if poppy.Conf.enable_speed_tests():
            t_stop = time.time()
            deltat=t_stop-t_start
            if self.verbose: _log.info(" nulled in %g " % deltat)
        return(self.wavefront, wavefront_bright)
    # self.pupil_plane_dark =	wavefront.copy()


