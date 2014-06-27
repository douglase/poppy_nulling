import null
import os,time
import numpy as np
import poppy #initially setup with 0.2.8
#import gc #http://pymotw.com/2/gc/index.html#forcing-garbage-collection, didn't actually help, memory overflow was from plotting  unnnecc.
import pprint
import scipy.interpolate, scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
import logging
_log = logging.getLogger('poppy')
import astropy.io.fits as fits
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm, Normalize  # for log scaling of images, with automatic colorbar support
from numpy.lib.stride_tricks import as_strided as ast


def nullwave(newnuller,wavelength,weight,tiltlist,star_counts,returnBright):
	'''
	nullwave nulls a list of targets at a given wavelength. designed for batch processing, i.e. PiCloud.
	wavelength is in meters.
	weight is the weight of the wavelength in the stellar spectrum.
	tiltlist is a np. array of x and y values in arcsec and flux from that coordinate:  np.array([x,y,elementflux*np.ones(npoints)])
	star_counts is just that, for calculating the central star leakage term.
	Input tilt list should *not* have a central star.
	'''
	t_start = time.time()
        n_sources=len(tiltlist[0,:])
	
	print("Nulling star, flux"+str(star_counts*weight)+", wavel="+str(wavelength))
        newnuller.null(offset_x=0,
		offset_y=0,
		flux=star_counts*weight,
		wavelength=wavelength)
	partial_image=newnuller.wavefront.intensity
	partial_bright=newnuller.wavefront_bright.intensity
        refpsf=partial_image
	image=newnuller.wavefront.intensity
	bright_output=newnuller.wavefront.intensity


	for k in range(n_sources):
		print(k)
		flux=tiltlist[2,k]
		if flux == 0:
			continue
		print('starting null')
		#newnuller.null(offset_x=tiltlist[0,k],offset_y=tiltlist[1,k],flux=tiltlist[2,k])
                weightedflux=tiltlist[2,k]*weight #flux of the source object, times this nuller run's wavelength's weight.
		newnuller.null(offset_x=tiltlist[0,k],
			offset_y=tiltlist[1,k],
			flux=weightedflux,
			wavelength=wavelength)

		partial_image=newnuller.wavefront.intensity
		if returnBright:
			print('stacking bright output')
			partial_bright = newnuller.wavefront_bright.intensity
			bright_output = np.sum(np.dstack((bright_output,partial_bright)),axis=2)

		print('stacking dark output-science image')
		print('max',np.max(partial_image))
		image=np.sum(np.dstack((image,partial_image)),axis=2)
		

        print("time to null "+str(n_sources)+ " targets: "+str(time.time()-t_start))
	if returnBright:
		return image,bright_output
	else:
		return image


def add_nuller_to_header(primaryHDU,nuller):
	'''
	takes a FITS HDU and appends important nuller charactaristics.
	'''
	
	primaryHDU.header['PIXELSCL']=nuller.wavefront.asFITS()[0].header['PIXELSCL']
	primaryHDU.header.update("shear",str(nuller.shear))
	primaryHDU.header.update('phase mismatch',str(nuller.phase_mismatch_fits))
	primaryHDU.header.update('phase flat field (DM correction)',str(nuller.phase_flat_fits))
	primaryHDU.header.update('Intensity Mismatch',str(nuller.intensity_mismatch))
	primaryHDU.header.update('pupil mask',str(nuller.pupilmask))
	primaryHDU.header.add_history('name: '+str(nuller.name))

	#return primaryHDU
	
def TiltfromField(field,arcsec_per_pixel,zero_init_wavefront=True):
	'''
	takes a numpy array of flux values and returns a list of x and y offsets (in arcsec) and  flux values.
	'''
	
	center=int(np.round(np.shape(field)[0]/2.0))
	center_y=int(np.round(np.shape(field)[1]/2.0))

	npix=field.shape[0]

	points=np.array(np.where(field>0))

	tiltlist=np.zeros([3,points.shape[1]])
	
	for index in range(points.shape[1]):
		point=((points[0,index]),(points[1,index]))
		i=point[0]
		j=point[1]
		flux=field[point]
		tiltlist[2,index]=flux
		r_pixel=arcsec_per_pixel*np.sqrt((i-center)**2+(j-center_y)**2) #arcsec
		angle=np.angle(complex((j-center_y),(i-center))) #radians.
		offset_x = r_pixel *-np.sin(angle)  # convert to offset X,Y in arcsec
		offset_y = r_pixel * np.cos(angle)  # using the usual astronomical angle convention
		tiltlist[0,index]=offset_x
		tiltlist[1,index]=offset_y
        return tiltlist

def draw_circle(r,dr,amp,cube=False):
	'''r is in radius, r+dr is outer radius.
	Given a data cube this function puts each value on its own plane
	given a plane, each value is added to the 2D array

	each value has intensity equal to amp/(number of values).
	'''
	array=np.zeros([512,512])
	centerx=array.shape[0]/2.0
	centery=array.shape[1]/2.0
	num_targ=0.0
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if ((i-centerx)**2+(j-centery)**2 <= (r+dr)**2) and ((i-centerx)**2+(j-centery)**2 >= r**2):
				if not cube:
					array[i,j]=amp
					num_targ=num_targ + 1.0
				if cube:
					plane=array.copy()
					plane[i,j]=amp
					array=np.dstack((array,plane))
					num_targ=num_targ + 1.0
	#
	array=array/num_targ
	return array

def plot_inset(inFITS,x1, x2, y1, y2,zoom=2.5):
	'''
    plots the first array of the FITS hdulist inFITS and a zoomed inset of the subregion defined by [x1:x2,y1:y2]
    using example from:
    http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#insetlocator

    '''
    fig, ax = plt.subplots(figsize=[5,5])
	
	# prepare the demo image
	Z = inFITS[0].data
	Z2 = Z
	halffov_x = inFITS[0].header['PIXELSCL']*inFITS[0].data.shape[1]/2.0
	halffov_y = inFITS[0].header['PIXELSCL']*inFITS[0].data.shape[0]/2.0
	extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]
	ax.imshow(inFITS[0].data, extent=extent, interpolation="none",
          origin="lower")

	axins = zoomed_inset_axes(ax, zoom, loc=1) # zoom = 6
	axins.imshow(Z2, extent=extent, interpolation="nearest",
             origin="lower")

	# sub region of the original image
	axins.set_xlim(x1, x2)
	axins.set_ylim(y1, y2)

	plt.xticks(visible=False)
	plt.yticks(visible=False)

	# draw a bbox of the region of the inset axes in the parent axes and
	# connecting lines between the bbox and the inset axes area
	mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

def downsample(A, block= (2,2), subarr=False):
    '''
    downsample, a downsampling function
    Conserves flux.
    
    Take a 2D numpy array, A, break it into subarrays of shape bloc
    k and sum the counts in them, returning a new array of the summed blocks.

    if subarray=True, if the size of the last block spills out of the array the block 
    will be discarded, otherwise an error will be returned.
    
    striding approach from: http://stackoverflow.com/a/5078155/2142498
    
    as_strided() is not limited to the memory block of your array, so added  check of dimensions.
    http://scipy-lectures.github.io/advanced/advanced_numpy/index.html#stride-manipulation-label
    '''
    if (np.remainder(block[0],np.floor(block[0])) !=0) or (np.remainder(block[0],np.floor(block[0])) !=0) :
        raise ValueError("Block size must be integers")

    if (np.remainder(A.shape[0],block[0]) !=0):
        if subarr:
            A=A[0:block[0]*int(np.floor(A.shape[0]/block[0])),:]
        else:
            raise ValueError("not an integer number of blocks in first dimension")
            
    if (np.remainder(A.shape[1],block[1]) !=0):
        if subarr:
            A=A[:,0:block[1]*int(np.floor(A.shape[1]/block[1]))]
        else:
            raise ValueError("not an integer number of blocks in second dimension")
    #shape of new array:
        
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block

    #strides that fill the new array:
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    #create an array of sub_arrays
    
    blocked= ast(A, shape= shape, strides= strides)
    #sum along the third and fourth axes, forming the rebinned, flux conserved array  
    #print(blocked)

    binned=blocked.sum(axis=(2,3))
    return binned





def find_annular_profiles(HDUlist_or_filename=None,
			  ext=0, EE=False, center=None,
			  stddev=False, binsize=None,
			  maxradius=None):
    """
    Hack of popp.utils.radial_profile adding an array of annular values when STDDEV=True
    """


    """ Compute a radial profile of the image. 

    This computes a discrete radial profile evaluated on the provided binsize. For a version
    interpolated onto a continuous curve, see measure_radial().

    Code taken pretty much directly from pydatatut.pdf

    Parameters
    ----------
    HDUlist_or_filename : string
        what it sounds like.
    ext : int
        Extension in FITS file
    EE : bool
        Also return encircled energy (EE) curve in addition to radial profile?
    center : tuple of floats
        Coordinates (x,y) of PSF center, in pixel units. Default is image center. 
    binsize : float
        size of step for profile. Default is pixel size.
    stddev : bool
        Compute standard deviation in each radial bin, not average?


    Returns
    --------
    results : tuple
        Tuple containing (radius, profile) or (radius, profile, EE) depending on what is requested.
        The radius gives the center radius of each bin, while the EE is given inside the whole bin
        so you should use (radius+binsize/2) for the radius of the EE curve if you want to be
        as precise as possible.
    """
    if isinstance(HDUlist_or_filename, str):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")

    image = HDUlist[ext].data
    pixelscale = HDUlist[ext].header['PIXELSCL']


    if maxradius is not None:
        raise NotImplemented("add max radius")


    if binsize is None:
        binsize=pixelscale

    y,x = np.indices(image.shape)
    if center is None:
        # get exact center of image
        #center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple( (a-1)/2.0 for a in image.shape[::-1])

    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2) *pixelscale / binsize # radius in bin size steps
    ind = np.argsort(r.flat)


    sr = r.flat[ind]
    sim = image.flat[ind]
    ri = sr.astype(int)
    deltar = ri[1:]-ri[:-1] # assume all radii represented (more work if not)
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=float) # cumulative sum to figure out sums for each bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile=tbin/nr
    annularvals=[]
    #pre-pend the initial element that the above code misses.
    radialprofile2 = np.empty(len(radialprofile)+1)
    if rind[0] != 0:
        radialprofile2[0] =  csim[rind[0]] / (rind[0]+1)  # if there are multiple elements in the center bin, average them
    else:
        radialprofile2[0] = csim[0]                       # otherwise if there's just one then just take it. 
    radialprofile2[1:] = radialprofile
    rr = np.arange(len(radialprofile2))*binsize + binsize*0.5  # these should be centered in the bins, so add a half.

    if stddev:
        stddevs = np.zeros_like(radialprofile2)
        r_pix = r * binsize
        for i, radius in enumerate(rr):
		if i == 0: wg = np.where(r < radius+ binsize/2)
		else: 
			wg = np.where( (r_pix >= (radius-binsize/2)) &  (r_pix < (radius+binsize/2)))
			#print radius-binsize/2, radius+binsize/2, len(wg[0])
			#wg = np.where( (r >= rr[i-1]) &  (r <rr[i] )))
		
		stddevs[i] = image[wg].std()
		annularvals.append(image[wg])
        return (rr, stddevs,annularvals)

    if not EE:
        return (rr, radialprofile2)
    else:
        #weighted_profile = radialprofile2*2*np.pi*(rr/rr[1])
        #EE = np.cumsum(weighted_profile)
        EE = csim[rind]
        return (rr, radialprofile2, EE) 


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

def InputWavefrontFromField(inwave,field,arcsec_per_pixel,zero_init_wavefront=True):
    '''Treats each pixel in the field array as a source
    calculate the angle and offset of that source from
    the center of the central pixel and add a
     copy of inwave with a value =  field[x,y]/np.sum(field)
    
    For fields where every point in the field is coherent.

    if zero_init_wavefront=True then the the initial wavefront will be set to zero before
    the field array points are added.
     
    ''' 
    centerx = np.shape(field)[0]/2.0 - 0.5 #need to subtract 0.5 to be at the center of the central pixel
    centery = np.shape(field)[1]/2.0 - 0.5 
    print(centerx,centery)
  
    pixwavefront_init=inwave.copy()

    if zero_init_wavefront:
        inwave *= 0
    else:
        inwave.normalize() 
    for i in range(np.shape(field)[0]):
        for j in range(np.shape(field)[1]):
            flux=field[i,j]/np.sum(field)
            if flux > 0:
                pixwavefront=pixwavefront_init.copy()
                pixwavefront *= flux
                r_pixel=arcsec_per_pixel*np.sqrt((i-centerx)**2+(j-centerx)**2) #arcsec
                angle=np.angle(complex((j-centery),(i-centerx))) #radians.
                offset_x = r_pixel * - np.sin(angle)  # convert to offset X,Y in arcsec
                offset_y = r_pixel * np.cos(angle)  # using the usual astronomical angle convention
                pixwavefront.tilt(Xangle=offset_x, Yangle=offset_y)
                tilt_msg="Tilted wavefront by theta_X=%f, theta_Y=%f arcsec, for target with relative flux of %f" % (offset_x, offset_y,flux)
                #print(tilt_msg)
                _log.debug(tilt_msg)
                inwave +=pixwavefront
                
    print(inwave.__class__)        
    inwave.display(what='other',nrows=2,row=1, colorbar=True)
    return inwave
