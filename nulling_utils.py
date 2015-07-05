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


def add_poisson_noise(photons):
	'''takes a numpy array of  values and finds a 
	random number from a poisson distribution centered 
	on the number of photons in that bin'''
	from scipy.stats import poisson
	
	vpoisson_rvs=np.vectorize(poisson.rvs) 

	if str(type(input)) == "<class 'astropy.io.fits.hdu.hdulist.HDUList'>":
		noisy_array=vpoisson_rvs(photons[0].data)
		noisy = fits.HDUList(fits.PrimaryHDU(data=noisy_array,header=photons[0].header))
	else:
		noisy=vpoisson_rvs(photons)
	return noisy


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
    newnuller.null(flux=star_counts*weight,wavelength=wavelength)
    partial_image=newnuller.wavefront.intensity
    partial_bright=newnuller.wavefront_bright.intensity
    refpsf=partial_image
    image=newnuller.wavefront.intensity
    bright_output=newnuller.wavefront_bright.intensity
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
        print("total counts",newnuller.wavefront.totalIntensity)
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

def nulltilt(newnuller,wavelength,offset_x,offset_y,flux,returnNuller):
    '''
    null given a tilted wavefront at a particular wavelength
    '''
    newnuller.null(offset_x=offset_x,
                offset_y=offset_y,
                flux=flux,
                wavelength=wavelength)
    image=newnuller.wavefront.intensity
    bright_output=newnuller.wavefront_bright.intensity

    if returnNuller:
        return newnuller
    else:
        return image

def add_nuller_to_header(primaryHDUList,nuller):
    '''
    takes a FITS HDU and appends important nuller charactaristics.
    '''
    primaryHDUList[0].header['PIXELSCL']=nuller.wavefront.asFITS()[0].header['PIXELSCL']
    primaryHDUList[0].header.add_history("shear: "+str(nuller.shear))
    primaryHDUList[0].header.add_history("Phase Mismatch File:"+str(nuller.phase_mismatch_fits))
    primaryHDUList[0].header.add_history("Phase flattening file:"+str(nuller.phase_flat_fits))
    primaryHDUList[0].header.add_history("Intensity Mismatch:"+str(nuller.intensity_mismatch))

    primaryHDUList[0].header.add_history("Pupil Mask file:"+str(nuller.pupilmask))
    primaryHDUList[0].header.add_history("Nuller Name:"+str(nuller.name))
    if nuller.defocus:
        primaryHDUList[0].header["OPD_PV"]=str(np.abs(nuller.defocus.opd.max())+np.abs(nuller.defocus.opd.min()))
    else:
        primaryHDUList[0].header["OPD_PV"]=str(0)

	#return primaryHDU
	
def TiltfromField(field,arcsec_per_pixel):
	'''
	takes a numpy array of flux values and returns a list of x and y offsets (in arcsec) and  flux values.

    The image center is defined as half the number of pixels in the array.
    The flux from each pixel is taken as coming from the center of the pixel.
	'''
	
	center_x = np.shape(field)[0]/2.0
	center_y = np.shape(field)[1]/2.0

	npix=field.shape[0]

	points=np.array(np.where(field>0))

	tiltlist=np.zeros([3,points.shape[1]])
	
	for index in range(points.shape[1]):
		point=((points[0,index]),(points[1,index]))
		i=point[0]+0.5
		j=point[1]+0.5
		flux=field[point]
		tiltlist[2,index]=flux
		r_pixel=arcsec_per_pixel*np.sqrt((i-center_x)**2+(j-center_y)**2) #arcsec
		angle=np.angle(complex((j-center_y),(i-center_x))) #radians.
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

def halfcircularTiltList(r,totalflux,npoints,phase=0):
	''' draws a circle of radius r in cartesian coordinates.
	'''
	elementflux=totalflux/float(npoints)
	x=r*np.sin(np.arange(npoints)*6.28/npoints+phase)
	y=r*np.cos(np.arange(npoints)*6.28/npoints+phase)
	return np.array([x,y,elementflux*np.ones(npoints)])

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
			  maxradius=None,weights=None):
    """
    Hack of poppy.utils.radial_profile
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
    weights :weight array, same dimensions as input array in fits[ext]

    Returns
    --------
    results : Dict
        'rr':        The radius gives the center radius of each bin (in arcseconds'.
        'mean': the unweighted average of the counts within the annular bin.
        'EE': The EE is given inside the whole bin so you should use (radius+binsize/2)
          for the radius of the EE curve if you want to be as precise as possible.
        'annularvals': array of arrays of values at each radius.
        'stddevs': standard deviation at each radius
        'weight_avg':weighted_avg at each radius.
        'weighted std': weighted standard deviation from 'biased variance' which may underestimate the
        true variance
        (from http://stackoverflow.com/a/2415343/2142498, same as default of np.std
        (http://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html))
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

 
    stddevs = np.zeros_like(radialprofile2)
    weighted_avg = np.zeros_like(radialprofile2)
    weighted_std = np.zeros_like(radialprofile2)
    r_pix = r * binsize
    for i, radius in enumerate(rr):
        if i == 0: wg = np.where(r < radius+ binsize/2)
        else: 
            wg = np.where( (r_pix >= (radius-binsize/2)) &  (r_pix < (radius+binsize/2)))
            #print radius-binsize/2, radius+binsize/2, len(wg[0])
            #wg = np.where( (r >= rr[i-1]) &  (r <rr[i] )))
        #print(wg)
        stddevs[i] = image[wg].std()
        if weights[wg].sum()>0:
            weighted_avg[i]=np.average(image[wg],weights=weights[wg])

            weighted_std[i] = np.sqrt(np.average((image[wg]- weighted_avg[i])**2, weights=weights[wg]))
        annularvals.append(image[wg])  

    EE = csim[rind]
    return {'rr':rr,
            'mean':radialprofile2,
            'EE':EE,
            'annularvals':annularvals,
            'stddevs':stddevs,
            'weighted_avg':weighted_avg,
            'weighted_std':weighted_std}


def downsample_display(input,block=(10,10),
                        save=False,
                        filename='DownsampledOut.fits',
                        vmin=1e-8,vmax=1e1,
                        ax=False,norm='log',add_noise=False,skip_plot=False,**kwargs):
    '''
    takes an HDUList first frame and generates a downsampled fits image for display and saving to disk.
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
    downsampled=downsample(inFITS[0].data,block=block)
    titlestring=str(inFITS[0].data.shape)+" array, downsampled by:"+str(block)
    if not skip_plot:
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
        plt.title(titlestring)
        poppy.utils.imshow_with_mouseover(downsampled,ax=ax, interpolation='none',  extent=extent, norm=norm, cmap=cmap)
        plt.colorbar(ax.images[0])
    outFITS = fits.HDUList(fits.PrimaryHDU(data=downsampled,header=inFITS[0].header))
    newpixelscale=inFITS[0].header['PIXELSCL']*block[0]
    outFITS[0].header.update('PIXELSCL', newpixelscale, 'Scale in arcsec/pix (after oversampling and subsequent downsampling)')
    outFITS[0].header.add_history(titlestring)
    try:
        outFITS.writeto(filename,**kwargs)
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

def display_inset(inFITS,x1, x2, y1, y2,zoom=2.0,title="",suppressinset=False,figsize=[7,5],cmap='gist_heat',**kwargs):
    '''
    
    displays the first array of the FITS hdulist inFITS and a zoomed inset of the subregion defined by the  [x1:x2,y1:y2] 
    where x1 etc... are in display units (arc seconds) not pixels number.
    
    kwargs are passed to imshow.
    
    using example from:
    http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#insetlocator
    '''
 

    fig, ax = plt.subplots(figsize=figsize)
    
    # prepare the demo image
    Z = inFITS[0].data
    Z2 = Z
    halffov_x = inFITS[0].header['PIXELSCL']*inFITS[0].data.shape[1]/2.0
    halffov_y = inFITS[0].header['PIXELSCL']*inFITS[0].data.shape[0]/2.0
    ax.set_title(title)
    extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]
    ax.set_xlim([-halffov_x, halffov_x])
    ax.imshow(Z, extent=extent, interpolation="none",
              origin="lower",cmap=cmap,**kwargs)
    ax.set_xlabel("Arcseconds",fontsize=16)
    ax.set_ylabel("Arcseconds",fontsize=16)
    axins = zoomed_inset_axes(ax, zoom, loc=1) # zoom = 6
    if suppressinset==False:
        axins.imshow(np.array(Z2), extent=extent, interpolation="none",origin="lower",cmap=cmap,**kwargs)
        
        # sub region of the original image
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",color="white",linewidth=2.)
    for ax, color in zip([ax, axins], ['white', 'white']):
        plt.setp(ax.spines.values(), color=color)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color="white")
        
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    cax = fig.add_axes([0.88, 0.05, 0.05, 0.9])
    cax.set_title("Counts",size=12)
    cax.tick_params(labelsize=12)
    plt.colorbar(ax.images[0],cax=cax) 
    plt.setp(ax.spines.values(), color='white')
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='white')
    ax.tick_params(labelsize=16)
        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
    #fig.tight_layout()

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''
    http://wiki.scipy.org/Cookbook/Rebinning
    Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords
        # makes a view that affects newcoords
        newcoords_tr += ofs
        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas
        newcoords_tr -= ofs
        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
          "Currently only \'neighbour\', \'nearest\',\'linear\',", \
          "and \'spline\' are supported."
        return None
    


    
def simulate_noise(HDUList,t_exp,n_exp,read_noise,dark_noise_rate):
    '''
    Parameters
    ----------
    HDUList:
         astropy.io.fits.HDUList object to use as source
    t_exp:
         float [typically seconds]
    read_noise:
         float [electrons/exposure]
    dark_noise_rate
         float [dark noise per unit of t_exp, i.e. electrons/sec]

    ----------

    Return a numpy array

    inject gaussian dark noise and poisson photon noise to data from first frame of an HDUlist.
    '''
    
    detx,dety=np.shape(HDUList[0].data)
    field_read_noise=np.sqrt(n_exp)*np.random.normal(0,read_noise, detx*dety).reshape([detx,dety]) 
    DarkAndReadNoise= np.random.normal(0,np.sqrt(n_exp*read_noise**2 + dark_noise_rate*t_exp),detx*dety).reshape([detx,dety]) 
    return fits.HDUList([fits.PrimaryHDU(add_poisson_noise(HDUList[0].data*t_exp) + DarkAndReadNoise,header=HDUList[0].header)])
    
