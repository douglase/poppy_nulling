def defined_targets():
	import pysynphot as S
	import matplotlib.pyplot as plt
	import scipy.io
	import numpy as np
	import scipy.interpolate
	#define johnson V band
	jv=S.ObsBandpass('johnson,v')
	#set the aperture in refdata to PICTURE's **effective** aperture in cm^2
	S.refs.PRIMARY_AREA=3.14*(35.0/2.0)**2

	eps_eri_pickles=S.FileSpectrum('../synphot/synphot2/pickles/dat_uvi/pickles_33.fits')
 	rigel_pickles=S.FileSpectrum("../synphot/synphot2/pickles/dat_uvi/pickles_118.fits")
#get Gamma Tau, K0III
#Parallaxes mas:	20.19 [0.32] A 2007A&A...474..653V
	gammaT_pickles=S.FileSpectrum("../synphot/synphot2/pickles/dat_uvi/pickles_78.fits")


	rigel_pickles.convert(S.units.Photlam)
	eps_eri_pickles.convert(S.units.Photlam)
	gammaT_pickles.convert(S.units.Photlam)
	#renormalize it to the V band magnitude listed by Simbad
	eps_eri_Vmag=3.73
	rigelVmag=0.12
	rigel_pickles=rigel_pickles.renorm(rigelVmag,'stmag',jv)
	eps_eri_pickles=eps_eri_pickles.renorm(eps_eri_Vmag,'stmag',jv)
	gammaT_pickles=gammaT_pickles.renorm(3.654 ,'stmag',jv)

	waves=tuple(np.arange(15)*0.01e-6+0.6e-6)

	weight=1.0/float(len(waves)) #flat

	etaSCI=0.24

	import scipy.interpolate
	rigel_counts=8.625e+08*etaSCI#
	eEri_counts=5.571e+07*etaSCI# 5571157796.96
	gammaTcounts=5.9e7
	diskflux=2e-4*(eEri_counts)*etaSCI#

	spec_eEri_func=scipy.interpolate.interp1d(eps_eri_pickles.wave*1e-10,eps_eri_pickles.flux) #nm
	spec_Rigel_funct=scipy.interpolate.interp1d(rigel_pickles.wave*1e-10,rigel_pickles.flux) #nm
	spec_gammaT_funct=scipy.interpolate.interp1d(gammaT_pickles.wave*1e-10,gammaT_pickles.flux) #nm

	spec_eEri=spec_eEri_func(waves)
	spec_Rigel=spec_Rigel_funct(waves)
	spec_gammaT=spec_gammaT_funct(waves)

	#calculate relative count weightings, this underestimates total counts somewhat
	weight_eEri=spec_eEri/max(spec_eEri)/float(len(waves))
	weight_Rigel=spec_Rigel/max(spec_Rigel)/float(len(waves))
	weight_gammaT=spec_gammaT/max(spec_gammaT)/float(len(waves))
	return {'rigel_counts':rigel_counts,'eEri_counts':eEri_counts,'gammaTcounts':gammaTcounts,'weight_eEri':weight_eEri,'weight_Rigel':weight_Rigel,'weight_gammaT':
		weight_gammaT,'waves':waves}

