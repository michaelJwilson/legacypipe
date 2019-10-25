import  fitsio
import  galsim
import  tractor
import  numpy                     as      np
import  pylab                     as      plt

from    legacypipe.survey         import  RexGalaxy, LogRadius
from    legacypipe.detection      import  run_sed_matched_filters, detection_maps
from    astrometry.util.util      import  Tan
from    tractor.sky               import  ConstantSky
from    astrometry.util.multiproc import  *


seed                = 2134
rng                 = galsim.BaseDeviate(seed)

def unpack_ccds(band='g', index=0, fname='/global/cscratch1/sd/mjwilson/BGS/SV-ASSIGN/ccds/ccds-annotated-decam-dr8.fits'):
  '''
  Get the index'th occurence of a 'band' ccd. 
  '''
  
  ##  Mean sky count level per pixel in the CP-processed frames measured (with iterative rejection) for each CCD in the image section.                                                                                                 
  decam_accds       = fitsio.FITS(fname)
  
  bands             = decam_accds[1]['filter'][:]
  inband            = bands == band.encode('utf-8')

  assert  np.any(inband == True)
  
  sky_levels_pixel  = decam_accds[1]['ccdskycounts'][:][inband]            # sky_level_pixel = sky_level * pixel_scale**2

  ##  Median per-pixel error standard deviation, in nanomaggies.                                                                                                                                                                      
  sky_levels_sigs   = decam_accds[1]['sig1'][:][inband].astype(float)
  
  psf_fwhm_pixels   = decam_accds[1]['fwhm'][:][inband]
  psf_fwhms         = psf_fwhm_pixels * pixscale                           # arcsecond.        

  exptimes          = decam_accds[1]['exptime'][:][inband]

  zpts              = decam_accds[1]['zpt'][:][inband]

  psf_thetas        = decam_accds[1]['psf_theta'][:][inband]               # PSF position angle [deg.]   
  psf_ells          = decam_accds[1]['psf_ell'][:][inband]

  return  sky_levels_sigs[index],  psf_fwhms[index], zpts[index], psf_thetas[index], psf_ells[index]

def gen_psf(fwhm, ell, theta, pixscale, H, W):
  psf               = galsim.Gaussian(flux=1.0, fwhm=fwhm)

  ##  http://galsim-developers.github.io/GalSim/_build/html/_modules/galsim/shear.html                                                                                                                                                 
  psf               = psf.shear(galsim.Shear(q=1.-psf_ell, beta=psf_theta * galsim.degrees))
  psf               = psf.drawImage(scale=pixscale, nx=W+1, ny=H+1)

  ##  psf           = tractor.GaussianMixturePSF(1., 0., 0., v, v, 0.)                                                                                                                                                                
  psf               = tractor.psf.PixelizedPSF(psf.array)

  return  psf


if __name__ == '__main__':
  print('Welcome to montelg src.')

  bands             = ['g', 'r', 'z']

  red               = dict(g=2.5, r=1., i=0.4, z=0.4)

  ##  Source.
  ra, dec           = 40., 10.

  gre               = 0.40                                              # [arcsec]. 
  gmag              = 23.0
  gflux             = 10. ** (-0.4 * ( gmag - 22.5 ))                   # [Nanomaggies].
  ##  gflux         = exptime * 10.**((zpt - gmag) / 2.5)               # [Total counts on the image].
    
  ##  https://github.com/dstndstn/tractor/blob/13d3239500c5af873935c81d079c928f4cdf0b1d/doc/galsim.rst                                                                                                                                 
  gflux             = tractor.NanoMaggies(**{'g': gflux, 'r': gflux / red['g'],  'z': red['z'] * gflux / red['g']})
  src               = RexGalaxy(tractor.RaDecPos(ra, dec), gflux, LogRadius(gre))

  print('Solving for {}'.format(src))
  
  ##  Image. 
  H, W              = 100, 100
  
  pixscale          = 0.262                                             # [arcsec / pixel], DECAM. 
  ps                = pixscale / 3600.

  ##  WCS.
  wcs               = Tan(ra, dec, W/2.+0.5, H/2.+0.5, -ps, 0., 0., ps, float(W), float(H))
  wcs               = tractor.ConstantFitsWcs(wcs)                      # tractor.TanWcs(wcs)
  
  tims              = []
  
  for band in bands:
    sky_level_sig,  psf_fwhm, zpt, psf_theta, psf_ell  = unpack_ccds(band=band, index=0)

    photcal                                            = tractor.MagsPhotoCal(band, zpt) 

    psf                                                = gen_psf(psf_fwhm, psf_ell, psf_theta, pixscale, H, W)
    
    sky_level_sig                                      = tractor.NanoMaggies(**{band: sky_level_sig})
    sky_level_sig                                      = photcal.brightnessToCounts(sky_level_sig)
        
    noise                                              = np.random.normal(loc=sky_level_sig, scale=np.sqrt(sky_level_sig), size=(H,W))
        
    tim                                                = tractor.Image(data=np.zeros((H,W),  np.float32),
                                                                       inverr=np.ones((H,W), np.float32),
                                                                       psf=psf,
                                                                       wcs=wcs,
                                                                       photocal=photcal)

    tr                                                 = tractor.Tractor([tim], [src])
    mod                                                = tr.getModelImage(0)

    tim.data                                           = tim.data + noise.data + mod.data
    tims.append(tim)
    
    print('Appended {} image: sky {:.4f} [nanomaggies];  psf fwhm  {:.4f};  zpt  {:.4f}'.format(band, sky_level_sig, psf_fwhm, zpt))
    
  ##
  mp = multiproc()
  
  ##  detmaps, detivs, satmaps = detection_maps(tims, wcs, bands, mp=mp, apodize=None)
  ##                                                                                                                                                                                                                       
  ##  Tnew, newcat, hot        = run_sed_matched_filters(SEDs, bands, detmaps, detivs, omit_xy=None, targetwcs, nsigma=6)
  
  ##
  cat = tractor.Catalog(src)
        
  ##
  tr  = tractor.Tractor(tims, cat)
  
  ##  Evaluate likelihood.
  lnp                = tr.getLogProb()

  for nm,val in zip(tr.getParamNames(), tr.getParams()):
    print('{} \t {}'.format(nm, val))
    
  ##  Reset the source params.
  src.brightness.setParams([.5] * 3)

  tr.freezeParam('images')

  ##
  print('Fitting:')
  tr.printThawedParams()
  tr.optimize_loop()
    
  ##
  print('Fit:', src)

  ##  Take several linearized least squares steps.
  for i in range(20):
    dlnp, X, alpha = tr.optimize()

    if dlnp < 1e-4:
      break

  ##  Generate all models output.
  
    
  ##  Plot optimized models.
  plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)

  ima     = dict(interpolation='nearest', origin='lower', cmap='gray')
  
  mods    = [tr.getModelImage(i) for i in range(len(tims))]
  plt.clf()

  nepochs = 1
    
  for i, band in enumerate(bands):
    for e in range(nepochs):
      plt.subplot(nepochs, len(bands), e*len(bands) + i +1)
      plt.imshow(mods[nepochs*i + e])
      plt.xticks([]); plt.yticks([])
      plt.title('%s #%i' % (band, e+1))
        
  plt.suptitle('Optimized models')
  plt.savefig('opt.png')

  print('\n\nDone.\n\n')
