import  os
import  fitsio
import  galsim
import  tractor
import  numpy                           as      np
import  pylab                           as      plt 

from    legacypipe.survey               import  RexGalaxy, LogRadius, wcs_for_brick
from    legacypipe.detection            import  run_sed_matched_filters, detection_maps, sed_matched_filters
from    astrometry.util.util            import  Tan
from    tractor.sky                     import  ConstantSky
from    astrometry.util.multiproc       import  *
from    legacypipe.runbrick             import  stage_srcs, stage_fitblobs
from    legacypipe.survey               import  LegacySurveyData
from    astrometry.util.fits            import  *
from    astrometry.util.util            import  *
from    astrometry.util.starutil_numpy  import degrees_between


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

  survey                          = LegacySurveyData()

  bands                           = ['g', 'r', 'z']
  
  red                             = dict(g=2.5, r=1., i=0.4, z=0.4)

  os.environ['LEGACY_SURVEY_DIR'] = '/project/projectdirs/cosmo/data/legacysurvey/dr8/'
  
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

  ##  Pixscale.
  pixscale          = 0.262                                             # [arcsec / pixel], DECAM.                                                                                                                                     
  ps                = pixscale / 3600.
    
  ##  Image. 
  H, W              = 100, 100
                                                                                                                                                                                                           
  ##  WCS.
  targetwcs         = Tan(ra, dec, W/2. + 0.5, H/2. + 0.5, -ps, 0., 0., ps, float(W), float(H))
  wcs               = tractor.ConstantFitsWcs(targetwcs)              

  ##  Bricks.
  B                 = survey.get_bricks_readonly()
  B.about()    
  B.cut(np.argsort(degrees_between(ra, dec, B.ra, B.dec)))

  brick             = B[0]
    
  targetrd          = np.array([targetwcs.pixelxy2radec(x,y) for x,y in [(1,1), (W,1), (W,H), (1,H), (1,1)]])
  
  tims              = []
  
  for band in bands:
    sky_level_sig,  psf_fwhm, zpt, psf_theta, psf_ell  = unpack_ccds(band=band, index=0)
    psf_sigma                                          = psf_fwhm / (2. * np.sqrt(2. * np.log(2.)))
    psf_sigma2                                         = psf_sigma ** 2.
    psfnorm                                            = 1./(2. * np.sqrt(np.pi) * psf_sigma)
    
    photcal                                            = tractor.MagsPhotoCal(band, zpt) 

    psf                                                = tractor.GaussianMixturePSF(1., 0., 0., psf_sigma2, psf_sigma2, 0.)
    ##  psf                                            = gen_psf(psf_fwhm, psf_ell, psf_theta, pixscale, H, W)
    
    sky_level_sig                                      = tractor.NanoMaggies(**{band: sky_level_sig})
    sky_level_sig                                      = photcal.brightnessToCounts(sky_level_sig)
        
    noise                                              = np.random.normal(loc=sky_level_sig, scale=np.sqrt(sky_level_sig), size=(H,W))
        
    tim                                                = tractor.Image(data=np.zeros((H,W),  np.float32),
                                                                       inverr=np.ones((H,W), np.float32),
                                                                       psf=psf,
                                                                       wcs=wcs,
                                                                       sky=None,
                                                                       photocal=photcal)

    tim.band                                           = band

    tim.psf_fwhm                                       = psf_fwhm

    ##  Gaussian approximation. 
    tim.psf_sigma                                      = psf_fwhm / (2. * np.sqrt(2. * np.log(2.)))
    tim.psfnorm                                        = psfnorm
    
    tim.dq                                             = None
    tim.sig1                                           = sky_level_sig
    tim.x0, tim.y0                                     = int(0), int(0)

    subh, subw                                         = tim.shape
    tim.subwcs                                         = targetwcs.get_subimage(tim.x0, tim.y0, subw, subh)
    
    tr                                                 = tractor.Tractor([tim], [src])
    mod                                                = tr.getModelImage(0)

    tim.data                                           = tim.data + noise.data + mod.data

    tims.append(tim)
    
    print('Appended {} image: sky {:.4f} [nanomaggies];  psf fwhm  {:.4f};  zpt  {:.4f}'.format(band, sky_level_sig, psf_fwhm, zpt))

  ##
  mp                           = multiproc()  

  detmaps, detivs, satmaps     = detection_maps(tims, targetwcs, bands, mp=mp, apodize=None) 
  
  ##                                                                                                                                                                                                                                  
  cat = tractor.Catalog(src)
  
  SEDs                         = sed_matched_filters(bands)
  Tnew, newcat, hot            = run_sed_matched_filters(SEDs, bands, detmaps, detivs, omit_xy=None, targetwcs=targetwcs, nsigma=6.0)  

  ##
  keys                         = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W', 'H', 'bands', 'tims', 'ps', 'brickid', 'brickname', 'brick', 'custom_brick', 'target_extent', 'ccds', 'bands', 'survey']
  src_rtn                      = stage_srcs(targetrd=targetrd,
                                            pixscale=pixscale,
                                            targetwcs=targetwcs,
                                            W=W,
                                            H=H,
                                            bands=bands,
                                            ps=None,
                                            tims=tims,
                                            plots=False,
                                            plots2=False,
                                            brickname=None,
                                            mp=mp,
                                            nsigma=6.0,
                                            survey=survey,
                                            brick=None,
                                            bailout_sources=False,
                                            tycho_stars=False,
                                            gaia_stars=False,
                                            large_galaxies=False,
                                            star_clusters=True,
                                            star_halos=False)


  for x in src_rtn.keys():
    print(src_rtn[x])

  keys                         = ['T', 'tims', 'blobsrcs', 'blobslices', 'blobs', 'cat', 'ps', 'refstars', 'gaia_stars', 'saturated_pix', 'T_donotfit', 'T_clusters']
    
  blob_keys                    = ['cat', 'invvars', 'T', 'blobs', 'brightblobmask']
  blob_rtn                     = stage_fitblobs(T=src_rtn['T'],
                                                T_clusters=None,
                                                brickname=None,
                                                brickid=None,
                                                brick=brick,
                                                version_header=None,
                                                blobsrcs=src_rtn['blobsrcs'],
                                                blobslices=src_rtn['blobslices'],
                                                blobs=src_rtn['blobs'],
                                                cat=cat,
                                                targetwcs=targetwcs,
                                                W=W,
                                                H=H,
                                                bands=bands,
                                                ps=ps,
                                                tims=tims,
                                                survey=survey,
                                                plots=False,
                                                plots2=False,
                                                nblobs=None,
                                                blob0=None,
                                                blobxy=None,
                                                blobradec=None,
                                                blobid=None,
                                                max_blobsize=None,
                                                simul_opt=False,
                                                use_ceres=True,
                                                mp=mp,
                                                checkpoint_filename=None,
                                                checkpoint_period=600,
                                                write_pickle_filename=None,
                                                write_metrics=True,
                                                get_all_models=True,
                                                refstars=None,
                                                bailout=False,
                                                record_event=None,
                                                custom_brick=False)
  
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
