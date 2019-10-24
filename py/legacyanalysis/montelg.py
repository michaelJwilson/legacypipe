import  os
import  sys
import  math
import  coord
import  logging
import  galsim
import  pylab              as      pl
import  numpy              as      np
import  matplotlib.pyplot  as      plt      
import  astropy.io.fits    as      fits
import  fitsio

from    scipy.optimize     import  minimize
from    astropy.table      import  Table


seed     = 2134
rng      = galsim.BaseDeviate(seed)

def chi2(_model, _image, _ivar):
  chi2   = (_image - _model)**2.
  chi2  *=  _model / np.sum(_model)
  chi2  *=  _ivar
  chi2   =  np.sum(chi2)

  return  chi2

def Convolve(model, _psf_fwhm, pixel_scale=0.263, nx=200, ny=200, shear=False, position_angle=0.0, psf_ell=0.0):
  if _psf_fwhm > 0.0:
    psf         = galsim.Gaussian(flux=1.0, fwhm=_psf_fwhm)

    if shear:
      ##  http://galsim-developers.github.io/GalSim/_build/html/_modules/galsim/shear.html 
      psf       = psf.shear(galsim.Shear(q=1.-psf_ell, beta=position_angle * galsim.degrees))

    impsf       = psf.drawImage(scale=pixel_scale, nx=nx, ny=ny)

    return  galsim.Convolve([model, psf]), impsf.array

  else:
    return  model, None

def gen_exponential(_gal_flux, _gal_hlight, _psf_fwhm=0.0, sky_level_pixel=0.0, pixel_scale=0.263, nx=200, ny=200, shear=False, position_angle=0.0, psf_ell=0.0):
  model             = galsim.Exponential(flux=_gal_flux, half_light_radius=_gal_hlight)
  
  # All inputs are convolved to the final profile.                                                                                                                                                                           
  cmodel, psf       = Convolve(model, _psf_fwhm, pixel_scale=pixel_scale, nx=nx, ny=ny, shear=shear, position_angle=position_angle, psf_ell=psf_ell)

  # The returned image has a member, added_flux, that gives the total flux added.                                                                                                                                            
  cmodel            = cmodel.drawImage(scale=pixel_scale, nx=nx, ny=ny)

  noise             = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
  
  if sky_level_pixel > 0.0:
    ivar            = np.ones_like(cmodel.array) / sky_level_pixel

  else:
    ivar            = np.ones_like(cmodel.array)
  
  image             = galsim.Image(cmodel, copy=True)
  image.addNoise(noise)

  return  model, psf, cmodel.array, ivar, image.array, image

def _fit_rexmodel(params, _image, _ivar, psf_fwhm):
  flux              = params[0]
  hlight            = params[1]
    
  _, _, cmodel, _, _, _ = gen_exponential(flux, hlight, psf_fwhm)

  return  chi2(cmodel, _image, _ivar)

def callback(x):
  ##  Prior on allowed parameters.                                                                                                                                                                                                   
  if (x[0] < 1.) |  (x[0] > 5.e5) | (x[1] < 0.01) | (x[1] > 50.):
    return  True

  else:
    print('{:.6e} \t {:.6e}'.format(x[0], x[1]))

def fit_rexmodel(_image, _ivar, psf_fwhm, disp=False, x0=np.array([3.5e3, 0.2])):
  ##  callback=callback
  result = minimize(_fit_rexmodel, x0, args=(_image, _ivar, psf_fwhm), method='CG', tol=1.e-8, options={'disp': disp})

  return  result.x, result.success, result.fun, result.nfev, result.nit

def is_rex(dchi2_psf, dchi2_rex):
  if (dchi2_rex - dchi2_psf) < 1.0:
    return  'PSF'

  else:
    return  'REX'

def main(argv):
    '''
    -  Use a circular exponential (REX) profile for the galaxy.
    -  Convolve it by a circular Gaussian PSF.
    -  Add Poisson noise to the image.
    '''

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger            = logging.getLogger("Legacy.")

    ##
    pixel_scale       = 0.2630    # [arcsec / pixel], DECAM.                                                                                                                                   
    nx, ny            = 200, 200  # [pixels].    

    ##  Mean sky count level per pixel in the CP-processed frames measured (with iterative rejection) for each CCD in the image section.                                                      
    decam_accds       = fitsio.FITS('/Users/MJWilson/Work/desi/legacy/catalogs/ccds-annotated-decam-dr8.fits')
    sky_levels_pixel  = decam_accds[1]['ccdskycounts'][:]

    psf_fwhm_pixels   = decam_accds[1]['fwhm'][:]
    psf_fwhm          = psf_fwhm_pixels[0] * pixel_scale                  # arcsecond.

    exptimes          = decam_accds[1]['exptime'][:]
    exptime           = exptimes[0]                                       # seconds.  

    zpts              = decam_accds[1]['zpt'][:]  
    zpt               = zpts[0]

    psf_thetas        = decam_accds[1]['psf_theta'][:]                    # PSF position angle [deg.]
    psf_theta         = psf_thetas[0]

    psf_ells          = decam_accds[1]['psf_ell'][:]                                                                                                                                                       
    psf_ell           = psf_ells[0]

    # sky_level_pixel = sky_level * pixel_scale**2
    sky_level_pixel   = sky_levels_pixel[0]                               # [counts / pixel]                                                                                                                

    nreal             = 10

    gal_mag           = 23.0                                              # [AB].

    gal_flux          = exptime * 10.**((zpt - gal_mag) / 2.5)            # [Total counts on the image].  
    gal_hlight        = 0.25                                              # [arcsec].                                                                                                                          
                           
    logger.info('\n\nStarting legacy script using:\n')
    logger.info('  - Circular Exponential galaxy (REX, flux = %.3e total counts, half-light radius = %.4e arcsecond).', gal_flux, gal_hlight)
    logger.info('  - Gaussian PSF (fwhm = %.1f arcsecond, beta %.2f, minor-to-major: %.2f).', psf_fwhm, psf_theta, 1. - psf_ell)
    logger.info('  - Pixel scale = %.2f arcsec / pixel.', pixel_scale)
    logger.info('  - Poisson noise (sky level = %.1e cnts / pixel).', sky_level_pixel)

    print('\n\nSolving for {} realizations.'.format(nreal))
    
    for rl in np.arange(0, nreal, 1):
      model, psf, cmodel, ivar, image, gimage = gen_exponential(gal_flux, gal_hlight, psf_fwhm, sky_level_pixel, pixel_scale, nx=nx, ny=ny, shear=False, position_angle=psf_theta, psf_ell=psf_ell)

      SNR              =  np.sqrt(np.sum(image ** 2. * ivar))

      print('\nSolving for {:d} (SNR {:.4f}).'.format(rl, SNR))

      nosource         =  np.ones_like(cmodel)
      
      chi2_nos         =  chi2(nosource, image, ivar)
      chi2_psf         =  chi2(psf,      image, ivar)
      chi2_rex         =  chi2(cmodel,   image, ivar)

      dchi2_psf        = -(chi2_psf - chi2_nos)
      dchi2_rex        = -(chi2_rex - chi2_nos)

      mtype            = is_rex(dchi2_psf, dchi2_rex)
      
      ##  logger.info('\n\nLegacy Chi Sq.: [%.4f, %.4f].\n\n', dchi2_psf, dchi2_rex)

      '''
      extent           = 45. * pixel_scale * np.array([-1., 1., -1., 1.])
    
      plt.imshow(image, extent=extent)
      pl.colorbar()
      pl.title(r"{:.1f}'' REX ({:.1f} AB,  {:.0f} S/N, DECAM {:.1f}'' PSF, {:.1f} sky e$^-$/pixel); $\log|\chi^2|$ = {:.2f}, {:.2f}, {:.2f}; {:s}".format(gal_hlight, gal_mag, SNR, psf_fwhm, sky_level_pixel,
                                                                                                                                                          np.log10(chi2_nos), np.log10(chi2_psf), np.log10(chi2_rex), mtype),\
                                                                                                                                                          fontsize=9)
      pl.xlabel("RA  ['']")
      pl.ylabel("DEC ['']")
      pl.show()
      '''

      if rl == 0:
        x0                       = np.array([gal_flux, gal_hlight])
        x0                      *= 1. + np.array([np.random.uniform(low=-0.15, high=0.15, size=1)[0], np.random.uniform(low=-0.15, high=0.15, size=1)[0]]) 

      x, success, fun, nfev, nit = fit_rexmodel(image, ivar, psf_fwhm, disp=False, x0=x0)

      bgal_flux, bgal_hlight     = x[0], x[1]
      
      bmag                       = zpt - 2.5 * np.log10(bgal_flux / exptime)

      _, _, cmodel, _, _, _      = gen_exponential(bgal_flux, bgal_hlight, psf_fwhm, sky_level_pixel, pixel_scale, nx=nx, ny=ny, position_angle=psf_theta, psf_ell=psf_ell)

      bchi2_rex                  = chi2(cmodel, image, ivar)
      bdchi2_rex                 = -(bchi2_rex - chi2_nos)

      bmtype                     = is_rex(dchi2_psf, bdchi2_rex)

      print('{:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {}'.format(x0[0], x0[1], x[0], x[1], bmtype))

      '''
      plt.imshow(cmodel, extent=extent)
      pl.colorbar()
      pl.title('Best fit:  {:2f}  {:.6}  {:.6}  {}'.format(bmag, bgal_flux, bgal_hlight, bmtype))
      pl.show()
    
      gimage.write(os.path.join('output', 'legacy.fits'))
      '''

    print('\n\nDone.\n\n')


if __name__ == "__main__":
    main(sys.argv)
