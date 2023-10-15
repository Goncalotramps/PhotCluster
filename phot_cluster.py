from curses import BUTTON3_RELEASED
from http.client import BAD_GATEWAY
from math import cos
import numpy as np
import astropy
import photutils
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import wcs
import astropy.units as u
from matplotlib.pyplot import *
from scipy import interpolate
from shapely.geometry import LineString, Point
from astropy.stats import sigma_clipped_stats

import matplotlib.pyplot as plt

from astropy.visualization import astropy_mpl_style

plt.style.use(astropy_mpl_style)

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

import os
output_dir = 'photometry'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def overwrite(filename):
    if os.path.isfile(filename):
    # If it exists, remove it (overwrite)
        os.remove(filename)

# Now, save the new image
    plt.savefig(filename, dpi=500)

#######################################################################
######## INITIAL SETTINGS #############################################
#######################################################################

print("Background removal? (Y/N):") #Getting the background removal option
ans2=str(input())

if ans2=="n" or ans2=="N":
    bkg_rem = False
elif ans2=="y" or ans2=="Y":
    bkg_rem = True
else:
    bkg_rem = False
    
def match_gaia(sources,header,ra,dec,width=0.2,height=0.2): #Defining Gaia Matching function

        from astropy.coordinates import SkyCoord
        from astroquery.gaia import Gaia         
        from astropy.wcs import WCS
        from astropy import units as u
        Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"#edr3 or dr2
        Gaia.ROW_LIMIT = -1

        ## get ra,dec from WCS astrometry header
        wcs_header = WCS(header)
        coords = wcs_header.pixel_to_world(sources['xcentroid'],sources['ycentroid'])


        ## get Gaia catalog around center ra/dec values
        cencoord = SkyCoord(ra=ra,dec=dec,unit=(u.deg,u.deg),frame='icrs')
        width,height = u.Quantity(width, u.deg),u.Quantity(height, u.deg)
        gaia_stars = Gaia.query_object_async(coordinate=cencoord, width=width, height=height)
        gaia_coords = SkyCoord(ra=gaia_stars['ra'],dec=gaia_stars['dec'])

        ## match catalogs
        gidx, gd2d, gd3d = coords.match_to_catalog_sky(gaia_coords)
        gbestidx=(gd2d.deg < 0.0008)                         #<0.00015deg=0.54''

        ## output variables
        star_ra,star_dec = np.zeros(len(sources),dtype=float),np.zeros(len(sources),dtype=float)
        star_ra[:],star_dec[:] = np.nan,np.nan
        star_ra[gbestidx] = gaia_stars['ra'][gidx[gbestidx]]
        star_dec[gbestidx] = gaia_stars['dec'][gidx[gbestidx]]
        star_par,star_parer = np.zeros(len(sources),dtype=float)*np.nan,np.zeros(len(sources),dtype=float)*np.nan
        star_par[gbestidx] = gaia_stars['parallax'][gidx[gbestidx]]
        star_parer[gbestidx] = gaia_stars['parallax_error'][gidx[gbestidx]]
        star_pmra,star_pmraerr = np.zeros(len(sources),dtype=float),np.zeros(len(sources),dtype=float)
        star_pmra[:],star_pmraerr[:] = np.nan,np.nan
        star_pmra[gbestidx] = gaia_stars['pmra'][gidx[gbestidx]]
        star_pmraerr[gbestidx] = gaia_stars['pmra_error'][gidx[gbestidx]]
        star_pmdec,star_pmdecerr = np.zeros(len(sources),dtype=float),np.zeros(len(sources),dtype=float)
        star_pmdec[:],star_pmdecerr[:] = np.nan,np.nan
        star_pmdec[gbestidx] = gaia_stars['pmdec'][gidx[gbestidx]]
        star_pmdecerr[gbestidx] = gaia_stars['pmdec_error'][gidx[gbestidx]]
        return star_ra,star_dec,star_par,star_parer,star_pmra,star_pmraerr,star_pmdec,star_pmdecerr 
        

obs_order=[1,2] #we have two images so we are making the same data treatment for both in a for loop

#################################################################################
######### CALIBRATION ###########################################################
#################################################################################


#################################################################################
########### BIAS ################################################################
#################################################################################


for obs in obs_order: #creating the global for loop with the all data treatment
    i=0
    biastotal=0

    from PIL import Image
    import glob
    bias_list = []

    for filename in glob.glob(f'bias/bias_{obs}/*.fits'): #just getting the images of bias in paste
        bias_data = fits.getdata(filename, ext=0)
        bias_list.append(bias_data)                       #collecting all of them to a list

    masterbias=np.median(bias_list, axis=0) #doing the median of values in each pixel for all images

    lo,up = np.percentile(masterbias, 1),np.percentile(masterbias, 99) #setting the display limits of the image
    plt.imshow(masterbias, origin= "lower", cmap="Greys_r", clim=(lo,up))

    output_dir = 'photometry/masterbias' #creating an output to put the images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename=os.path.join(output_dir,f'masterbias_{obs}.png') #creating a filename 

    overwrite(filename)

    hdu = fits.PrimaryHDU(masterbias) #creating a .fit file with the masterbias data 

    hdul = fits.HDUList([hdu])

    
    hdul.writeto(f'masterbias_{obs}.fits', overwrite=True) # save the FITS file

    ##############################################################################
    ############### FLATS ########################################################
    ##############################################################################

    j=0

    filter=["blue","green","red"]  #list of all filters
    flatstotal=0
    for fil in filter:
        norm=["linear","log","asinh"] #possible normalizations if we put see_norm=True (old version of code)
        flats_list = []
        for filename in glob.glob(f'flats/flats_{fil}_{obs}/*.fits'): #just getting the images of flats in paste
            flats_data = fits.getdata(filename, ext=0) - masterbias #correcting the flats subtracting the masterbias
            flatsmedian= np.median(flats_data) #doing the median value of all pixels for each flat
            flats_data = (flats_data/flatsmedian) #using that median to normalize each picture
            flats_list.append(flats_data) #add to a list of normalized flats 

        masterflat=np.median(flats_list, axis=0) #doing the median of values in each pixel for all images
        
        plt.imshow(masterflat, origin= "lower", cmap="Greys_r")

        output_dir = 'photometry/masterflat' 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir,f'masterflat{fil}_{obs}.png')

        overwrite(filename)

        hdu = fits.PrimaryHDU(masterflat) #creating the .fit file for masterflat


        hdul = fits.HDUList([hdu])

        hdul.writeto(f'masterflat{fil}_{obs}.fits', overwrite=True)

    ###############################################################################
    ######### PLOTTING THE OBSERVATION IMAGE ######################################
    ###############################################################################

        import astroalign as aa
        from PIL import Image
        import glob
        data_list = []
        for filename in glob.glob(f'standard_star/hip106049_{fil}_{obs}/*.fits'): # getting the standard star files
            data = fits.getdata(filename, ext=0)
            data_list.append(data)
        

        datamedian=np.median(data_list, axis=0) #doing the median value for each pixel of each image

        image_data=(datamedian-masterbias)/masterflat #normalizing the data with masterbias and masterflats

        plt.imshow(image_data, origin= "lower", cmap="Greys_r")

        output_dir = 'photometry/standard_star/standard_image'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir,f'standard_{fil}_{obs}.png')

        overwrite(filename)

        hdu = fits.PrimaryHDU(image_data) #getting the normalized data pixels values

        
        hdul = fits.HDUList([hdu])

        
        hdul.writeto(f'justimage_standard{fil}_{obs}.fits', overwrite=True)

        plt.clf()

        ########################################################################
        ########## SEEING THE BACKGROUND #######################################
        ########################################################################

        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground

        sigma_clip = SigmaClip(sigma=3) #here we are setting a function to put a limit where all above that is considered background
        bkg_estimator = MedianBackground()
        bkg = Background2D(image_data, (50, 50), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator) #here we get the background info using the limits that we stabilished

        plt.imshow(bkg.background, origin='lower', cmap='Greys_r',
                interpolation='nearest')
        
        output_dir = 'photometry/standard_star/standard_bkg'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir, f'bkgmap_standard_{fil}_{obs}.png')

        overwrite(filename)

        plt.clf()

        hdu = fits.PrimaryHDU(bkg.background)


        hdul = fits.HDUList([hdu])

        # Save the FITS file
        hdul.writeto(f'background_standard{fil}_{obs}.fits', overwrite=True)

        if bkg_rem == True: 

            image_data = image_data - bkg.background #we remove the background from the image data

            plt.imshow(image_data, origin='lower',
                    cmap='Greys_r', interpolation='nearest')
            
            output_dir = 'photometry/standard_star/standard_bkgsub'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            filename=os.path.join(output_dir, f'bkgsub_data_standard{fil}_{obs}.png')

            overwrite(filename)

            plt.clf()

        #########################################################################
        ########## FINDING OBJECTS ##############################################
        #########################################################################

        from astropy.stats import sigma_clipped_stats

        import math

        mean,median,std= sigma_clipped_stats(image_data, sigma=3.0) #here we are getting some statistical information from the data with bkg removed like the mean value, median and 
                                                                    #std of pixel's value to use in DaoFinder that will allow us to detect possible stars

        from photutils.detection import DAOStarFinder

        daofind= DAOStarFinder(fwhm= 3, threshold = 5.0*std, exclude_border=True) #here we define the DAOFinder function 

        sources= daofind(image_data-median) #here we use it to detect the objects above the median of pixel values pf the image 

        i=0

        import numpy as np 
        import matplotlib.pyplot as plt
        from photutils.aperture import CircularAperture
        
        nbright = 1 #here we are seeking for the brightest star (standard star in our image)
        brightest = np.argsort(sources['flux'])[::-1][0:nbright] #so we order all the detected objects from the ones with bigger flux to the ones with less flux and choose the first one (brighter one)
        brsources = sources[brightest] #thats our brightest star (standard star)

        brpositions=np.transpose((brsources["xcentroid"],brsources["ycentroid"])) #just checking if the detection is correct plotting it (it should detect the standard star)

        pos_apertures= CircularAperture(brpositions, r=5)

        plt.imshow(image_data,origin= "lower", norm="asinh", cmap="Greys_r", interpolation= "nearest")

        pos_apertures.plot(color ="red", lw=1.5, alpha = 0.5)

        output_dir = 'photometry/standard_star/standard_indentify'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir, f'identified_brobjects_standard_{fil}_{obs}.png')

        overwrite(filename)

        plt.clf()

        #########################################################################
        ############ PREPARING PSF PHOTOMETRY ###################################
        #########################################################################

        from astropy.stats import sigma_clipped_stats,gaussian_sigma_to_fwhm
        
        #in the next function we are getting some statistical informations to use in psf model using the information obtained with DAOFINDER
        #with that we can find a good fwhm using the brightest stars , starting with good values for stars radius 

        rmax = 25
        (ny,nx) = np.shape(image_data)
        from astropy.modeling import models,fitting
        fit_g = fitting.LevMarLSQFitter()
        allxfwhm, allyfwhm = np.zeros(len(brsources)),np.zeros(len(brsources))
        allfwhm,alltheta = np.zeros(len(brsources)),np.zeros(len(brsources))
        for i,src in enumerate(brsources):
            if int(src['ycentroid']) > rmax and int(src['ycentroid']) < ny-rmax and int(src['xcentroid']) > rmax and int(src['xcentroid']) < nx-rmax:
                img = image_data[int(src['ycentroid'])-rmax:int(src['ycentroid'])+rmax,
                            int(src['xcentroid'])-rmax:int(src['xcentroid'])+rmax]
                subx,suby = np.indices(img.shape) # instead of meshgrid
                p_init = models.Gaussian2D(amplitude=np.max(img),x_mean=rmax,y_mean=rmax,x_stddev=1.0,y_stddev=1.0)
                fitgauss = fit_g(p_init, subx, suby, img - np.min(img))
                allxfwhm[i] = np.abs(fitgauss.x_stddev.value)
                allyfwhm[i] = np.abs(fitgauss.y_stddev.value)
                allfwhm[i] = 0.5*(allxfwhm[i]+allyfwhm[i])
                alltheta[i] = fitgauss.theta.value

        xfwhm,yfwhm = np.median(allxfwhm)*gaussian_sigma_to_fwhm,np.median(allyfwhm)*gaussian_sigma_to_fwhm
        fwhm = np.median(allfwhm)*gaussian_sigma_to_fwhm
        sigfwhm, sigxfwhm, sigyfwhm = np.std(allfwhm), np.std(allxfwhm), np.std(allyfwhm)
        medtheta = np.median(alltheta)

        ###############################################################################
        ############# PSF MODEL PHOTOMETRY ############################################
        ###############################################################################

        from photutils.psf import IntegratedGaussianPRF
        from photutils.psf import PSFPhotometry
        from astropy.visualization import simple_norm
        from photutils.psf import prepare_psf_model,extract_stars,EPSFBuilder,IntegratedGaussianPRF

        psf = IntegratedGaussianPRF(sigma=fwhm/gaussian_sigma_to_fwhm)  ## 1D Gauss and thats the model that we are going to use to do the PSFphotometry
        two2dgauss = models.Gaussian2D(x_mean=0.0,y_mean=0.0,theta=medtheta,x_stddev=xfwhm/gaussian_sigma_to_fwhm,y_stddev=yfwhm/gaussian_sigma_to_fwhm)
        two2dpsf = prepare_psf_model(two2dgauss,xname='x_mean',yname='y_mean',fluxname='amplitude')

        fit_shape=(5,5)

        positions=[sources["xcentroid"],sources["ycentroid"]] 
        
        psfphot =  PSFPhotometry(psf, fit_shape, finder=daofind, #it is using the model that we have created above and uses daofinder to find the stars 
                                aperture_radius=0.5) 

        phot = psfphot(image_data,error=bkg.background) #do the photometry of our image_data and creates a model without the residuals 
                                                        #from where we get the flux and the residuals

        resid = psfphot.make_residual_image(image_data, (25, 25)) #residuals that we got from the model 

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) #plotting the images of the actual data, PSFPhotometry model and residuals 
        norm = simple_norm(image_data, 'asinh', percent=99)
        ax[0].imshow(image_data, origin='lower', norm=norm)
        ax[1].imshow(image_data - resid, origin='lower', norm=norm)
        im = ax[2].imshow(resid, origin='lower')
        ax[0].set_title('Data')
        ax[1].set_title('Model')
        ax[2].set_title('Residual Image')
        plt.tight_layout()

        output_dir = 'photometry/standard_star/standard_pdf_phot'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir,f'pdfphotometry_{fil}_{obs}.png')

        overwrite(filename)

        plt.clf()

        #########################################################################
        ############## MAGNITUDE CALCULATION AND CALIBRATION ####################
        #########################################################################

        standard=[]
        maxpeak=max(psfphot.finder_results[0]["peak"]) #here we are just getting the standard star to use it to calibrate the ZP and get the calibrated magnitude 

        if obs==1: #we are setting the exposure time used in each observation and in each filter and the angle theta (azimutal angle of the object)
            exp_blue=10
            exp_green=3
            exp_red=1
            theta=48.2
        if obs==2:
            exp_blue=8
            exp_green=6
            exp_red=5
            theta=56.8

        i=0

        for item in phot: #we get the standard star flux (brightest one) and calibrate it 
            if item["flux_fit"]==np.max(phot["flux_fit"]):
                standard.append((item["x_fit"],item["y_fit"]))
                standard_flux=item["flux_fit"]

                if fil=="blue": #we need to check the theorical values for Std Star in each filter
                    if obs==1:
                        standard_mag_t=7.52 
                        mag_er_t=0.02
                        print("m for HD180450 in B filter:")
                    elif obs==2:
                        standard_mag_t=7.062
                        mag_er_t=0.015
                        print("m for Hip106049 in B filter:")
                    m_inst=-2.5*math.log10(standard_flux/exp_blue)+0.25*(1/math.cos((math.pi*theta)/180)) #doing the corrections
                    zp=m_inst-standard_mag_t
                    standard_mag=m_inst-zp
                    psfphot.finder_results[0]["mag"][i]=standard_mag
                    flux_err=phot["flux_err"][i]
                    mag_err1=np.sqrt((2.5/(phot["flux_fit"][i])*np.log(10))*(2.5/(phot["flux_fit"][i])*np.log(10))*flux_err*flux_err)
                    zp_err=np.sqrt(mag_err1*mag_err1+mag_er_t*mag_er_t)
                    mag_err=np.sqrt(mag_err1*mag_err1+zp_err*zp_err)


                elif fil=="green":
                    if obs==1:
                        print("m for HD180450 in G filter:")
                        standard_mag_t=5.85 
                        mag_er_t=0.003089
                    elif obs==2:
                        standard_mag_t=6.719122
                        mag_er_t=0.002773
                        print("m for Hip106049 in G filter:")
                    m_inst=-2.5*math.log10(standard_flux/exp_green)+0.15*(1/math.cos((math.pi*theta)/180))
                    zp=m_inst-standard_mag_t
                    standard_mag=m_inst-zp
                    psfphot.finder_results[0]["mag"][i]=standard_mag
                    flux_err=item["flux_err"]
                    mag_err1=np.sqrt((2.5/(item["flux_fit"])*np.log(10))*(2.5/(item["flux_fit"])*np.log(10))*flux_err*flux_err)
                    zp_err=np.sqrt(mag_err1*mag_err1+mag_er_t*mag_er_t)
                    mag_err=np.sqrt(mag_err1*mag_err1+zp_err*zp_err)

                elif fil=="red":
                    if obs==1:
                        standard_mag_t=2.57 
                        mag_er_t=0.262
                        print("m for HD180450 in R filter:")
                    elif obs==2:
                        standard_mag_t=6.163
                        mag_er_t=0.017
                        print("m for Hip106049 in R filter:") 
                    m_inst=-2.5*math.log10(standard_flux/exp_red)+0.09*(1/math.cos((math.pi*theta)/180))
                    zp=m_inst-standard_mag_t
                    standard_mag=m_inst-zp
                    psfphot.finder_results[0]["mag"][i]=standard_mag
                    flux_err=phot["flux_err"][i]
                    mag_err1=np.sqrt((2.5/(phot["flux_fit"][i])*np.log(10))*(2.5/(phot["flux_fit"][i])*np.log(10))*flux_err*flux_err)
                    zp_err=np.sqrt(mag_err1*mag_err1+mag_er_t*mag_er_t)
                    mag_err=np.sqrt(mag_err1*mag_err1+zp_err*zp_err)

                print (standard_mag)
                print("Error:")
                print (mag_err)

            i=i+1

        verify_standard= CircularAperture(standard, r=5) #check if we got the star right

        plt.imshow(image_data,origin= "lower", norm="asinh", cmap="Greys_r", interpolation= "nearest")

        verify_standard.plot(color ="blue", lw=1.5, alpha = 0.5)

        output_dir = 'photometry/standard_star/standard_identify'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir, f'standart_identify_standard_{fil}_{obs}.png')

        overwrite(filename)

        plt.clf()

        

    ##############################################################################################################
    ######################################### CLUSTER TREATMENT ##################################################
    ##############################################################################################################

        from PIL import Image
        import glob
        from astroalign import *

        data_list = []
        for filename in glob.glob(f'cluster/{fil}_{obs}/*.fits'): #the process is really similar to the standard star 
            data = fits.getdata(filename, ext=0)
            data_list.append(data)

        reference_image = data_list[0]

        aligned_data_list = [reference_image] #Here we align the images using astroalign (we didnt do that for standard images because they were really align and my pc memory was almost dying)
        for data in data_list[1:]:
            aligned_image, _ = aa.register(data, reference_image)
            aligned_data_list.append(aligned_image)
        

        datamedian=np.median(aligned_data_list, axis=0)

        norm=["linear","log","asinh"] #normalization that were possible in old code

        image_data=(datamedian-masterbias)/masterflat #doing the correction of the cluster image

        plt.imshow(image_data, origin= "lower", cmap="Greys_r")

        output_dir = 'photometry/cluster/cluster_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir,f'cluster_{fil}_{obs}.png')

        overwrite(filename)

        hdu = fits.PrimaryHDU(image_data)


        hdul = fits.HDUList([hdu])

        hdul.writeto(f'cluster_{fil}_{obs}.fits', overwrite=True)

        plt.clf()

        ########################################################################
        ########## SEEING THE BACKGROUND #######################################
        ########################################################################

        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground
        sigma_clip = SigmaClip(sigma=3)                                 # same for background as the standaerd star
        bkg_estimator = MedianBackground()
        bkg = Background2D(image_data, (50, 50), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        plt.imshow(bkg.background, origin='lower', cmap='Greys_r',
                interpolation='nearest')

        output_dir = 'photometry/cluster/cluster_bkgmap'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir, f'bkgmap_{fil}_{obs}.png')

        overwrite(filename)

        plt.clf()

        hdu = fits.PrimaryHDU(bkg.background)

        hdul = fits.HDUList([hdu])

        hdul.writeto(f'background_{fil}_{obs}.fits', overwrite=True)

        if bkg_rem == True:

            image_data = image_data - bkg.background

            plt.imshow(image_data, origin='lower',
                    cmap='Greys_r', interpolation='nearest')
            
            output_dir = 'photometry/cluster/cluster_bkgsub'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            filename=os.path.join(output_dir, f'bkgsub_cluster_{fil}_{obs}.png')

            overwrite(filename)

        plt.clf()

        #########################################################################
        ########## FINDING OBJECTS AND GAIA MATCHING ############################
        #########################################################################

        from astropy.stats import sigma_clipped_stats

        import math

        mean,median,std= sigma_clipped_stats(image_data, sigma=3.0) #really similar to standard star treatment

        from photutils.detection import DAOStarFinder

        daofind= DAOStarFinder(fwhm= 3, threshold = 5.0*std, exclude_border=True)

        sources= daofind(image_data-median)

        if fil=="blue": 
            hdul = fits.open(f"wcsblue_{obs}.fits")  #we got the images WCS using astrometry.net for each observation and each filter 
            header=hdul[0].header                    #so here we are getting the wcs accessing the .fits headers to make the matching with GAIA catalog
            if obs==2:
                ra=301.559   #image center coordinates 
                dec=35.791
                

            elif obs==1:
                ra=301.435
                dec=35.783
                ra_blue=ra

                
        elif fil=="green":
            hdul = fits.open(f"wcsgreen_{obs}.fits")
            header=hdul[0].header
            if obs==2:
                ra=301.561
                dec=35.791

            elif obs==1:
                ra=301.435
                dec=35.783
                ra_green=ra


        elif fil=="red":
            hdul = fits.open(f"wcsred_{obs}.fits")
            header=hdul[0].header
            if obs==2:
                ra=301.562
                dec=35.791

            elif obs==1:
                ra=301.438
                dec=35.783
                ra_red=ra


        if obs==1:
            gaia_sources1=match_gaia(sources,header,ra=ra,dec=dec) #we do the Gaia matching (we get the Stars Coordinates from the Gaia Catalog for our sky area and match them with out stars coordinades obtained in DAOFinder)

            index_interest1=[]

            while i<len(gaia_sources1[0]): #here we select the interessing objects (the ones that should be from our cluster) using the proper motion and parallex information obtained in the Gaia Matching
                if  0.4<gaia_sources1[2][i] and gaia_sources1[2][i]< 0.6 and -3.35 <gaia_sources1[4][i] and gaia_sources1[4][i]< -2.8: #constraining limits to the objects PM and Parallex with theorical values in center
                    index_interest1.append(i)
                i=i+1
            
            if fil=="blue":
                interest_b_ra=[]
                interest_b_dec=[]

                ramax_b=np.max(gaia_sources1[0][~np.isnan(gaia_sources1[0])])

                for index in index_interest1:
                    interest_b_ra.append(gaia_sources1[0][index])
                    interest_b_dec.append(gaia_sources1[1][index])
            elif fil=="green":
                interest_g_ra=[]
                interest_g_dec=[]
                ramax_g=np.max(gaia_sources1[0][~np.isnan(gaia_sources1[0])])
                for index in index_interest1:
                    interest_g_ra.append(gaia_sources1[0][index])
                    interest_g_dec.append(gaia_sources1[1][index])
            elif fil=="red":
                interest_r_ra=[]
                interest_r_dec=[]
                ramax_r=np.max(gaia_sources1[0][~np.isnan(gaia_sources1[0])])
                for index in index_interest1:
                    interest_r_ra.append(gaia_sources1[0][index])
                    interest_r_dec.append(gaia_sources1[1][index])
            
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            # x-axis values
            
            # plotting points as a scatter plot
            plt.scatter(gaia_sources1[0],gaia_sources1[1] ,color= "blue", #we plot the stars that matched the gaia catalog for each filter and observation
                        marker= "o", s=15)
            
            for index in index_interest1:
                plt.scatter(gaia_sources1[0][index],gaia_sources1[1][index] ,color= "cyan", #here we just plot the ones that should be from the cluster
                            marker= "o", s=15)
            
            # x-axis label
            plt.xlabel('RA (Degrees)')
            # frequency label
            plt.ylabel('Dec (Deg)')
            # plot title

            plt.title('Coords Gaia Match')
            # showing legend
            plt.legend()

            output_dir = 'photometry/cluster/gaia'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename=os.path.join(output_dir, f'matching_gaia_RA_DEC_{fil}_{obs}.png')

            overwrite(filename)

            plt.clf()  

            
        elif obs==2:
            i=0
            gaia_sources2=match_gaia(sources,header,ra=ra,dec=dec) #same for second image

            index_interest2=[]

            if fil=="blue":
                ra1=[]
                dec1=[]
                ra1=interest_b_ra
                dec1=interest_b_dec
                racent=ramax_b

            elif fil=="green":
                racent=ramax_g
                ra1=[]
                dec1=[]
                ra1=interest_g_ra
                dec1=interest_g_dec
            elif fil=="red":
                racent=ramax_r
                ra1=[]
                dec1=[]
                ra1=interest_r_ra
                dec1=interest_r_dec

            while i<len(gaia_sources2[0]):
                if 0.4<gaia_sources2[2][i] and gaia_sources2[2][i]< 0.6 and -3.35<gaia_sources2[4][i] and gaia_sources2[4][i]< -2.8: #gaia_sources2[0][i]>racent and
                    index_interest2.append(i)
                i=i+1
            
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            
            plt.scatter(gaia_sources2[0],gaia_sources2[1] ,color= "blue", 
                        marker= "o", s=15)
            
            for index in index_interest2:
                plt.scatter(gaia_sources2[0][index],gaia_sources2[1][index] ,color= "cyan", 
                            marker= "o", s=15)
            
        
            plt.xlabel('RA (Degrees)')
      
            plt.ylabel('Dec (Deg)')
          
            plt.title('Coords Gaia Match')
        
            plt.legend()

            output_dir = 'photometry/cluster/gaia'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename=os.path.join(output_dir, f'matching_gaia_RA_DEC_{fil}_{obs}.png')

            overwrite(filename)

            plt.clf()  


            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            index=[]

            
            for index in index_interest2:
                plt.scatter(gaia_sources2[0][index],gaia_sources2[1][index] , color= "black", #here we are going to do the plot of ALL stars that should be from our cluster
                            marker= "o", s=15)
                
            for index in index_interest1:
                plt.scatter(ra1,dec1,color= "black",
                            marker= "o", s=15)
            
     
            plt.xlabel('RA (Degrees)')
           
            plt.ylabel('Dec (Deg)')

            plt.title('Coords Gaia Match')
            
            plt.legend()

            output_dir = 'photometry/cluster/gaia'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename=os.path.join(output_dir, f'matching_gaia_RA_DEC_obj_comp_{fil}.png')

            overwrite(filename)

            plt.clf()  

            if fil=="red": #here we do the color map of the parallex in order of the proper motion in each coordinate to all stars (not only the interesting ones) 
                                                                    #just to check the proximity and relation of the parameters (we should see stars with close parallax in the same region of the PM space )
                pmraerr1=[gaia_sources1[5],gaia_sources1[5]]
                pmdecerr1=[gaia_sources1[7],gaia_sources1[7]]

                pmraerr2=[gaia_sources2[5],gaia_sources2[5]]
                pmdecerr2=[gaia_sources2[7],gaia_sources2[7]]

                plt.errorbar(gaia_sources1[4],gaia_sources1[6], xerr=pmraerr1, capsize=2,markersize=2,fmt='o',color='green',ecolor='lightgrey',zorder=1)
                plt.errorbar(gaia_sources1[4],gaia_sources1[6], yerr=pmdecerr1, capsize=2,markersize=2, fmt='o',color='green',ecolor='lightgrey',zorder=1)

                plt.errorbar(gaia_sources2[4],gaia_sources2[6], xerr=pmraerr2, capsize=2,markersize=2,fmt='o',color='green',ecolor='lightgrey',zorder=1)
                plt.errorbar(gaia_sources2[4],gaia_sources2[6], yerr=pmdecerr2, capsize=2,markersize=2, fmt='o',color='green',ecolor='lightgrey',zorder=1)

                cm = matplotlib.cm.get_cmap('viridis')

                sc = plt.scatter(gaia_sources1[4],gaia_sources1[6], c=gaia_sources1[2], vmin=0, vmax=1, s=6, cmap=cm,zorder=2)
                sc = plt.scatter(gaia_sources2[4],gaia_sources2[6], c=gaia_sources2[2], vmin=0, vmax=1, s=6, cmap=cm,zorder=2)
                plt.colorbar(sc)


                plt.subplots_adjust(left=0.05, top=0.9, right=1, bottom=0.1)
                # x-axis label
                plt.xlabel('Proper Motion RA (arcsec/yr)')
                # frequency label
                plt.ylabel('Proper Motion Dec (arcsec/yr)')
                # plot title

                plt.title('Proper Motion and Parallax Gaia Matching')
                # showing legend
                plt.legend()

                filename=os.path.join(output_dir, f'matching_gaia_PM_comp.png')

                overwrite(filename)

                plt.clf()  

                
                for index in index_interest1: #then we see that relation only for the stars that we detected as the cluster ones
                    pmraerr1=gaia_sources1[5][index]
                    pmdecerr1=gaia_sources1[7][index]
                    plt.errorbar(gaia_sources1[4][index],gaia_sources1[6][index], xerr=pmraerr1, capsize=2,markersize=5,fmt='o',color='blue',ecolor='lightgrey',zorder=1)
                    plt.errorbar(gaia_sources1[4][index],gaia_sources1[6][index], yerr=pmdecerr1, capsize=2,markersize=5, fmt='o',color='blue',ecolor='lightgrey',zorder=1)
                for index in index_interest2:
                    pmraerr2=gaia_sources2[5][index]
                    pmdecerr2=gaia_sources2[7][index]

                    plt.errorbar(gaia_sources2[4][index],gaia_sources2[6][index], xerr=pmraerr2, capsize=2,markersize=5,fmt='o',color='blue',ecolor='lightgrey',zorder=1)
                    plt.errorbar(gaia_sources2[4][index],gaia_sources2[6][index], yerr=pmdecerr2, capsize=2,markersize=5, fmt='o',color='blue',ecolor='lightgrey',zorder=1)

                plt.xlabel('Proper Motion RA (arcsec/yr)')
                # frequency label
                plt.ylabel('Proper Motion Dec (arcsec/yr)')
                # plot title
                plt.subplots_adjust(left=0.05, top=0.9, right=1, bottom=0.1)

                plt.title('Proper Motion and Parallax Gaia Matching (Used)')
                # showing legend
                plt.legend()

                filename=os.path.join(output_dir, f'matching_gaia_PM_used.png')

                overwrite(filename)

                plt.clf()  

        #########################################################################
        ############ PREPARING PSF PHOTOMETRY ###################################
        #########################################################################
            
        i=0
        
        index=[] #HERE is a difference. sometimes the daofind detects 2 stars in brightest spots where theres just one star. 
        #I made a function that just checks the distance between those 2 close "stars" and if they are two close (distance<13) it will clear both of them and set the
        # new position as the mean point of them (that would be really cool if we could use those positions in psf) 

        while i<len(sources["id"]):
            x1=sources["xcentroid"][i]
            y1=sources["ycentroid"][i]
            z=i+1
            while z<len(sources["id"]):
                x2=sources["xcentroid"][z]
                y2=sources["ycentroid"][z]
                distance= math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
                if distance<12:
                    sources["xcentroid"][i]=(x1+x2)/2  
                    sources["ycentroid"][i]=(y1+y2)/2
                    index.append(z)
                z=z+1
            i=i+1

        #This is only used to create the brightest stars list so it shoulden't really afect the PSF photometry

        mask = np.logical_not(np.isin(np.arange(len(sources)), index)) #cleaning it up and making a new sources list with the mean points
        sources=sources[mask]

        nbright = 3
        brightest = np.argsort(sources["flux"])[::-1][0:nbright] #just checking the 5 brightest stars to use them to found the best fwhm to use in psf function
        brsources = sources[brightest]

        brpositions=np.transpose((brsources["xcentroid"],brsources["ycentroid"]))
        pos_apertures= CircularAperture(brpositions, r=5)

        plt.imshow(image_data,origin= "lower", norm="asinh", cmap="Greys_r", interpolation= "nearest") #plotting the brightest stars

        pos_apertures.plot(color ="red", lw=1.5, alpha = 0.5)

        output_dir = 'photometry/cluster/cluster_br_identify'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir, f'identified_brobj_cluster_{fil}_{obs}.png')

        overwrite(filename)

        plt.clf()

        from astropy.stats import sigma_clipped_stats,gaussian_sigma_to_fwhm

        rmax = 25
        (ny,nx) = np.shape(image_data)
        from astropy.modeling import models,fitting
        fit_g = fitting.LevMarLSQFitter()
        allxfwhm, allyfwhm = np.zeros(len(brsources)),np.zeros(len(brsources))
        allfwhm,alltheta = np.zeros(len(brsources)),np.zeros(len(brsources))
        for i,src in enumerate(brsources):
            if int(src['ycentroid']) > rmax and int(src['ycentroid']) < ny-rmax and int(src['xcentroid']) > rmax and int(src['xcentroid']) < nx-rmax:
                img = image_data[int(src['ycentroid'])-rmax:int(src['ycentroid'])+rmax,
                            int(src['xcentroid'])-rmax:int(src['xcentroid'])+rmax]
                subx,suby = np.indices(img.shape) # instead of meshgrid
                p_init = models.Gaussian2D(amplitude=np.max(img),x_mean=rmax,y_mean=rmax,x_stddev=1.0,y_stddev=1.0)
                fitgauss = fit_g(p_init, subx, suby, img - np.min(img))
                allxfwhm[i] = np.abs(fitgauss.x_stddev.value)
                allyfwhm[i] = np.abs(fitgauss.y_stddev.value)
                allfwhm[i] = 0.5*(allxfwhm[i]+allyfwhm[i])
                alltheta[i] = fitgauss.theta.value

        xfwhm,yfwhm = np.median(allxfwhm)*gaussian_sigma_to_fwhm,np.median(allyfwhm)*gaussian_sigma_to_fwhm
        fwhm = np.median(allfwhm)*gaussian_sigma_to_fwhm
        sigfwhm, sigxfwhm, sigyfwhm = np.std(allfwhm), np.std(allxfwhm), np.std(allyfwhm)
        medtheta = np.median(alltheta)
        print("     x-FWHM %f +/- %f (pix) for this image " %(xfwhm,sigxfwhm))
        print("     y-FWHM %f +/- %f (pix) for this image " %(yfwhm,sigyfwhm))
        print("     FWHM %f +/- %f (pix) for this image " %(fwhm,sigfwhm))

        ###################################################################################
        ############# PSFPHOTOMETRY FOR CLUSTER ############################################
        ###################################################################################

        from photutils.psf import IntegratedGaussianPRF
        from photutils.psf import PSFPhotometry
        from astropy.visualization import simple_norm
        from photutils.psf import prepare_psf_model,extract_stars,EPSFBuilder,IntegratedGaussianPRF

        psf = IntegratedGaussianPRF(sigma=fwhm/gaussian_sigma_to_fwhm)  

        fit_shape=(5,5)

        psfphot =  PSFPhotometry(psf, fit_shape, finder=daofind,  #very much the same as standard stars
                                aperture_radius=0.5) 

        phot = psfphot(image_data,error=bkg.background)

        resid = psfphot.make_residual_image(image_data, (25, 25))

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        norm = simple_norm(image_data, 'asinh', percent=99)
        ax[0].imshow(image_data, origin='lower', norm=norm)
        ax[1].imshow(image_data - resid, origin='lower', norm=norm)
        im = ax[2].imshow(resid, origin='lower')
        ax[0].set_title('Data')
        ax[1].set_title('Model')
        ax[2].set_title('Residual Image')
        plt.tight_layout()

        output_dir = 'photometry/cluster/cluster_psf_phot'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename=os.path.join(output_dir, f'psfphotometry_{fil}_{obs}.png')

        overwrite(filename)

        plt.clf()

        ########################################################################
        #### MAGNITUDE CORRECTION AND CALCULATION ##############################
        ########################################################################

        i=0

        if obs==1: #exposure times used for different observation and filters, and angles of the cluster in different days
            exp_blue=10
            exp_green=6
            exp_red=3
            theta=60
        if obs==2:
            exp_blue=8
            exp_green=6
            exp_red=5
            theta=42
        
        mag_c_err=[]

        while i<len(psfphot.finder_results[0]["id"]): #here we are correcting the magnitudes using the zp, airmass and exptime (the psfphot has the number of elements that PSFPhot model has)
            if fil=="blue":
                if phot["flux_fit"][i]>0: #here we calculate the magnitude with corrections and their errors for each filter

                    psfphot.finder_results[0]["mag"][i]=-2.5*math.log10(phot["flux_fit"][i]/exp_blue)+0.25*(1/math.cos((math.pi*theta)/180))-zp #Using psfphot.finder_results just because it has already a list for magnitudes and is more pratical but all the calculations are made with PSFPhot model data
                    flux_err=phot["flux_err"][i]
                    mag_c_err.append(np.sqrt((2.5/((phot["flux_fit"][i])*np.log(10)))*(2.5/((phot["flux_fit"][i])*np.log(10)))*flux_err*flux_err+zp_err*zp_err+(0.25*math.pi*math.tan((math.pi*theta)/180)/(180*math.cos(math.pi*theta)/180))*(0.25*math.pi*math.tan((math.pi*theta)/180)/(180*math.cos(math.pi*theta)/180))))

                elif phot["flux_fit"][i]<0:
                    mag_c_err.append(0)
            elif fil=="green":

                if phot["flux_fit"][i]>0:

                    psfphot.finder_results[0]["mag"][i]=-2.5*math.log10(phot["flux_fit"][i]/exp_green)+0.15*(1/math.cos((math.pi*theta)/180))-zp  
                    flux_err=phot["flux_err"][i]
                    mag_c_err.append(np.sqrt((2.5/((phot["flux_fit"][i])*np.log(10)))*(2.5/((phot["flux_fit"][i])*np.log(10)))*flux_err*flux_err+zp_err*zp_err+(0.15*math.pi*math.tan((math.pi*theta)/180)/(180*math.cos(math.pi*theta)/180))*(0.15*math.pi*math.tan((math.pi*theta)/180)/(180*math.cos(math.pi*theta)/180))))

                elif phot["flux_fit"][i]<0:
                    mag_c_err.append(0)

            elif fil=="red":
                if phot["flux_fit"][i]>0:
                    psfphot.finder_results[0]["mag"][i]=-2.5*math.log10(phot["flux_fit"][i]/exp_red)+0.09*(1/math.cos((math.pi*theta)/180))-zp     
                    flux_err=phot["flux_err"][i]
                    mag_c_err.append(np.sqrt((2.5/((phot["flux_fit"][i])*np.log(10)))*(2.5/((phot["flux_fit"][i])*np.log(10)))*flux_err*flux_err+zp_err*zp_err+(0.09*math.pi*math.tan((math.pi*theta)/180)/(180*math.cos(math.pi*theta)/180))*(0.09*math.pi*math.tan((math.pi*theta)/180)/(180*math.cos(math.pi*theta)/180))))

                elif phot["flux_fit"][i]<0:
                    mag_c_err.append(0)

            i=i+1

        if obs==1:     

            if fil=="blue": #here we get all the important parameters of the cluster for each filter (magnitudes, positions,fluxes, ra, dec and brightest sources)
                mag_cluster_blue=[]
                posx_blue=[]
                posy_blue=[]
                fluxb=[]
                mag_err_b=[]
                rainteressb=[]
                decinteressb=[]

                for index in index_interest1: 
                    mag_cluster_blue.append(psfphot.finder_results[0]["mag"][index]) 
                    posx_blue.append(psfphot.finder_results[0]["xcentroid"][index])
                    posy_blue.append(psfphot.finder_results[0]["ycentroid"][index])
                    fluxb.append(phot["flux_fit"][index])
                    mag_err_b.append(mag_c_err[index])
                    rainteressb.append(gaia_sources1[0][index])
                    decinteressb.append(gaia_sources1[1][index])
                    obj_blue=brsources

            elif fil=="green":
                mag_cluster_green=[]
                posx_green=[]
                posy_green=[]
                parallaxg=[]
                fluxg=[]
                mag_err_g=[]
                rainteressg=[]
                decinteressg=[]
                for index in index_interest1: 
                    mag_cluster_green.append(psfphot.finder_results[0]["mag"][index])
                    posx_green.append(psfphot.finder_results[0]["xcentroid"][index])
                    posy_green.append(psfphot.finder_results[0]["ycentroid"][index])
                    fluxg.append(phot["flux_fit"][index])
                    mag_err_g.append(mag_c_err[index])
                    parallaxg.append(gaia_sources1[2][index])
                    rainteressg.append(gaia_sources1[0][index])
                    decinteressg.append(gaia_sources1[1][index])
                    obj_green=brsources

            elif fil=="red":
                mag_cluster_red=[]
                posx_red=[]
                posy_red=[]
                fluxr=[]
                mag_err_r=[]
                rainteressr=[]
                decinteressr=[]
                for index in index_interest1: 
                    mag_cluster_red.append(psfphot.finder_results[0]["mag"][index])
                    posx_red.append(psfphot.finder_results[0]["xcentroid"][index])
                    posy_red.append(psfphot.finder_results[0]["ycentroid"][index])
                    obj_red=brsources
                    mag_err_r.append(mag_c_err[index])
                    fluxr.append(phot["flux_fit"][index])
                    obj_red=brsources
                    rainteressr.append(gaia_sources1[0][index])
                    decinteressr.append(gaia_sources1[1][index])

        elif obs==2: 
            if fil=="blue":
                for index in index_interest2: 
                    mag_cluster_blue.append(psfphot.finder_results[0]["mag"][index])
                    posx_blue.append(psfphot.finder_results[0]["xcentroid"][index])
                    posy_blue.append(psfphot.finder_results[0]["ycentroid"][index])
                    obj_blue=brsources
                    mag_err_b.append(mag_c_err[index])
                    fluxb.append(phot["flux_fit"][index])
                    rainteressb.append(gaia_sources2[0][index])
                    decinteressb.append(gaia_sources2[1][index])

            elif fil=="green":
                for index in index_interest2: 
                    mag_cluster_green.append(psfphot.finder_results[0]["mag"][index])
                    posx_green.append(psfphot.finder_results[0]["xcentroid"][index])
                    posy_green.append(psfphot.finder_results[0]["ycentroid"][index])
                    obj_green=brsources
                    mag_err_g.append(mag_c_err[index])
                    fluxg.append(phot["flux_fit"][index])
                    parallaxg.append(gaia_sources2[2][index])
                    rainteressg.append(gaia_sources2[0][index])
                    decinteressg.append(gaia_sources2[1][index])
                    

            elif fil=="red":
                for index in index_interest2: 
                    mag_cluster_red.append(psfphot.finder_results[0]["mag"][index])
                    posx_red.append(psfphot.finder_results[0]["xcentroid"][index])
                    posy_red.append(psfphot.finder_results[0]["ycentroid"][index])
                    obj_red=brsources
                    mag_err_r.append(mag_c_err[index])
                    fluxr.append(phot["flux_fit"][index])
                    rainteressr.append(gaia_sources2[0][index])
                    decinteressr.append(gaia_sources2[1][index])

        
#####################################################################################
######################## DIFFERENT FILTERS MATCHING #################################
####################################################################################

i=0 
index_bg=[]
index_br=[]
stars=[]
stars2=[]

while i<len(mag_cluster_blue):#here we save the stars of each filter that have the same ra and dec and make pairs of them in order to calculate the B_V
    z=0
    y=0
    while z<len(mag_cluster_green): # one star with one index saved in mag_cluster_green could have another index in mag_cluster_blue
        if z not in stars:
            if rainteressb[i]==rainteressg[z] and decinteressb[i]==decinteressg[z]:
                index_bg.append((i,z))
                stars.append(z)
                break

        z=z+1
    while y<len(mag_cluster_red):
        if y not in stars2:
            if rainteressb[i]==rainteressr[y] and decinteressb[i]==decinteressr[y]:
                index_br.append((i,y))
                stars2.append(y)
                break
        y=y+1
    i=i+1

                    

b_g=[]
b_g_err=[]
flux_g=[]
g_mag=[]
g_err=[]
b_r=[]
b_r_err=[]
r_mag=[]
r_err=[]
flux_r=[]
b_mag=[]
b_err=[]
i=0
pargreen=[]
ragreen=[]
decgreen=[]


for index in index_bg: #here its where we calculate the B_G and get the green magnitudes (considering the visual as green)
    print(index)
    b_g.append(mag_cluster_blue[index[0]]-mag_cluster_green[index[1]])
    b_g_err.append(np.sqrt(mag_err_b[index[0]]*mag_err_b[index[0]]+mag_err_g[index[1]]*mag_err_g[index[1]]))
    g_mag.append(mag_cluster_green[index[1]])
    flux_g.append(fluxg[index[1]])
    g_err.append(mag_err_g[index[1]])
    b_mag.append(mag_cluster_blue[index[0]])
    b_err.append(mag_err_b[index[0]])
    pargreen.append(parallaxg[index[1]])  
    ragreen.append(rainteressg[index[1]])    
    decgreen.append(decinteressg[index[1]])          

print()
    

for index in index_br: #here its where we calculate the B_R and get the red magnitudes (considering the visual as red)
    b_r.append(mag_cluster_blue[index[0]]-mag_cluster_red[index[1]])
    b_r_err.append(np.sqrt(mag_err_b[index[0]]*mag_err_b[index[0]]+mag_err_r[index[1]]*mag_err_r[index[1]]))
    r_mag.append(mag_cluster_red[index[1]])
    flux_r.append(fluxr[index[1]])
    r_err.append(mag_err_r[index[1]])


from collections import OrderedDict

print("m of detected objects in B")
print(b_mag)
print("error of m in B:")
print(b_err)
print("m of detected objects in V")
print(g_mag)
print("error of m in V:")
print(g_err)
print("m of WR133 objects in V:")
print(np.min(g_mag))
print("m of detected objects in R")
print(r_mag)
print("error of min R")
print(r_err)
print("b_g")
print(b_g)
print("error of b_g")
print(b_g_err)
print("flux of detected objects in V")
print(fluxg)
print("parallax of detected objects in V")
print(pargreen)
print("ra of detected objects in V")
print(ragreen)
print("dec of detected objects in V")
print(decgreen)


print("number of objects detected:")
print(len(g_mag))

import matplotlib.pyplot as plt
  

errorg=[g_err,g_err]  #just organizing the errors bars to use in the HR plot
errorb_g=[b_g_err,b_g_err]

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

############################################################################
############ ISOCHRONES AND HR DIAGRAM #####################################
############################################################################
   
isochrones = Table.read('isochrones2.dat', format='ascii') #here we read one generated text file with diferent LogAge and Z parameters that characterize different isochrones
                                                           #in order to find the parameters that generate the isochrone that adjusts our HR diagram points the best 

ages=np.unique(isochrones['logAge'])
metal=np.unique(isochrones['Z'])
imf=np.unique(isochrones['int_IMF'])
mass=np.unique(isochrones['Mass'])

N=230 #

#we will make steps in a certain interval of distances to get the best value for cluster distance 
dmin,dmax,step=1500,1650,5
distances = np.arange(dmin,dmax,step) 

#define an array to save the root-mean-square deviation values
rmsd=np.zeros(shape=(len(ages),len(distances)))
for i in range(len(ages)):
    age=ages[i]
    for j in range(len(distances)):
        
        ## model
        distance=distances[j]
        DM=5*np.log10(distance)-5 #distance modulus
        isochrone=isochrones[isochrones['logAge'] == age][0:N]
        col_iso = isochrone['Bmag'] - isochrone['Rmag'] #color isochrone
        mag_iso = isochrone['mbolmag'] + DM #magnitude isochrone, shifted to the distance of the cluster
        line = LineString(np.asarray([col_iso,mag_iso]).T) #Representation of the isochrone as a continuous line 

        ## data
        d=np.empty(len(g_mag))
        for k in range(len(g_mag)):
            col_data=b_g[k]
            mag_data=g_mag[k]
            point=Point(col_data,mag_data)
            d[k] = point.distance(line) #shortest distance of the point to the line of the isochrone
        rmsd[i,j]=np.sqrt(np.nanmean(d)**2) 

fig,ax = plt.subplots(figsize=(7,7))
lo,up = np.percentile(rmsd, 1),np.percentile(rmsd, 99)
pos=ax.imshow(rmsd,cmap='PiYG', norm=LogNorm(),origin='lower',
               extent=[distances[0],distances[-1],10**ages[0]/1e6,10**ages[-1]/1e6],aspect='auto')
fig.colorbar(pos, ax=ax,format= "%d")

#Find the grid position of the minimum rmsd
minrmsd_pos=np.unravel_index(rmsd.argmin(), rmsd.shape)
print(np.nanmin(rmsd),minrmsd_pos)
print("*** Best fit model: age = ", (10**ages[minrmsd_pos[0]]/1e6),'Myr; distance=',distances[minrmsd_pos[1]],'pc')
print("Best fit model:metal= ",metal[minrmsd_pos[0]],"; IMF:",imf[minrmsd_pos[0]],";Mass:",mass[minrmsd_pos[0]])
best_age=ages[minrmsd_pos[0]]
best_dist=distances[minrmsd_pos[1]]
plt.xlabel('distance in pc',fontsize=15)
plt.ylabel('age in Myr',fontsize=15)

output_dir = 'photometry/isochrones'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

filename=os.path.join(output_dir, f'best_isochrone.png')

overwrite(filename)

plt.clf()  

fig = plt.figure(figsize=(7,7))
#plt.scatter(b_g,g_mag) # x can be also  data['phot_bp_mean_mag']-data['phot_rp_mean_mag']
plt.xlim(-10,10)
plt.ylim(21,2)
plt.xlabel('B - V',fontsize=14)
plt.ylabel('V ',fontsize=14)

plt.errorbar(b_g, g_mag, xerr=errorb_g, capsize=2,markersize=5,fmt='o', color='cornflowerblue',ecolor='lightgrey',zorder=1) #here we plot our HR Diagram with the isochrone that fits it better
plt.errorbar(b_g, g_mag, yerr=errorg, capsize=2,markersize=5, fmt='o',color='cornflowerblue', ecolor='lightgrey',zorder=1)

#Important: isochrones are given in absolute magnitudes, while the data come in apparent
#median_parallax=np.nanmedian(data['parallax'])
#DM=5*np.log10(best_dist)-5 #distance modulus
#print('distance modulus:',DM)

age_1 = isochrones['logAge'] == best_age
plt.plot(isochrones['Bmag'][age_1][0:N] - isochrones['Rmag'][age_1][0:N], isochrones['mbolmag'][age_1][0:N] + DM,label=str(np.round(10**best_age/1e6))+' Myr',color='red')
plt.legend()


filename=os.path.join(output_dir, f'isochrone_age_cluster.png')

overwrite(filename)

plt.clf() 


