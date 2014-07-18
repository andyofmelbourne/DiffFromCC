
import numpy as np
import crossTermsTools as cm
import matplotlib.pyplot as plt
import random
import scipy as sp
from scipy import signal
import viewTools as vt
import time
import sys


class CCextract:
    def __init__(self,configpath=None,configfile=None):

        # hack for testing
        np.random.seed( 2343423)

        #
        # Input diffraction pattern/other initial set up
        #
        if configpath == None :
            # configpath = '/home/amartin1/Work/Results/multipleParticlesCC/Experiment_Feb2013/'
            # configpath = '/home/amartin1/Work/Results/multipleParticlesCC/Simulations_Feb2013/'
            configpath = '/home/amartin1/Work/Experiments/Fermi/threeVirus/A1/'
        if configfile == None :
            # configfile = 'config.txt'
            configfile = 'config_crop.txt'
        
        print "config:", configpath+configfile
        cr  = cm.cross(configfile=configpath+configfile)
        
        #
        # Initialise display
        #
        if len(cr.display_options) > 0:
            self.wm  = vt.WindowManager()
        if 'difpat' in cr.display_options:
            self.wm.NewWindow(cr.image,label="Figure",pause=cr.pause)
        
        
        #
        # output diffraction pattern if requested
        #
        if cr.noise_flag > 0 :
            cr.addPoissonNoise()
            
        #
        # output diffraction pattern if requested
        #
        if 'difpat' in cr.output :
            cr.output_image(cr.image,outname='difpat_input',pngGamma=cr.pngGamma,pngColor=cr.pngcolor)

        #
        # remove saturation from detector
        #
        if cr.unsaturate_flag == 1 :
            cr.unsaturate()
            if 'saturation_removed' in cr.display_options:
                self.wm.windowlist["Figure"].update(cr.image,pause=cr.pause)

            if 'config' in cr.log_options:
                cr.flog.write("Saturation masking has been applied\n")
            if 'config' in cr.verbose:
                print "Saturation masking has been applied"
        else :
            if 'config' in cr.log_options:
                cr.flog.write("Saturation masking has not been applied\n")
            if 'config' in cr.verbose:
                print "Saturation masking has not been applied"

        #
        # apply filter to remove beamstop region (central detector hole)
        #
        if cr.beamstopfilter_flag == 1:
            cr.beamStopFilter(cr.image,rad=0.03)
            if 'beamstop_applied' in cr.display_options:
                self.wm.windowlist["Figure"].update(cr.image,pause=cr.pause)
                
            if 'config' in cr.log_options:
                cr.flog.write("Beam-stop filter has been applied\n")
            if 'config' in cr.verbose:
                print "Beam-stop filter has been applied"
        else :
            if 'config' in cr.log_options:
                cr.flog.write("Beam-stop filter has not been applied\n")
            if 'config' in cr.verbose:
                print "Beam-stop filter has not been applied"


        #
        # Perform the extraction of the single particle diffraction patterns
        #
        if cr.scan == 1:
            if 'config' in cr.log_options:
                cr.flog.write("Calcuation of patterns by SCANNING through Gaussian masks\n")
            if 'config' in cr.verbose:
                print "Calcuation of patterns by SCANNING through Gaussian masks"
                
            self.scan(cr)
            
        else :
            if 'config' in cr.log_options:
                cr.flog.write("Gaussian masks assembled into single mask\n")
            if 'config' in cr.verbose:
                print "Gaussian masks assembled into single mask"

            self.calculate_diffs(cr)
            
        #
        # Output retrieved diffraction patterns
        #
        self.output_diffs(cr)


        #
        # Close log file.
        # wm.show() calls mainloop() routine for the image display
        #
        cr.flog.close()

        if len(cr.display_options) > 0:
            self.wm.show()


        ##########################################################################
        #
        # Function to do the main calculation by ptychographic scanning
        #
        ##########################################################################

    def scan(self,cr):
        #
        # Define CC regions from combined mask (not scanned)
        #
        #self.applyGauss(cr)
        #self.formAutoc(cr)
        #self.findCCs(cr)
        #cr.makePermutations()
        #self.findTriplets(cr)
        
        
        #
        # save the lists of guassian parameters
        #
        gx = cr.xgauss
        gy = cr.ygauss
        gw = cr.wgauss

        l = len(gx)
        diffs = []

        for i in np.arange(l):
            if 'config' in cr.log_options:
                cr.flog.write("\n Calculating scan position %(i)i / %(len)i \n\n" %{"i":i+1,"len":l} )
                cr.flog.write("mask x : %(x)i \n" %{"x":gx[i]} )
                cr.flog.write("mask y : %(x)i \n" %{"x":gy[i]} )
                cr.flog.write("mask width : %(x)i \n" %{"x":gw[i]} )
            if 'config' in cr.verbose:
                print '\n Calculating scan position', i+1, '/', l, '\n'
                print 'mask x : ', gx[i]
                print 'mask x : ', gy[i]
                print 'mask x : ', gw[i] 
            
            #
            # select the appropriate guassian parameters
            #
            cr.xgauss = []
            cr.xgauss.append( gx[i]  )

            cr.ygauss = []
            cr.ygauss.append( gy[i]  )

            cr.wgauss = []
            cr.wgauss.append( gw[i]  )

            #
            # Using CC regions determined from combined mask
            #
            #self.applyGauss(cr)
            #self.formAutoc(cr)
            #cr.updateCross()
            #cr.makeDiffsDivMask()

            # Determine CC regions for each position
            self.calculate_diffs(cr)
            

            ld_new  = len(cr.diffs)
            ld_old  = len(diffs)
            #print "ld_new, ld_old :", ld_new, ld_old

            #
            # This could be more sophisticated
            # to check that the right permutations are always added
            #
            
            if   ld_new > ld_old:
                for i in np.arange(ld_new - ld_old):
                    diffs.append( np.zeros( (cr.nx,cr.ny) ) )

            for j in np.arange(ld_new):
                diffs[j] += (cr.diffs[j] - diffs[j]*cr.gauss)

            self.output_diffs(cr)

        #
        # 
        #
        ld_old  = len(diffs)
        cr.Ndiffs = ld_old
        cr.diffs = np.zeros((ld_old,cr.ny,cr.nx))
        for j in np.arange(ld_old):
            cr.diffs[j] = diffs[j]

        #
        # restore the original lists of gaussian positions
        #
        cr.xgauss = gx
        cr.ygauss = gy
        cr.wgauss = gw
            

        ##########################################################################
        #
        # Function to do the main calculation steps to extract to the
        # single particle diffraction patterns
        #
        ##########################################################################

    def calculate_diffs(self,cr):

            #
            # apply Gaussians to select the useful area of the pattern
            #
            print '*********************************************************'
            print 'cr.gaussMask',cr.gaussMask
            if cr.gaussMask != 0 :            
                self.applyGauss(cr)
            else:
                cr.gauss = np.ones( (cr.nx, cr.nx) )

            #
            # form the autocorrelation function
            #
            self.formAutoc(cr)
            
            
            # sys.exit(0)

            #
            # find the useful signal in the autocorrelation function
            #
            self.findCCs(cr)
            

            #
            # Make the combinations of cross-correlation terms
            #
            cr.makePermutations()

                        
            #
            # Identify the correct combinations of cross-correlation terms
            #
            self.findTriplets(cr)
            # now we have a list of valid cross term combinations

            #
            # Calculate the diffraction patterns
            #
            if cr.div_quad_solution == 1 :
                cr.makeDiffsDivMask()
                # cr.makeDiffsDivMask_GoodRegions()

            #
            # Calculate the diffraction patterns using the quadratic formula
            #

            if cr.div_quad_solution == 2:
                cr.makeDiffsQuadratically()

            #
            # Regularise to clean up the diffraction patterns
            #
            self.projectCentre(cr)

            
            #
            # Low pass filter
            # 
            if cr.lowPassDiffs == 1:
                cr.lowPassFilterDiffs()

         
            #
            #  Mask artifacts and Wiener filter
            #                
            if cr.mask_artifacts_flag == 1:
                cr.mask_artifacts()

            #
            # Clean up the diffs where the mask is very low valued
            #

            # print "gaussMask:", cr.gaussMask
            # if cr.gaussMask != 0 :        
            #     cr.cleanDiffsGauss()
            #     cr.test_noise_level()


    def applyGauss(self,cr):
            #
            # apply Gaussians to select the useful area of the pattern
            #
            if cr.gauss_flag == 1:
                    cr.makeGauss()
           
            if cr.applygauss == True:
                cr.image *= cr.gauss
                if 'mask' in cr.display_options:
                    fig_gauss = self.wm.NewWindow(cr.gauss,label="Gaussian mask",pause=cr.pause)

                if 'mask_applied' in cr.display_options:
                    self.wm.windowlist["Figure"].update(cr.image,pause=cr.pause)

                if 'config' in cr.log_options:
                    cr.flog.write("Gaussian mask has been used\n")
                if 'config' in cr.verbose:
                    print "Gaussian mask has been used"

                if 'difpat_mask' in cr.output:
                    cr.output_image(np.abs(cr.gauss),outname='difpat_mask',\
                                    pngGamma=cr.pngGamma,pngColor=cr.pngcolor)

                if 'difpat_after_mask' in cr.output:
                    cr.output_image(np.abs(cr.image),outname='difpat_after_mask',\
                                    pngGamma=cr.pngGamma,pngColor=cr.pngcolor)

            else :
                if 'config' in cr.log_options:
                    cr.flog.write("Gaussian mask has not been used\n")
                if 'config' in cr.verbose:
                    print "Gaussian mask has not been used"


    def formAutoc(self,cr):
            #
            # form the autocorrelation function
            #            
            cr.autoc = cm.ifft2(cr.image)
            if 'autocorrelation' in cr.display_options:
                self.wm.windowlist["Figure"].update(cr.autoc,pause=cr.pause)

            if 'autoc' in cr.output:
                cr.output_image(np.abs(cr.autoc),outname='autocorrelation',pngGamma=cr.pngGamma,pngColor=cr.pngcolor)
                cr.output_image(np.real(cr.autoc),outname='autocorrelation_real',pngGamma=cr.pngGamma,pngColor=cr.pngcolor)
                cr.output_image(np.imag(cr.autoc),outname='autocorrelation_imag',pngGamma=cr.pngGamma,pngColor=cr.pngcolor)



    def findCCs(self,cr):

            cr.mask_from_statistical_threshold()

            if cr.Ncross <= 2:
                cr.Ndiffs = 0
                return

            if 'autoc_mask' in cr.output:
                cr.output_image(np.abs(cr.mask),outname='autocorrelation_mask',pngGamma=cr.pngGamma,pngColor=cr.pngcolor)

            if 'autoc_times_mask' in cr.output:
                cr.output_image(np.abs(cr.mask)*np.abs(cr.autoc),outname='autocorrelation_times_mask',pngGamma=cr.pngGamma,pngColor=cr.pngcolor)

    def findTriplets(self,cr):

            if  cr.permMethod  == 'imaginary':
                    cr.checkPermutations(fout=cr.flog)
            elif cr.permMethod == 'distance':
                    cr.checkPermByDistance()
            else :
                print "Method for checking permutations not valid.\n" \
                      +"Please choose 'distance' or 'imaginary'\n"
                cr.flog.close()
                wm.show()    


    def projectCentre(self,cr):
            if cr.project_flag == 1:
                cr.project()
                if 'config' in cr.log_options:
                    cr.flog.write("Regularization by projection has been applied\n")
                if 'config' in cr.verbose:
                    print "Regularization by projection has been applied"
            else :
                if 'config' in cr.log_options:
                    cr.flog.write("Regularization by projection has not been applied\n")
                if 'config' in cr.verbose:
                    print "Regularization by projection has not been applied"

        ##########################################################################
        #
        # Function to output the recovered diffraction patterns to file
        #
        ##########################################################################
    def output_diffs(self,cr):

        if cr.Ndiffs >= 3:
            if 'diffs' in cr.display_options:
                fig2 = self.wm.NewWindow(cr.diffs[0],label="diffs[0]",pause=cr.pause)
                fig3 = self.wm.NewWindow(cr.diffs[1],label="diffs[1]",pause=cr.pause)
                fig4 = self.wm.NewWindow(cr.diffs[2],label="diffs[2]",pause=cr.pause)

            #
            # Output retrieved diffraction patterns
            #
            if 'diffs' in cr.output:
                for i in np.arange( len(cr.diffs) ):
                    fname = 'diffs'+('%(i)02d' %{'i':i+1} )
                    #cr.output_image(np.abs(cr.diffs[i]),outname=fname,pngGamma=cr.pngGamma,pngColor=cr.pngcolor)
                    cr.output_image( cr.diffs[i] ,outname=fname,pngGamma=cr.pngGamma,pngColor=cr.pngcolor)

                    fname = 'autoc'+('%(i)02d' %{'i':i+1} )
                    #cr.output_image(np.abs(cr.diffs[i]),outname=fname,pngGamma=cr.pngGamma,pngColor=cr.pngcolor)
                    cr.output_image( np.abs(np.fft.fft2(cr.diffs[i])) ,outname=fname,pngGamma=cr.pngGamma,pngColor=cr.pngcolor)
            #
            # check the reconstruction by comparing to the central part
            #                
            cr.checkCentralCC()
            if 'autocorrelation_check' in cr.display_options:
                fig5 = self.wm.NewWindow(cr.estimate,label="Estimate",pause=cr.pause)
                fig6 = self.wm.NewWindow(cr.central,label="Central Part",pause=cr.pause)

