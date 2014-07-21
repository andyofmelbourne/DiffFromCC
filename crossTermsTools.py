#------------------------------------------------------------------------------------------
# Authors: Andrew Mortin, Andrew Morgan
# Paper reference: 
#   A. Martin, A. Morgan, T. Ekeberg, N. Loh, F. Maia, F. Wang, J. Spence, and H. Chapman, 
#   "The extraction of single-particle diffraction patterns from a multiple-particle 
#   diffraction pattern," Opt. Express  21, 15102-15112 (2013).
#
# Based on CrossTermsMod.py by A. Morgan 09/2011
# 
# A. Martin 10/2011 
#   - changed input/output variables for cross class 
#       - only necessary functions retained
#   - cross.project2Fast renamed cross.project()
#       - many functions moved to MorgansFunctions.py
#------------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mpcolors
import mahotas
# import pylab

from scipy import fftpack
from scipy import ndimage
from scipy.stats.distributions import poisson
import scipy as sp

import pymorph
#from random import *

import random
import time
import my_favorites as mf
from MorgansFunctions import *
import os
from platform import system
import string
import shutil
import sys


#
#  Class cross:
#
#  functions:
#       - parse_config_file(self,configfile="config.txt")
#       - write_settings(self)
#       - unsaturate(self)
#       - beamStopFilter(self,arrayin,rad=0.07)
#       - makeGauss(self,a=0.0001)
#       - maskAutoc(self)
#       - maskAutocWatershed(self)
#       - makeCross(self)
#       - makePermutations(self)
#       - checkPermutations(self,fout=None)
#       - makeDiffsDivMask(self)
#       - project(self)
#       - mask_from_statistical_threshold(self)
#       - output_image(self,image,outname='image',pngGamma=1.0,pngColor='ocean')
#

class cross:
    """A class for crossTerms."""
    def __init__(self,fname = None,ny=None,nx=None,path=None,outpath=None,h5field=None,configfile=None):
        """Initialise the class arrays and variables."""
        if path == None :
            self.path = '/cfel/user/amartin/manyparticle/' 
        else :
            self.path = path
                
        if h5field == None :
            self.h5field = '/data/data0' 
        else :
            self.h5field = h5field

        if fname == None :
            self.image = None
        else :
            self.fname = fname
            self.fbase, self.fext = os.path.splitext(self.path+fname)

        if outpath == None:
            self.outpath = self.path
        else :
            self.outpath = outpath

        self.flogname = "log.txt"
                
        self.autoc        = None
        self.mask         = None
        self.cmask        = None
        self.badpix       = None
        self.permutations = None 
        self.gauss        = None
        self.Ncross       = None
        self.Ndiffs       = None
        self.cross        = None
        self.exitWave     = None
        self.diffs        = None
        self.diffsScan    = None
        self.diffKnown    = None
        self.autocCentralMask = None
        self.crossnoiseEst   = None
        self.diffPairs    = []
        self.tol          = 0.6
        self.distTol      = 5   # distance in pixels
        self.display_options  = [] # 'difpat' 'saturation_removed' 'beamstop_applied', 'mask', 'mask_applied',
                                   # 'diffs', 'autocorrelation', 'autocorrelation_check'
        self.output           = [] # 'difpat', 'diffs', 'autoc', 'mask_centre_cut', 'mask_guassian_blur',
                                   # 'mask_otsu', 'mask_labelled', 'difpat_mask', 'difpat_after_mask' 
        self.output_formats   = [] # '.raw', '.h5', '.png'
        self.log_options      = [] # 'write', 'config', 'perm_err', 'errors'
        self.verbose          = [] # 'write', 'config', 'perm_err', 'errors'
        self.xgauss           = []
        self.ygauss           = []
        self.wgauss           = []
        self.xfilter          = []
        self.yfilter          = []
        self.wfilter          = []
        self.applygauss       = False
        self.unsaturate_flag  = 0
        self.beamstopfilter_flag = 0
        self.pngcolor         = 'ocean'
        self.project_flag     = 0
        self.permMethod       = 'imaginary'
        self.pause            = 0.01
        self.divtol           = -1 # negative values mean tolerance is ignored
        self.centralError     = -1.
        self.lowpassrad       = 0.05 #the radius of a circular lowpass filter a fraction of the array size (for autoc)
        self.pngGamma         = 0.3
        self.noise_flag       = 0
        self.scan_regions     = scan_regions()
        self.filter_regions   = scan_regions()
        self.scan             = 0  # default is not to scan
        self.gaussMask        = 0  # default is not to apply a gaussian mask
        self.div_quad_solution = 1  # default is to use complex division
        self.lowPassDiffs      = 0  #default not to apply a lowpass filter to the diffs.
        self.speckleWidth     = 20
        self.artifact_noise_tol = 3.0
        self.mask_artifacts_flag = 0
       
        self.maxCCs = 15

        # morphological parameters
        self.autoc_mask_thresh = []
        self.rado = 4
        self.radc = 8
        self.radd = 8

        # Crop parameters
        self.cropx_min = -1
        self.cropx_max = -1
        self.cropy_min = -1
        self.cropy_max = -1


        #
        # Gaussian mask parameters
        #
        self.gauss_flag = 1     # 1 = make mask ; 0 = mask read from file
        self.maskpath = self.path
        self.maskname = None

        self.autocCentralMaskname = None
    

        #
        # read the settings from a file and open log and image files.
        #
        self.parse_config_file(configfile=configfile)

        self.flog = open(os.path.normpath(self.outpath+self.flogname),'w')

        if self.fname != None:                       
            try :
                if (self.fext == '.raw') | (self.fext == '.txt'): 
                    if ny != None :
                       self.ny       = ny
                    if nx != None :
                        self.nx      = nx
                    self.image = imageJinRaw(os.path.normpath(self.path+self.fname),self.ny,self.nx)
                elif (self.fext == '.h5'):
                    self.image = mf.h5read(os.path.normpath(self.path+self.fname),field=self.h5field) 

                    self.ny = self.image.shape[0]
                    self.nx = self.image.shape[1]
                    print "image has been read.", np.max(self.image)
            except :
                print "No valid input format given"           
                print "Could not open file:", os.path.normpath(self.path+self.fname)
                print self.h5field

        # Crop image
        image2 = self.image[self.cropx_min:self.cropx_max,self.cropy_min:self.cropy_max]
        self.image = np.copy(image2)
        self.ny = self.image.shape[0]
        self.nx = self.image.shape[1]
                 

        if self.maskname != None :
            self.gauss = mf.h5read(os.path.normpath(self.maskpath+self.maskname),field=self.h5field) 
            self.gauss_flag = 0

        if self.autocCentralMaskname != None:
            print "reading central mask:", self.autocCentralMaskname
            self.autocCentralMask = mf.h5read(self.autocCentralMaskname,field='/data/data') 


        # ensure image is type float
        self.image = self.image.astype(float)
        self.inputImage = np.copy(self.image)

        #
        # write the config information into the log file and terminal.
        #
        self.write_settings()

        #
        # Fill in more Gaussian data from scan regions
        #
        num_regions, max_num_regions = self.scan_regions.check_list_sizes()
        for i in np.arange(num_regions):
                numx = (self.scan_regions.xend[i] - self.scan_regions.xstart[i]) \
                                          / self.scan_regions.xstep[i]
                numy = (self.scan_regions.yend[i] - self.scan_regions.ystart[i]) \
                                          / self.scan_regions.ystep[i]

                for j in np.arange(numx+1):
                        for k in np.arange(numy+1):
                                x0 = self.scan_regions.xstart[i] + j* self.scan_regions.xstep[i]
                                y0 = self.scan_regions.ystart[i] + k* self.scan_regions.ystep[i]
                                self.xgauss.append( x0 )
                                self.ygauss.append( y0 )
                                self.wgauss.append( self.scan_regions.width[i] )

        #
        #  Check if all wguass is same value (then we can speed up the calculation)
        #
        test=0
        for i in np.arange(len(self.wgauss)-1)+1:
            if self.wgauss[i] != self.wgauss[0]:
                test += 1
        if test != 0:
            self.slowGaussSum = 1
        else:
            self.slowGaussSum = 0
        print "slowGaussSum", self.slowGaussSum
   
        #
        # Fill in scan information for local filter
        #
        num_regions, max_num_regions = self.filter_regions.check_list_sizes()
        for i in np.arange(num_regions):
                numx = (self.filter_regions.xend[i] - self.filter_regions.xstart[i]) \
                                          / self.filter_regions.xstep[i]
                numy = (self.filter_regions.yend[i] - self.filter_regions.ystart[i]) \
                                          / self.filter_regions.ystep[i]

                for j in np.arange(numx+1):
                        for k in np.arange(numy+1):
                                x0 = self.filter_regions.xstart[i] + j* self.filter_regions.xstep[i]
                                y0 = self.filter_regions.ystart[i] + k* self.filter_regions.ystep[i]
                                self.xfilter.append( x0 )
                                self.yfilter.append( y0 )
                                self.wfilter.append( self.filter_regions.width[i] )

        if len(self.xgauss) != 0 : self.applygauss = True
                
    def parse_config_file(self,configfile="config.txt"):
        fin       = open(configfile,'r')
        fnameflag = 0

        for line in fin:
            linesplit = string.split(line,None,2)
            
            # if there is a 'dud' line then skip it 
            is_dud = False
            if len(linesplit) < 3 :
                is_dud = True
            elif line[0] == '#' :
                is_dud = True
            elif linesplit[1] != "=" :
                is_dud = True

            if not is_dud : 
                linesplit0_low = string.lower(linesplit[0])
                linesplit[2]   = linesplit[2].rstrip()
            
                if linesplit0_low == 'path':
                        self.path = linesplit[2]
                elif linesplit0_low == 'outpath':
                        self.outpath = linesplit[2]
                elif linesplit0_low == 'fname':
                        self.fname = linesplit[2]
                        fnameflag = 1
                elif linesplit0_low == 'h5field':
                        self.h5field = linesplit[2]
                elif linesplit0_low == 'tol':
                        self.tol = float(linesplit[2])
                elif linesplit0_low == 'disttol':
                        self.distTol = float(linesplit[2])
                elif linesplit0_low == 'unsaturate':
                        self.unsaturate_flag = float(linesplit[2])
                elif linesplit0_low == 'beamstopfilter':
                        self.beamstopfilter_flag = float(linesplit[2])
                elif linesplit0_low == 'output':
                        self.output.append(linesplit[2])
                elif linesplit0_low == 'output_formats':
                        self.output_formats.append(linesplit[2])
                elif linesplit0_low == 'log_options':
                        self.log_options.append(linesplit[2])
                elif linesplit0_low == 'verbose':
                        self.verbose.append(linesplit[2])
                elif linesplit0_low == 'logfile':
                        self.flogname = linesplit[2]
                elif linesplit0_low == 'xgauss':
                        self.xgauss.append(int(linesplit[2]))
                elif linesplit0_low == 'ygauss':
                        self.ygauss.append(int(linesplit[2]))
                elif linesplit0_low == 'wgauss':
                        self.wgauss.append(float(linesplit[2]))
                elif linesplit0_low == 'pngcolor':
                        self.pngcolor = linesplit[2]
                elif linesplit0_low == 'project':
                        self.project_flag = int(linesplit[2])
                elif linesplit0_low == 'checkpermutationsmethod':
                        self.permMethod = linesplit[2]
                elif linesplit0_low == 'pause':
                        self.pause = float(linesplit[2])
                elif linesplit0_low == 'divtol':
                        self.divtol = float(linesplit[2])
                elif linesplit0_low == 'lowpassrad':
                        self.lowpassrad = float(linesplit[2])
                elif linesplit0_low == 'pnggamma':
                        self.pngGamma = float(linesplit[2])
                elif linesplit0_low == 'noise':
                        self.noise_flag = long(linesplit[2])
                elif linesplit0_low == 'mask_xstart':
                        self.scan_regions.xstart.append( int(linesplit[2]) )
                elif linesplit0_low == 'mask_ystart':
                        self.scan_regions.ystart.append( int(linesplit[2]) )
                elif linesplit0_low == 'mask_xend':
                        self.scan_regions.xend.append(  int(linesplit[2])  )
                elif linesplit0_low == 'mask_yend':
                        self.scan_regions.yend.append(  int(linesplit[2])  )
                elif linesplit0_low == 'mask_xstep':
                        self.scan_regions.xstep.append(  int(linesplit[2]) )
                elif linesplit0_low == 'mask_ystep':
                        self.scan_regions.ystep.append(  int(linesplit[2]) )
                elif linesplit0_low == 'mask_width':
                        self.scan_regions.width.append(  float(linesplit[2]) )
                elif linesplit0_low == 'filter_xstart':
                        self.filter_regions.xstart.append( int(linesplit[2]) )
                elif linesplit0_low == 'filter_ystart':
                        self.filter_regions.ystart.append( int(linesplit[2]) )
                elif linesplit0_low == 'filter_xend':
                        self.filter_regions.xend.append(  int(linesplit[2])  )
                elif linesplit0_low == 'filter_yend':
                        self.filter_regions.yend.append(  int(linesplit[2])  )
                elif linesplit0_low == 'filter_xstep':
                        self.filter_regions.xstep.append(  int(linesplit[2]) )
                elif linesplit0_low == 'filter_ystep':
                        self.filter_regions.ystep.append(  int(linesplit[2]) )
                elif linesplit0_low == 'filter_width':
                        self.filter_regions.width.append(  float(linesplit[2]) )
                elif linesplit0_low == 'scan':
                        self.scan  = int(linesplit[2])
                elif linesplit0_low == 'display_options':
                        self.display_options.append( linesplit[2] )
                elif linesplit0_low == 'binary_opening_radius':
                        self.rado = int(linesplit[2])
                elif linesplit0_low == 'binary_closing_radius':
                        self.radc = int(linesplit[2])
                elif linesplit0_low == 'binary_dilation_radius':
                        self.radd = int(linesplit[2])
                elif linesplit0_low == 'autoc_mask_thresh':
                        self.autoc_mask_thresh.append( float(linesplit[2]) )
                elif linesplit0_low == 'maskpath':
                        self.maskpath  = linesplit[2]
                elif linesplit0_low == 'maskname':
                        self.maskname  = linesplit[2]
                elif linesplit0_low == 'ny':
                        self.ny = int(linesplit[2])
                elif linesplit0_low == 'nx':
                        self.nx = int(linesplit[2])
                elif linesplit0_low == 'gaussmask':
                        self.gaussMask = int(linesplit[2])
                elif linesplit0_low == 'div_quad_solution':
                        self.div_quad_solution = int(linesplit[2])
                elif linesplit0_low == 'lowpassdiffs':
                        self.lowPassDiffs = int(linesplit[2])
                elif linesplit0_low == 'autoccentralmaskname':
                        self.autocCentralMaskname  = linesplit[2]
                elif linesplit0_low == 'specklewidth':
                        self.speckleWidth = int(linesplit[2])
                elif linesplit0_low == 'artifact_noise_tol':
                    self.artifact_noise_tol = float(linesplit[2])
                elif linesplit0_low == 'maskartifactsflag':
                    self.mask_artifacts_flag = int(linesplit[2])
                elif linesplit0_low == 'cropx_min':
                    self.cropx_min = int(linesplit[2])
                elif linesplit0_low == 'cropx_max':
                    self.cropx_max = int(linesplit[2])
                elif linesplit0_low == 'cropy_min':
                    self.cropy_min = int(linesplit[2])
                elif linesplit0_low == 'cropy_max':
                    self.cropy_max = int(linesplit[2])
                else :
                        print "option not found:", linesplit[0]

        fin.close() 
        if fnameflag == 1:
            self.fbase, self.fext = os.path.splitext(self.path+self.fname)
            if os.path.normpath(configfile) != os.path.normpath(self.outpath+"config.txt") :
                shutil.copy2(os.path.normpath(configfile), os.path.normpath(self.outpath+"config.txt"))

    def write_settings(self):
        if 'config' in self.log_options:
            self.flog.write("Options that were read from file:\n")
            self.flog.write("path : "+self.path+'\n')
            self.flog.write("outpath : "+self.outpath+'\n')
            self.flog.write("fname : "+self.fname+'\n')
            self.flog.write("h5field : "+self.h5field+'\n')
            self.flog.write("tol : "+('%(tol).2f\n' %{"tol":self.tol})  )
            self.flog.write("distTol : "+('%(tol).2f\n' %{"tol":self.distTol})  )
            self.flog.write("png color : "+self.pngcolor+'\n')
            self.flog.write("png gamma : "+('%(tol).2f\n' %{"tol":self.pngGamma})  )
            self.flog.write("Method for checking permutations : "+self.permMethod+'\n')
            self.flog.write("division tolerance : "+('%(tol).2f\n' %{"tol":self.divtol})  )
            self.flog.write("low pass filter radius : "+('%(tol).2f\n' %{"tol":self.lowpassrad})  )
            self.flog.write("noise flag : "+('%(tol)i\n' %{"tol":self.noise_flag})  )
            self.flog.write("binary opening radius : "+('%(rad).2f\n' %{"rad":self.rado})  )
            self.flog.write("binary closing radius : "+('%(rad).2f\n' %{"rad":self.radc})  )
            self.flog.write("binary dilation radius : "+('%(rad).2f\n' %{"rad":self.radd})  )
            self.flog.write( "Output low pass filtered : "+('%(l)i\n' %{"l":self.lowPassDiffs})   )
            self.flog.write( "tolerance (# of standard deviations) for finding artifacts : "+('%(l).2f\n' %{"l":self.artifact_noise_tol})   )
            self.flog.write( "artifacts will be masked : "+('%(l)i\n' %{"l":self.mask_artifacts_flag})   )
            

            for i in np.arange(len(self.autoc_mask_thresh)):
                    self.flog.write( ("autoc mask threshold %(i)d : %(rad).2f\n" \
                                            %{"i":i,"rad":self.autoc_mask_thresh[i]}) )
                    
            self.flog.write("\n")

        if 'config' in self.verbose:
                print "Options that were read from file:"
                print "path : "+self.path
                print "outpath : "+self.outpath
                print "fname : "+self.fname
                print "h5field : "+self.h5field
                print "tol : "+('%(tol).2f' %{"tol":self.tol})
                print "distTol : "+('%(tol).2f' %{"tol":self.distTol})
                print "png color : "+self.pngcolor
                print "png gamma : "+('%(tol).2f' %{"tol":self.pngGamma})
                print "Method for checking permutations: "+self.permMethod
                print "Division tolerance : "+('%(tol).2f' %{"tol":self.divtol})
                print "Low pass filter radius : "+('%(tol).2f' %{"tol":self.lowpassrad})
                print "noise flag : "+('%(tol)i' %{"tol":self.noise_flag})
                print "Output low pass filtered : "+('%(l)i' %{"l":self.lowPassDiffs}   )
                print "tolerance (# of standard deviations) for finding artifacts : "+('%(l)i\n' %{"l":self.artifact_noise_tol})   
                if self.gauss_flag == 0:
                        print "Mask read from file"
                else:
                        print "Mask will be created from input parameters"

                for i in np.arange(len(self.autoc_mask_thresh)):
                         print ("autoc mask threshold %(i)d : %(rad).2f" \
                                                %{"i":i+1,"rad":self.autoc_mask_thresh[i]})
                print '\n'
    
    def unsaturate(self):
        """Zeros saturated pixels from a detector array.
        """
        array2 = np.asarray(self.image,dtype=np.uint32)
        array2 = ndimage.gaussian_filter(array2,8)
        print array2.shape
        T      = mahotas.thresholding.otsu(array2)
        self.image = self.image * (self.image<T)
    
    def beamStopFilter(self,arrayin,rad=0.07):
        """Filters using circular beam stop"""
        array = np.zeros(arrayin.shape)
        array = circle(array,radius = rad)
               
        array = 1.0 - ndimage.gaussian_filter(array,8)
        array = arrayin * array
        self.image = array
    
    def makeGauss(self):
        """Make the gaussian used to mask the image."""
        
        gsum = np.zeros((self.ny,self.nx))

        ng = len(self.xgauss)
        if ng == 0:
            self.applygauss = False
        else :
            self.applygauss = True

            if (self.slowGaussSum == 1):
                for i in np.arange(ng):
                    gtemp = mf.make_gaussian(self.ny,self.nx, rad=self.wgauss[i], norm=False, \
                                                 cenx=0, ceny=0)
                    gaussian = mf.array_shift( gtemp, self.xgauss[i], self.ygauss[i] )
                    gsum += gaussian
                    
                # gsum = normaliseInt(gsum)
                self.gauss = gsum / float(np.max(gsum))
            else:
                gtemp = mf.make_gaussian(self.ny,self.nx, rad=self.wgauss[0], norm=False, \
                                             cenx=0, ceny=0)
                for i in np.arange(ng):
                    gsum[self.xgauss[i], self.ygauss[i]] = 1.0
                
                self.gauss = np.abs(np.fft.ifft2( np.fft.fft2( gsum) * np.fft.fft2(gtemp) ))
                self.gauss *= 1.0/np.max(self.gauss)
                print "DEBUG:  max value of Guassian mask is :", np.max( self.gauss )


    def maskAutoc(self):
        """Make a mask for the cross terms in the autocorrelation function."""
        array = self.autoc

        # make a grey scale image of self.autoc
        array = greyScale(np.abs(array))

        # blur with a gaussian  : 2 
        array2 = ndimage.gaussian_filter(array,16)
        if 'mask_gaussian_blur' in self.output:
            self.output_image(np.abs(array2),outname='mask_gaussian_blur',pngColor=self.pngcolor)

        # apply an otsu threshold  : 3 
        thresh = np.max(np.abs(array2))*0.5
        array3 = np.array(1 * (np.abs(array2) > thresh),dtype=np.int)  
        T      = mahotas.thresholding.otsu(array2)
        array3 = 1 * (array > T)
        
        if 'mask_otsu' in self.output:
            self.output_image(np.abs(array3),outname='mask_otsu',pngColor=self.pngcolor)

        # cut out the centre of self.autoc 
        array4 = array3 * (1.0 - circle(self.autoc,radius=0.05))
        if 'mask_centre_cut' in self.output:
            self.output_image(np.abs(array4),outname='mask_centre_cut',pngColor=self.pngcolor)

        # expand the binary region : 4
        #array3 = sp.ndimage.morphology.binary_dilation(array3,iterations=20).astype(np.int)
        array4 = filterThreshFast(array4,blur=16)
                
        labeled,nr_objects = ndimage.label(array4.astype(float))

        self.mask = labeled
        
        if 'mask_labelled' in self.output:
            self.output_image(np.abs(labeled),outname='mask_labelled',pngColor=self.pngcolor)
        
        # once we have the mask we need to cut out all of the cross terms
        # except for the central bit (the sum of the autocs for the samples)
        N          = nr_objects 
        self.Ncross= N
        self.cross = np.zeros((N+1,self.ny,self.nx),dtype=np.complex128)
        
        ii = 0
        for i in range(N+1):
            self.cross[ii] = self.autoc * (labeled == i)
            self.cross[ii] = fft2(self.cross[ii])
            ii += 1

        # store the central part of the autocorrelation function as well
        array4 = array3 * circle(self.autoc,radius=0.05)
        array4 = filterThreshFast(array4,blur=16)
        labeled2, nr_objects = ndimage.label(array4.astype(float))
        self.central = self.autoc * (labeled2 == 1) 

   
    def makeCross(self):
        """Make the cross terms."""
        self.cross = np.zeros((self.Ncross+1,self.ny,self.nx),dtype=np.complex128)
        
        ii = 0
        for i in range(self.Ncross+1):
            self.cross[ii] = self.autoc * (self.mask == i)
            self.cross[ii] = fft2(self.cross[ii])
            ii += 1 

    def makePermutations(self):
        """Run through the permutations and identify the good combinations."""
        # I have the numbers 1 -> N where N is the number of cross terms
        # if N = 6 I want the combinations 
        # 1 23   2 13   3 12  3 13
        # 1 24   2 14   3 14  3 14
        # 1 25   2 15   3 15  3 15
        # 1 26   2 16   3 16  3 16
        # 1 27   2 17   3 17  3 17
        # 1 34   2 34   3 24  3 34
        # 1 35   2 35   3 25  3 35
        # 1 36   2 36   3 26  3 36
        # 1 37   2 37   3 27  3 37
        # 1 45   2 45   3 45  3 45
        # 1 46   2 46   3 46  3 46
        # 1 47   2 47   3 47  3 47
        # 1 56   2 56   3 56  3 56
        # 1 57   2 57   3 57  3 57
        # 1 67   2 67   3 67  3 67

        N = self.Ncross
        self.permutations = []
        for k in range(1,(N/2)+1):
            for i in range(1,N+1):
                if i != k:
                    for j in range(i+1,N+1):
                        if j != k:
                               self.permutations.append([k,i,j])
    
    def checkPermutations(self,fout=None):
        """identify the good combinations."""
        if (self.cross == None) or (self.permutations == None):
            print 'you need to run makePermutations.'

        if 'perm_err'in self.log_options:
            self.flog.write("\n")
            self.flog.write("Permutations of cross-terms: \n")
        if 'perm_err'in self.verbose:
            print "Permutations of cross-terms: "

        self.error = []
        perm2      = []
        for i in self.permutations:
            array = self.cross[i[1]]*self.cross[i[2]]/self.cross[i[0]]

            self.error.append(realPositive(array))
                        
            if fout != None: 
                if 'perm_err'in self.log_options:
                    self.flog.write(('%(i)i, %(j)i, %(k)i, %(err).3f \n' \
                    %{"i":i[1],"j":i[2],"k":i[0],"err":self.error[-1]}))
                if 'perm_err'in self.verbose:
                    print ('%(i)i, %(j)i, %(k)i, %(err).3f' \
                    %{"i":i[1],"j":i[2],"k":i[0],"err":self.error[-1]})

            if self.error[-1] < self.tol :
                perm2.append(i + [self.error[-1]])

        self.permutations = perm2
        if 'perm_err'in self.log_options:
            self.flog.write("\n")

        #
        #  Find good triplets by checking distances
        #

    def checkPermByDistance(self,fout=None):
        if (self.cross == None) or (self.permutations == None):
            print 'you need to run makePermutations.'

        perm2          = []
        self.error     = []
        self.errorDist = []

        nx, ny = self.cross[0].shape
        maskList = []
        for i in range(self.Ncross+1):
            maskList.append( np.where(self.mask == i) )

        distList = []
        for i in maskList:
            distList.append( ( np.sum( i[0] - nx/2 ) / len(i[0]) , \
                               np.sum( i[1] - ny/2 ) / len(i[1]) ) )


        if 'perm_err'in self.log_options:
                        self.flog.write("\n")
                        self.flog.write("Permutations of cross-terms (distance error): \n")
        if 'perm_err'in self.verbose:
                        print "Permutations of cross-terms  (distance error): "

        count = 0
        for i in self.permutations:
            dxy = ( (distList[i[0]][0] - distList[i[2]][0] - distList[i[1]][0] )**2, \
                    (distList[i[0]][1] - distList[i[2]][1] - distList[i[1]][1] )**2 )
            d = np.sqrt( dxy[0] + dxy[1])
            
            self.errorDist.append(d)
            if 'perm_err'in self.log_options:
                self.flog.write(('%(i)i, %(j)i, %(k)i, %(err).3f \n' \
                %{"i":i[1],"j":i[2],"k":i[0],"err":self.errorDist[-1]}))
            if 'perm_err'in self.verbose:
                print ('%(i)i, %(j)i, %(k)i, %(err).3f' \
                %{"i":i[1],"j":i[2],"k":i[0],"err":self.errorDist[-1]})

            if d < self.distTol:
                count += 1

                if self.divtol > 0. :
                    #
                    # Division with regularisation
                    #
                    cross0max = np.max(np.abs(self.cross[i[0]])**2)
                    array = np.conj(self.cross[i[0]])*self.cross[i[1]]*self.cross[i[2]]\
                                    /(np.abs(self.cross[i[0]])**2 + cross0max *self.divtol)
                else :
                    #
                    # simple division
                    #
                    array = self.cross[i[1]]*self.cross[i[2]]/self.cross[i[0]]

                self.error.append(realPositive(array)) 
                        
                if self.error[-1] < self.tol:
                    perm2.append(i + [self.error[-1]])

                if 'perm_err'in self.log_options:
                    self.flog.write(('Permutation found, error from imaginary part : %(i).3f \n' \
                    %{"i":self.error[-1]}))
                if 'perm_err'in self.verbose:
                    print ('Permutation found, error from imaginary part : %(i).3f \n' \
                    %{"i":self.error[-1]})

                if 'error_map' in self.output:
                    if self.gaussMask != 0 :
                        errorMap = self.gauss*(1. - self.realPositiveMap( array ))
                    else :
                        errorMap = (1. - self.realPositiveMap( array ))
                    outname  = 'errorMap'+('%(i)02d' %{'i':count} ) 
                    self.output_image(errorMap,outname=outname,\
                                      pngGamma=self.pngGamma,pngColor=self.pngcolor)
                                

        self.permutations = perm2
        if 'perm_err'in self.log_options:
            self.flog.write("\n")
        
    def makeDiffsDivMask(self):
        """From the good permutations make the diffs with simple divide."""
        N = 0
        for i in self.permutations:
            if i[3] < self.tol:
                N += 1
        #print 'number elements below the tol =',N
        self.diffs = np.zeros((N,self.ny,self.nx))
        self.Ndiffs = N

        self.error = np.zeros((N))
        self.diffIndex = []

        N = 0
        icount = 0
        for i in self.permutations:
            if i[3] < self.tol:
                print "a permutation: ", i
                self.flog.write(('a permutation: %(i)i, %(j)i, %(k)i, %(err).3f \n' \
                        %{"i":i[1],"j":i[2],"k":i[0],"err":i[3]}))

                

                if self.divtol > 0. :
                    #
                    # Division with regularisation
                    #
                    cross0max = np.max(np.abs(self.cross[i[0]])**2)
                    array = np.conj(self.cross[i[0]])*self.cross[i[1]]*self.cross[i[2]]\
                                    /(np.abs(self.cross[i[0]])**2 + cross0max *self.divtol)
                else :
                    #
                    # simple division
                    #
                    array = self.cross[i[1]]*self.cross[i[2]]/self.cross[i[0]]

                self.diffs[N,:,:] = np.abs(np.array(array.real)) 
                self.diffIndex.append(icount)
 
                N += 1

            icount += 1


    #
    # An alternative version for making the diffs, using the known good regions of the cross terms
    #

    def makeDiffsDivMask_GoodRegions(self):
        """From the good permutations make the diffs with simple divide."""
        N = 0
        for i in self.permutations:
            if i[3] < self.tol:
                N += 1
        #print 'number elements below the tol =',N
        self.diffs = np.zeros((N,self.ny,self.nx))
        self.Ndiffs = N

        self.error = np.zeros((N))
        self.diffIndex = []

        N = 0
        icount = 0
        for i in self.permutations:
            if i[3] < self.tol:
                print "a permutation: ", i
                self.flog.write(('a permutation: %(i)i, %(j)i, %(k)i, %(err).3f \n' \
                        %{"i":i[1],"j":i[2],"k":i[0],"err":i[3]}))

                array = np.zeros( (self.nx, self.nx), dtype=np.complex64)
                c0 = self.cross[i[0]]
                c1 = self.cross[i[1]]
                c2 = self.cross[i[2]]
              
                ic = np.where( self.cross_goodR[i[0]] > 0.0)
                
                
                if self.divtol > 0. :
                    #
                    # Division with regularisation
                    #
                    cross0max = np.max(np.abs(self.cross[i[0]])**2)
                    array[ic] = np.conj(c0[ic])  * c1[ic] * c2[ic]  \
                                    /(np.abs(c0[ic])**2 + cross0max *self.divtol)
                else :
                    #
                    # simple division
                    #
                    array[ic] = c1[ic] * c2[ic] / c0[ic]

                self.diffs[N,:,:] = np.abs(np.array(array.real)) 
                self.diffIndex.append(icount)
 
                N += 1

            icount += 1

        #
        # a second pass to find where we can use the other terms
        #
        
        for i in np.arange(self.Ndiffs):
            
            new_est = np.copy(np.abs(self.central))
            
            mask = np.zeros( (self.nx, self.nx) )
            icurr = np.where( self.diffs[i] < 1e-12 ) 
            mask[icurr] = 1.0
            
            for j in np.arange(self.Ndiffs):
                if i != j:
                    mask2 = np.zeros( (self.nx, self.nx) )
                    iblock = np.where( self.diffs[j] > 1e-12 ) 
                    mask2[iblock] = 1.0
                    mask *= mask2
                    
                    new_est += - self.diffs[j]
            
            im = np.where( mask == 1.0)
            t = self.diffs[i]
            t[im] = np.abs( new_est[im] )
            self.diffs[i] = np.copy(t)
            fname = 'second_pass_mask'+('%(i)02d' %{'i':i+1} )
            self.output_image( mask ,outname=fname,pngGamma=self.pngGamma,pngColor=self.pngcolor)
            print i, "mask max", np.max(mask)
            
    def makeDiffsQuadratically_verbose(self):
        """From the good permutations make the diffs by solving a quadratic set of equations.

        This routine makes no concessions whatsoever to memory efficiency or even to 
        computational efficiency. The aim is to make the steps as clear as possible."""

        # Let's assume that we have three particles with their corresponding 
        # two particle correlation functions (which have already been transformed to the far-field, except for the central bit)
        # C_nm = Psi_n Psi_m       : again these are far-field wave-functions  
        #
        # f = \sum |C_nn|^2 = \sum |psi_n|^2     : f is the sum of the desired diffraction patterns 
        #                                        : obtained from the central region of the autocorrelation function
        # g = |C_nm|^2 = |psi_n|^2 * |psi_m|^2   : for n != m , obtained from the mod square of the two particle correlation terms
        f      = np.abs(fft2(self.central))
        binary_out(f, self.outpath+'f'+'.raw')

        # I am just going to assume that we have three particles for now, 
        # and that permutations has already been purged of bad triplets
        print self.permutations
        i = self.permutations[2]
        print 'i = ', i
        N = len(self.permutations)
        g = np.zeros((N, N, self.ny, self.nx), dtype=np.float64)
        g[1,0] = np.abs(self.cross[i[1]])**2
        g[0,2] = np.abs(self.cross[i[2]])**2
        g[1,2] = np.abs(self.cross[i[0]])**2
        binary_out(np.abs(g[1,0]), self.outpath+'cross'+str(1)+'.raw')
        binary_out(np.abs(g[0,2]), self.outpath+'cross'+str(2)+'.raw')
        binary_out(np.abs(g[1,2]), self.outpath+'cross'+str(3)+'.raw')
        
        g[0,1] = g[1,0]
        g[2,0] = g[0,2]
        g[2,1] = g[1,2]
        #for n in range(N) :
        #    binary_out(np.abs(self.cross[i[n]]), self.outpath+'cross'+str(n)+'.raw')
        #    for m in range(N) :
        #        if n != m :
        #            g[n,m] = np.abs(self.cross[i[n]])**2
        #            
        #            binary_out(g[n,m], self.outpath+'g'+str(n)+str(m)+'.raw')

        # the solution to this set of equations is:
        #   I[n] = {f +- \sqrt( f^2 - 4 \sum_m g_nm ) } / 2.0  , for n != m
        # notice that there are two values per pixel per diffraction pattern
        # that means we have 2xN solutions and 2^N possible diffraction patterns
        # 0 --> +
        # 1 --> -
        Ipm = np.zeros((2, N, self.ny, self.nx), dtype=np.float64)
        for n in range(N) :
            array = f**2 - 4.0 * np.sum(g[n], axis=0)
             
            binary_out(4.0 * np.sum(g[n], axis=0), self.outpath+'4AC'+str(n)+'.raw')
            
            binary_out(array, self.outpath+'array'+str(n)+'.raw')
            
            array = array * (array >= 0.0)
            Ipm[0,n] = (f + np.sqrt(array)) / 2.0 
            Ipm[1,n] = (f - np.sqrt(array)) / 2.0 
            
            binary_out(Ipm[0,n], self.outpath+'Ipm0'+str(n)+'.raw')
            binary_out(Ipm[1,n], self.outpath+'Ipm1'+str(n)+'.raw')

        
        # Now I will make an Error array
        # this will tell us what error is accosiated with one of the eight combinations 
        # for a given pixel. In other words we will each combination and see how bad it is
        # So Error[perm[+,+,-]] = Error[1] = 
        #           (Ipm[0,0]Ipm[0,1] - g[0,1])^2 + (Ipm[0,0]Ipm[1,2] - g[0,2])^2 + (Ipm[0,1]Ipm[1,2] - g[1,2])^2
        # 
        # 0 +,+,+  --> perm[0,0] = 0, perm[0,1] = 0, perm[0,2] = 0
        # 1 +,+,-
        # 2 +,-,+
        # 3 +,-,-  --> perm[3,0] = 0, perm[3,1] = 1, perm[3,2] = 1
        # 4 -,+,+
        # 5 -,+,-
        # 6 -,-,+
        # 7 -,-,-

        # forgive my esoteric ways... 
        perm = np.zeros((2**N, N), dtype=np.int)
        print '2**N, N =', 2**N, N
        print 'permuations', perm
        for i in range(2**N):
            a = '00000000000000000000000' + Denary2Binary(i)
            for j in range(N):
                perm[i,j] = int(a[j - N ])

        print 'permuations', perm

        print 'Error:'
        y, x = (self.ny/2 + 10), (self.nx/2 + 10)
        Error = np.zeros((2**N, self.ny, self.nx), dtype = np.float64)
        array.fill(0.0)
        for i in range(len(perm)):
            for n in range(N) :
                array += Ipm[perm[i,n],n]
                for m in range(n+1, N) :
                    Error[i] += np.abs(g[n,m] - Ipm[perm[i,n],n]*Ipm[perm[i,m],m])**2 
                    print g[n,m,y,x],' - ',Ipm[perm[i,n],n,x,y],' * ', Ipm[perm[i,m],m,x,y], perm[i,n], perm[i,m]

            Error[i] += np.abs(f - array)**2
            array.fill(0.0)
            print i, Error[i,x,y]
            binary_out(Error[i], self.outpath+'Error'+str(i)+'.raw')
        
        # Now let us find the combinations which minimise the error
        # so now indexes = np.array( (self.ny, self.nx), dtype = np.int) ( numbers from 0 -> 7 )
        indexes = np.argmin(Error, axis = 0)

        binary_out(indexes, self.outpath+'indexes'+'.raw')
        
        # Let's get all of the diffs now 
        self.diffs  = np.zeros((N,self.ny,self.nx))
        self.Ndiffs = N

        # we want to choose the combinations per pixel from the indexes array
        # that tells us which combination (say + - + indicated by the number 2)
        # worked best at that pixel
        n = np.zeros((self.ny, self.nx), dtype=np.int)
        for j in range(N) :
            n.fill(j) 
            self.diffs[j]  = Ipm[0,j] * (perm[indexes, n] == 0)
            self.diffs[j] += Ipm[1,j] * (perm[indexes, n] == 1)
            
            binary_out(self.diffs[j], self.outpath+'diff'+str(j)+'.raw')
            
            binary_out(perm[indexes, n] == 0, self.outpath+'perm'+str(j)+'.raw')

    def makeDiffsQuadratically(self):
        """From the good permutations make the diffs by solving a quadratic set of equations.

        This routine makes no concessions whatsoever to memory efficiency or even to 
        computational efficiency. The aim is to make the steps as clear as possible."""

        # Let's assume that we have three particles with their corresponding 
        # two particle correlation functions (which have already been transformed to the far-field, except for the central bit)
        # C_nm = Psi_n Psi_m       : again these are far-field wave-functions  
        #
        # f = \sum |C_nn|^2 = \sum |psi_n|^2     : f is the sum of the desired diffraction patterns 
        #                                        : obtained from the central region of the autocorrelation function
        # g = |C_nm|^2 = |psi_n|^2 * |psi_m|^2   : for n != m , obtained from the mod square of the two particle correlation terms
        f      = np.abs(fft2(self.central))

        # I am just going to assume that we have three particles for now, 
        # and that permutations has already been purged of bad triplets
        i = self.permutations[2]
        N = len(self.permutations)
        g = np.zeros((N, N, self.ny, self.nx), dtype=np.float64)
        g[1,0] = np.abs(self.cross[i[1]])**2
        g[0,2] = np.abs(self.cross[i[2]])**2
        g[1,2] = np.abs(self.cross[i[0]])**2
        
        g[0,1] = g[1,0]
        g[2,0] = g[0,2]
        g[2,1] = g[1,2]
        #for n in range(N) :
        #    binary_out(np.abs(self.cross[i[n]]), self.outpath+'cross'+str(n)+'.raw')
        #    for m in range(N) :
        #        if n != m :
        #            g[n,m] = np.abs(self.cross[i[n]])**2
        #            
        #            binary_out(g[n,m], self.outpath+'g'+str(n)+str(m)+'.raw')

        # the solution to this set of equations is:
        #   I[n] = {f +- \sqrt( f^2 - 4 \sum_m g_nm ) } / 2.0  , for n != m
        # notice that there are two values per pixel per diffraction pattern
        # that means we have 2xN solutions and 2^N possible diffraction patterns
        # 0 --> +
        # 1 --> -
        Ipm = np.zeros((2, N, self.ny, self.nx), dtype=np.float64)
        for n in range(N) :
            array = f**2 - 4.0 * np.sum(g[n], axis=0)
            array = array * (array >= 0.0)

            Ipm[0,n] = (f + np.sqrt(array)) / 2.0 
            Ipm[1,n] = (f - np.sqrt(array)) / 2.0 
            
        
        # Now I will make an Error array
        # this will tell us what error is accosiated with one of the eight combinations 
        # for a given pixel. In other words we will each combination and see how bad it is
        # So Error[perm[+,+,-]] = Error[1] = 
        #           (Ipm[0,0]Ipm[0,1] - g[0,1])^2 + (Ipm[0,0]Ipm[1,2] - g[0,2])^2 + (Ipm[0,1]Ipm[1,2] - g[1,2])^2
        # 
        # 0 +,+,+  --> perm[0,0] = 0, perm[0,1] = 0, perm[0,2] = 0
        # 1 +,+,-
        # 2 +,-,+
        # 3 +,-,-  --> perm[3,0] = 0, perm[3,1] = 1, perm[3,2] = 1
        # 4 -,+,+
        # 5 -,+,-
        # 6 -,-,+
        # 7 -,-,-

        # forgive my esoteric ways... 
        perm = np.zeros((2**N, N), dtype=np.int)
        for i in range(2**N):
            a = '00000000000000000000000' + Denary2Binary(i)
            for j in range(N):
                perm[i,j] = int(a[j - N ])

        y, x = 585, 374
        Error = np.zeros((2**N, self.ny, self.nx), dtype = np.float64)
        array.fill(0.0)
        for i in range(len(perm)):
            for n in range(N) :
                array += Ipm[perm[i,n],n]
                for m in range(n+1, N) :
                    Error[i] += np.abs(g[n,m] - Ipm[perm[i,n],n]*Ipm[perm[i,m],m])**2 

            Error[i] += np.abs(f - array)**2 
            array.fill(0.0)
        
        # Now let us find the combinations which minimise the error
        # so now indexes = np.array( (self.ny, self.nx), dtype = np.int) ( numbers from 0 -> 7 )
        indexes = np.argmin(Error, axis = 0)

        # Let's get all of the diffs now 
        self.diffs  = np.zeros((N,self.ny,self.nx))
        self.Ndiffs = N

        # we want to choose the combinations per pixel from the indexes array
        # that tells us which combination (say + - + indicated by the number 2)
        # worked best at that pixel
        n = np.zeros((self.ny, self.nx), dtype=np.int)
        for j in range(N) :
            n.fill(j) 
            self.diffs[j]  = Ipm[0,j] * (perm[indexes, n] == 0)
            self.diffs[j] += Ipm[1,j] * (perm[indexes, n] == 1)

 
    # In A. Morgan's code this was known as project2Fast
    def project(self):
        """Given that array1+array2+array3=array123 find the least squared projection for the arrays.
        
        what if I wieghted the projection based on the divide by zero criteria?."""
        array = fft2(self.central)  #lowpass(self.image,self.lowpassrad)

        array00 = np.zeros((self.ny,self.nx))
        array11 = np.zeros((self.ny,self.nx))
        array22 = np.zeros((self.ny,self.nx))


        i = self.permutations[ self.diffIndex[0] ]

 
        t = np.abs( self.cross[ i[0]] )
        c = ( t < 0.02*np.max(t) ) 

        array00  =  self.diffs[0]*( 1 - c)
        array00 +=  (array - self.diffs[1] - self.diffs[2])*c # c01*c02 

        i = self.permutations[ self.diffIndex[1] ]
        c01 = ( np.abs(self.cross[ i[0] ] ) < np.abs(self.cross[ i[1] ]) ) | ( np.abs(self.cross[ i[0] ] ) < np.abs(self.cross[ i[2] ]) ) 
        c02 = ( np.abs(self.cross[ i[0] ] ) < np.abs(self.cross[ i[2] ]) )

        ic = np.where( ( np.abs(self.cross[ i[0] ] ) < np.abs(self.cross[ i[1] ]) ) \
                           & ( np.abs(self.cross[ i[0] ] ) < np.abs(self.cross[ i[2] ]) )  ) 

        array11  =  self.diffs[1]
        array11[ic] =  (array[ic] - self.diffs[0][ic] - self.diffs[2][ic]) 

        i = self.permutations[ self.diffIndex[2] ]
        c01 = ( np.abs(self.cross[ i[0] ] ) < np.abs(self.cross[ i[1] ]) )
        c02 = ( np.abs(self.cross[ i[0] ] ) < np.abs(self.cross[ i[2] ]) )
        array22  =  self.diffs[2]
        array22 +=  (- self.diffs[2] + array - self.diffs[0] - self.diffs[1])*c01*c02 

        self.diffs[0] = array00 ##lowpass(np.abs(array00),self.lowpassrad)
        self.diffs[1] = array11 ## lowpass(np.abs(array11),self.lowpassrad)
        self.diffs[2] = array22 ##lowpass(np.abs(array22),self.lowpassrad)


    def mask_from_statistical_threshold(self):
        #Define region to ignore
        IgnoreAreas = np.ones( self.image.shape )
            

        # Prepare autocorrelation function
            
        autoc  = np.abs(self.autoc)
        autoc  = autoc**0.3
            
        self.output_image(np.abs(autoc),outname='testautoc',\
                          pngGamma=self.pngGamma,pngColor=self.pngcolor)

                    
        #
        # apply multiple passes with statistical threshold
        #
        # start with all pixels
        isig  = np.where( np.abs(autoc) > -1.)
        print "size of mask", isig[0].size
        for thresh in self.autoc_mask_thresh:
            mean  = np.sum(autoc[isig]) / isig[0].size
            sigma = np.sqrt( np.sum( (autoc[isig] - mean)**2 ) / isig[0].size  )  
            isig  = np.where( autoc < (mean + thresh*sigma))
            print "mean, sigma, num_pix", mean, sigma, isig[0].size

        isig  = np.where( autoc > (mean + self.autoc_mask_thresh[-1]*sigma))
        #
        # Define the mask                
        #
            
        mask       = np.zeros(autoc.shape)
        mask[isig] = 1.
        mask      *= IgnoreAreas

        if 'mask_before_morphology' in self.output:
            self.output_image(np.abs(mask),outname='mask_before_morphology',pngColor=self.pngcolor)

        #
        # Apply opening and closing morphology
        #
        
        mostruct = mf.circle(2*self.rado,2*self.rado,rad=self.rado)
        mask     = ndimage.morphology.binary_opening(mask,structure=mostruct)       

        mcstruct = mf.circle(2*self.radc,2*self.radc,rad=self.radc)               
        mask     = ndimage.morphology.binary_closing(mask,structure=mcstruct)

        if 'mask_post_morphology' in self.output:
            self.output_image(np.abs(mask),outname='mask_post_morphology',pngColor=self.pngcolor)
            


        #
        # remove the central area (because it is not a cross-correlation term)
        #
        labeled, N   = ndimage.label(mask)

        mid          = labeled[mask.shape[0]/2, mask.shape[1]/2]
        
        maskCC       = mask * (labeled != mid)
        self.central = self.autoc *(labeled == mid )
        self.cmask = np.ones( (self.ny, self.nx) )*(labeled == mid)

        if self.autocCentralMaskname != None:
            self.central *= self.autocCentralMask

        self.output_image(np.abs(self.central), outname='central_part', pngColor=self.pngcolor)


        #
        # Apply dilation
        #
        mdstruct = mf.circle(2*self.radd,2*self.radd,rad=self.radd)               
        maskCC = ndimage.morphology.binary_dilation(maskCC,structure=mdstruct)
        if 'mask_dilation' in self.output:
            self.output_image(np.abs(maskCC),outname='mask_post_morphology',pngColor=self.pngcolor)


        #
        # get mask for CCs only
        #
        
        labeledCC, N = ndimage.label(maskCC)
        self.mask = labeledCC              
     

        print "Number of CCs found", N
        if 'mask_labelled' in self.output:
            self.output_image(np.abs(self.mask),outname='mask_labelled_stat',pngColor=self.pngcolor)

        if N > self.maxCCs:
            # print "Maximum number of cross terms exceeded"
            sys.exit("Maximum number of cross terms exceeded")

        #
        # once we have the mask we need to cut out all of the cross terms
        # except for the central bit (the sum of the autocs for the samples)
        #
     
        self.Ncross= N
        self.cross = np.zeros((N+1,self.ny,self.nx),dtype=np.complex128)
        
        self.updateCross()


        #
        # Make a noise estimate from the autocorrelation function
        #
        im = np.where( mask == 0 )
        asig = np.sqrt( np.sum( np.abs(self.autoc[im])**2 ) / len(im[0]) )
        print "asig", asig

        # test!
        testm = np.ones( (self.nx, self.nx) )
        ftest = fft2( testm )
        print "ftest", np.max(np.abs(ftest))

        #
        # if a mask is set then we can determine the noise level
        # for each cross-term...
        #
        self.crossnoiseEst = np.zeros( (N+1,2) )
        self.cross_goodR = np.zeros( (N+1,self.ny,self.nx) )
        if self.gauss != None:

            iin = np.where( self.gauss > np.exp(-1) )

            for i in np.arange(N+1):
                
                fact = float(len(iin[0])) / float(self.nx**2)
                n = np.sum( np.ones((self.nx, self.nx))*(self.mask == i))
                th = 3.0 * np.sqrt(n) * asig / np.sqrt(fact)
                print "i, asig, fact, sq(n), th", i, asig, 1/fact, np.sqrt(n), th 

                ic = np.where( np.abs(self.cross[i]) > th )
                temp = np.zeros( (self.nx, self.nx) )
                temp[ic] = 1.0
                self.cross_goodR[i] = np.copy(temp)

                fname = 'fcross'+('%(i)02d' %{'i':i} )
                self.output_image( np.abs(np.copy(self.cross[i])) \
                                       ,outname=fname,pngGamma=self.pngGamma,pngColor=self.pngcolor)


                if 'good_regions' in self.output:
                    fname = 'good_region'+('%(i)02d' %{'i':i} )
                    self.output_image( self.cross_goodR[i],\
                                           outname=fname,pngGamma=self.pngGamma,\
                                           pngColor=self.pngcolor)
          
                    Rsum = np.zeros( (self.nx, self.nx) )
                    for i in np.arange(N)+1:
                        Rsum += self.cross_goodR[i]
                        fname = 'good_region_sum'
                    self.output_image( Rsum,\
                                           outname=fname,pngGamma=self.pngGamma,\
                                           pngColor=self.pngcolor)
          


    def updateCross(self):
        ii = 0
        for i in np.arange(self.Ncross+1):
            self.cross[ii] = self.autoc * (self.mask == i)
            self.cross[ii] = fft2(self.cross[ii])
            print i, np.max(np.abs(self.cross[ii])), np.min(np.abs(self.cross[ii]))
            ii += 1
  

        
            



    def output_image(self,image,outname='image',pngGamma=1.0,pngColor='ocean'):
        if "raw" in self.output_formats:
            if 'write'in self.log_options:
                self.flog.write("Writing to file: "+self.outpath+outname+'.raw\n')
            if 'write'in self.verbose:
                print "Writing to file: "+self.outpath+outname+'.raw'

            imageJoutRaw(image,self.outpath+outname+'.raw')
                
        if "h5" in self.output_formats:
            if 'write'in self.log_options:
                self.flog.write("Writing to file: "+self.outpath+outname+'.h5\n')
            if 'write'in self.verbose:
                print "Writing to file: "+self.outpath+outname+'.h5'

            mf.h5write(os.path.normpath(self.outpath+outname+'.h5'),image,field=self.h5field)

        if "png" in self.output_formats:
            if 'write'in self.log_options:
                self.flog.write("Writing to file: "+self.outpath+outname+'.png\n')
            if 'write'in self.verbose:
                print "Writing to file: "+self.outpath+outname+'.png'

            plt.imsave(os.path.normpath(self.outpath+outname+'.png'),np.abs(image)**pngGamma,format='png',\
                               cmap=pngColor)
                        

    #
    # Use the central part of the autocorrelation function to check
    # the accuracy of the recovered single particle diffraction patterns.
    #
    
    def checkCentralCC(self):
        estimate = self.central*0.
        for i in np.arange(len(self.diffs)):
            estimate += ifft2( self.diffs[i] )  

        test = ifft2( self.diffs[0] )
        self.estimate = estimate

        if 'central_estimate' in self.output:
            self.output_image(np.abs(estimate)**self.pngGamma, outname='central_estimate', pngColor=self.pngcolor)
            self.output_image(np.abs(test)**self.pngGamma, outname='central_diffs0', pngColor=self.pngcolor)

        print "max central, max estimate, totals", np.max(np.abs(self.central)), np.max(np.abs(self.estimate)), \
                              np.sum(np.abs(self.central)), np.sum(np.abs(self.estimate))

        error = np.sum(  np.abs(self.central - estimate)**2 ) / np.sum( np.abs(self.central)**2 )
        error = np.sqrt(error)
        self.centralError = np.abs(error)
        if 'errors' in self.verbose:
            if self.project_flag == 0:
                print "Error checked with central part :", self.centralError
            elif self.project_flag == 1:
                print "Cannot use central part to check error when central part "\
                      +"is used for regularisation (project = 1)"

        if 'errors' in self.log_options:
            if self.project_flag == 0:
                self.flog.write( ("Error checked with central part : %(i)e \n " \
                            %{"i":self.centralError}) )
            elif self.project_flag == 1:
                self.flog.write("Cannot use central part to check error when central part "\
                          +"is used for regularisation (project = 1)\n")

    def addPoissonNoise(self):
        norm = np.sum(self.image)
        print "norm", norm, self.noise_flag
        self.image *= float(self.noise_flag) / norm
        self.inputImage = np.copy(self.image)

        # save the noise levels
        temp = np.sqrt(self.image)
        
        # add the noise
        self.image = np.random.poisson(self.image)
            
        self.Wiener_noisemap = temp


        # for m in np.arange(self.image.size):
        #         i = m % self.image.shape[0]
        #         j = m / self.image.shape[0]
        #         if self.image[i][j]>= 0:
        #                 self.image[i][j] = np.random.poisson(self.image[i][j],1)

    def calculateNoiseMaps(self):
        
        self.autoc_noise = np.fft.fft2( self.Wiener_noisemap**2 )
        self.WienerNoiseDist = np.zeros((self.Ndiffs,self.ny,self.nx))

        j = 0
        for i in self.permutations :
            temp = np.abs( np.fft.ifft2( self.autoc_noise*(self.mask == i[0]) ) )
            self.WienerNoiseDist[j] = np.sqrt(temp) 
            print "wiener noise dist", j, np.max(self.WienerNoiseDist[j]), np.min(self.WienerNoiseDist[j])
            c = self.nx/5
            th = np.sum( self.WienerNoiseDist[j][0:c,0:c]) / float(c**2)
            print "Wiener sum", th / 2, "this sets Wiener threshold"
            self.WienerThresh = th / 2
            
            j += 1

    def calculateNoiseMaps_forAutoc(self):
        
        self.autoc_noise = ifft2( self.Wiener_noisemap**2 )
        self.WienerNoiseDist_forAutoc = np.zeros((self.Ndiffs,self.ny,self.nx))
        self.WienerNoiseDist2 = np.zeros((self.Ndiffs,self.ny,self.nx))

        reg = 0.01

        j = 0
        for i in self.permutations :
            temp0 = np.abs( fft2( self.autoc_noise*(self.mask == i[0]) ) )
            temp1 = np.abs( fft2( self.autoc_noise*(self.mask == i[1]) ) )
            temp2 = np.abs( fft2( self.autoc_noise*(self.mask == i[2]) ) )
            
            combined =   (temp0 / (np.abs(self.cross[i[0]]) + reg) )**2 \
                       + (temp1 / (np.abs(self.cross[i[1]]) + reg) )**2 \
                       + (temp2 / (np.abs(self.cross[i[2]]) + reg) )**2

            combined *= np.abs( self.cross[i[1]] * self.cross[i[2]] / (self.cross[i[0]] + reg) )**2
            
            fcom = np.abs( ifft2( combined ) )

            self.WienerNoiseDist2[j] = np.sqrt( combined )
            
            self.WienerNoiseDist_forAutoc[j] = np.sqrt( fcom ) 
            print "wiener noise dist", j, np.max(self.WienerNoiseDist_forAutoc[j]), np.min(self.WienerNoiseDist_forAutoc[j])
         
            j += 1



    def realPositiveMap(self,array):
        """Calculate how real and positive an array is.
        
        sqrt[|array - RP(array)|^2]/sqrt[|array|^2]
        R - real part
        P - positive part
        """
        arrayRP = np.array(array.real,dtype=np.complex128)
        arrayRP = arrayRP * (arrayRP.real > 0.0)
        arrayRP = array - arrayRP

        errorMap = np.abs(arrayRP) / np.abs(array)
        return errorMap                

    def lowPassFilterDiffs(self):
            
        if 'central_estimate' in self.output:
            test = ifft2(self.diffs[0]) * circle(self.diffs[0],radius=self.lowpassrad)
            test = fft2(test)
            self.output_image(np.abs(test)**self.pngGamma, outname='central_diffs0_lowpass', pngColor=self.pngcolor)
       

        for j in range(self.Ndiffs) :
            self.diffs[j]  = lowpass(self.diffs[j],self.lowpassrad) 
            
    def cleanDiffsGauss(self):
        
        for j in range(self.Ndiffs) :
            self.diffs[j]  *= (self.gauss > 1e-2*np.max(self.gauss) ) 
            #self.diffs[j]  *= self.gauss 
        
    def test_noise_level(self):

        t = np.zeros( (self.nx,self.nx) )
        t = np.random.rand( self.nx, self.nx)
        t *= self.gauss
        ft = fft2( t )
        ft2 = ft * (self.mask == 1)
        t2 = ifft2(ft2)
        self.output_image(np.abs(t2), outname='test_noise', pngColor=self.pngcolor)
        

    #
    # Calculates noise level by randomly placing the mask
    # for cross term j
    #
    def calculate_cross_noise(self,j, cmask):

      
        #
        # A sum over lots of different noise calculations
        #
        noise_mean_map = np.zeros( cmask.shape, dtype=np.complex64 )
        noise_sig_map  = np.zeros( cmask.shape, dtype=np.float64 )  
        npos = 50

        for ipos in np.arange(npos):
            
            #
            # Verbose update of the noise calculation
            #
            if ((ipos+1) % 100) == 0:
                print "making noise map", ipos+1, "/", npos

            #
            # Shift mask randomly (max 150 times) to find a region of autoc to use
            #
            shift_found = 0
            for ish in np.arange(150):

                shifts = np.random.rand(2)*self.nx

                shifted = mf.array_shift(cmask, int(shifts[0]), int(shifts[1]))
                #print "shift attempt", ish, " error", np.sum(shifted*self.mask)
                
                if np.sum(shifted*(self.mask + self.cmask))==0:
                    shift_found = 1
                    break

            if shift_found == 0:
                print "No region was found to calculate the noise level", j, "/", self.Ndiffs

            #
            # Calculate the noise level in Fourier space and store it in a global variable
            #
            # noise = mf.array_shift(np.fft.fft2( self.autoc*shifted ), self.nx/2, self.nx/2 )
            noise = fft2( self.autoc*shifted )
              
            


            #
            # OBSELETE : standard deviation of a single noise map;
            #            from my first attempt
            #
            # noise_sig = np.sqrt( np.sum( np.abs(noise*self.gauss)**2 ) / np.sum(self.gauss) )

            #
            # Add the current noise estimate to the sum
            #
            noise_mean_map += noise
            noise_sig_map += np.abs(noise)**2
      
        #
        #  calculate average noise per pixel and standard deviation
        #
        noise_mean_map *= 1./float(npos)
        noise_sig_map = np.sqrt( (noise_sig_map/float(npos)) - np.abs(noise_mean_map)**2 )


        #
        # OBSELETE: calculation of radial average to improve noise estimation
        #           mapping the radial plot back into the 2d grid is broken
        #
        #
        # Maybe here we can calculate the radially averaged noise / noise sig
        # And maybe expand it back into a 2D array to make it easy to calculate the next bit
        #
        #noise_polar, noise_rav, noise_rdata =\
        #    mf.image_radial( np.real(noise), -self.nx/2, -self.nx/2,\
        #                         rbins=self.nx/8, method='linear')
        #noise_sym = mf.rav_to_image(noise_rav, noise_rdata)

        #p, = plt.plot( noise_rav )
        #outname = self.outpath+('rav_plot%(j)i.png'  %{"j":j})
        #plt.draw()
        #plt.savefig( outname )

        #noise2_polar, noise2_rav, noise2_rdata =\
        #    mf.image_radial( np.abs(noise)**2, -self.nx/2, -self.nx/2,\
        #                         rbins=self.nx/8, method='linear')
        #noise2_sym = mf.rav_to_image(noise2_rav, noise2_rdata)
#
#        sig_sim = np.sqrt( noise2_sym - noise_sym**2 )

        #return noise, noise_sig, sig_sim
        
        return noise_mean_map, noise_sig_map
                              



    def mask_artifacts(self):

        #
        # some lists for storing things
        #
        artmask_list  = []
        masked_list   = []
        wfilter_list  = []
        filtered_list = []

        artmask_prod = np.ones((self.ny,self.nx))
        wfilter_prod = np.ones((self.ny,self.nx))
        diffs_sum = np.zeros((self.ny,self.nx))
        filtered_sum = np.zeros((self.ny,self.nx))


        #
        # loop over the recovered diffraction patterns
        #
        for j in np.arange(self.Ndiffs) :
       
            #
            # Identify the cross-term used in the denominator
            # and create a mask
            #
            i = self.permutations[ self.diffIndex[j] ]
        
            cmask0 = np.ones((self.ny,self.nx)) * (self.mask == i[0])
            noise_mean0, noise_sig0 = self.calculate_cross_noise(j, cmask0)
          
            cmask1 = np.ones((self.ny,self.nx)) * (self.mask == i[1])
            noise_mean1, noise_sig1 = self.calculate_cross_noise(j, cmask1)

            cmask2 = np.ones((self.ny,self.nx)) * (self.mask == i[2])
            noise_mean2, noise_sig2 = self.calculate_cross_noise(j, cmask2)
       

            #
            # Analytic noise estimates
            #
            autocmask0 = np.abs(ifft2( np.abs( fft2(cmask0) )**2 ))/float(self.nx*self.nx)
            noise_sig_ana0 = np.sqrt( np.abs(fft2( ifft2(self.inputImage*self.gauss)*autocmask0 )) )
            autocmask1 = np.abs(ifft2( np.abs( fft2(cmask1) )**2 ))/float(self.nx*self.nx)
            noise_sig_ana1 = np.sqrt( np.abs(fft2( ifft2(self.inputImage*self.gauss)*autocmask1 )) )
            autocmask2 = np.abs(ifft2( np.abs( fft2(cmask2) )**2 ))/float(self.nx*self.nx)
            noise_sig_ana2 = np.sqrt( np.abs(fft2( ifft2(self.inputImage*self.gauss)*autocmask2 )) )

            print "max noise sig", np.max(noise_sig0), np.max(noise_sig_ana0)
            print "max autocmask", np.max(autocmask0), np.max(autocmask1)
            print "searching for normalisation", np.sum(cmask0*np.abs(self.autoc)**2 ),\
                np.sum(np.abs(self.autoc)**2)
            print "normalization test", np.sum(cmask0**2), np.sum( np.abs(fft2(cmask0))**2 )

            # print "Check noise stats:", j
            # print noise_mean0, noise_sig0
            # print noise_mean1, noise_sig1
            # print noise_mean2, noise_sig2

            #
            # Define the threshold for the Tikhonov filter/mask
            #
            threshold     = self.artifact_noise_tol*noise_sig0

            #
            # Use noise level to create aritfact_mask
            #   
            amask = cmask0 * 0.0
            ifc1 = np.where( np.abs(self.cross[i[0]]*self.gauss) > threshold )      
            amask[ifc1] = 1.0
            masked_diff = self.diffs[j] * amask

            #
            # Use the noise level to apply a Tikhonov filter.
            #
            threshold     = noise_sig0
            filtered = np.real(\
                np.conjugate(self.cross[i[0]])*\
                    self.cross[i[1]]*self.cross[i[2]]\
                     /(np.abs(self.cross[i[0]])**2 + threshold**2 )  )
                                                                          
            #
            # Wiener filter using noise level from c1*c2
            #
            c12g_abs = np.abs(self.cross[i[1]]*self.cross[i[2]]*self.gauss)

            c12_noise_map = np.sqrt( (np.abs(self.cross[i[2]])*noise_sig1)**2\
                                + (np.abs(self.cross[i[1]])*noise_sig2)**2 )
           
            Wiener_noise = c12_noise_map

            iabove = np.where( c12g_abs > Wiener_noise )

            signal = self.diffs[j] * 0.0
            signal[iabove] = c12g_abs[iabove] - Wiener_noise[iabove]

            wfilter = (signal**2) / (signal**2 + Wiener_noise**2)
            
            filtered *= wfilter

            #
            #  Overall error map
            #
            csq123 = self.gauss*np.abs(self.cross[i[0]]*self.cross[i[1]]*self.cross[i[2]])**2
            denom = np.abs(noise_sig0*self.cross[i[1]]*self.cross[i[2]])**2\
                + np.abs(noise_sig1*self.cross[i[0]]*self.cross[i[2]])**2\
                + np.abs(noise_sig2*self.cross[i[1]]*self.cross[i[0]])**2
            signal_to_noise = csq123 / np.sqrt(denom)


            #
            # Output mask, the masked data, write noise level to file.
            #

            if 'artifact_mask' in self.output:
                self.output_image(amask,\
                                      outname=('artifact_mask%(j)i'\
                                                   %{"j":j}),\
                                      pngColor=self.pngcolor,pngGamma=self.pngGamma)


            if 'error_map_denom' in self.output:
                self.output_image(np.abs(noise_sig0),\
                                      outname=('error_map_denominator%(j)i'\
                                                   %{"j":j}),\
                                      pngColor=self.pngcolor,pngGamma=self.pngGamma)

                self.output_image(np.abs(noise_sig_ana0),\
                                      outname=('error_map_denominator_analytic%(j)i'\
                                                   %{"j":j}),\
                                      pngColor=self.pngcolor,pngGamma=self.pngGamma)

            if 'diff_masked_artifacts' in self.output:
                self.output_image(np.abs(masked_diff),\
                                      outname=('diffs%(j)i_artifacts_masked'\
                                                   %{"j":j}),\
                                      pngColor=self.pngcolor,pngGamma=self.pngGamma)


            if 'diff_filtered' in self.output:
                self.output_image(np.abs(filtered),\
                                      outname=('diffs%(j)i_Tikh_Wiener_filtered'\
                                                   %{"j":j}),\
                                      pngColor=self.pngcolor,pngGamma=self.pngGamma)

            if 'amask_times_wfilter' in self.output:
                self.output_image(amask*wfilter,\
                                      outname=('artifact_mask_times_Wiener%(j)i'\
                                                   %{"j":j}),\
                                      pngColor=self.pngcolor,pngGamma=self.pngGamma)

            if 'signal_to_noise' in self.output:
                self.output_image(signal_to_noise,\
                                      outname=('signal_to_noise%(j)i'\
                                                   %{"j":j}),\
                                      pngColor=self.pngcolor,pngGamma=self.pngGamma)


            #
            # Make a histogram and calculate the lowest noise level...
            #
            hmax  = np.max( self.inputImage )
            hmin  = 0
            hbins = long( hmax - hmin )*10

            if hbins > 1:

                print "bins hmax hmin", hbins, hmax, hmin

                hist,  bin_edges  = np.histogram( amask*self.inputImage*self.gauss,\
                                                      bins=hbins, range=(hmin,hmax) )
                hist2, bin_edges2 = np.histogram( np.abs(1.0-amask)*self.inputImage*self.gauss,\
                                                      bins=hbins, range=(hmin,hmax) )
     
                print "bin_edges"
                print bin_edges
                print "hist"
                print hist
                print "hist2"
                print hist2

                hratio = hist*1.0 / (1.0*hist + 1.0*hist2)

                print "hratio"
                print hratio[0:30]

                plt.figure()
                hplot, = plt.plot( bin_edges[:-1], hratio )
                plt.axis([0, 10, 0.0, 1.0])
                plt.draw()
                plt.xlabel("intensity")
                plt.ylabel("percentage inside mask")
                if 'histogram_test' in self.output:
                    plt.savefig(self.outpath+'hratio_plot%(j)i.png'%{"j":j})


            #
            # Append the masks and filtered results to lists
            #
            artmask_list.append(amask)
            masked_list.append(masked_diff)
            wfilter_list.append(wfilter)
            filtered_list.append(filtered)

            #
            # Products and sum
            #
            artmask_prod *= amask
            wfilter_prod *= wfilter
            diffs_sum    += self.diffs[j]
            filtered_sum += filtered


        #
        # Output the products of the masks
        #
        if 'amask_product' in self.output:
            self.output_image(np.abs(artmask_prod),\
                                  outname=('artifacts_masks_product'\
                                               %{"j":j}),\
                                  pngColor=self.pngcolor,pngGamma=self.pngGamma)

        if 'wfilter_product' in self.output:
            self.output_image(np.abs(wfilter_prod),\
                                  outname=('wfilter_masks_product'\
                                               %{"j":j}),\
                                  pngColor=self.pngcolor,pngGamma=self.pngGamma)

            

        #
        #  Calculate errors by comparing to the central part
        #
        fcen = np.abs(fft2(self.central))
       
        error_unmasked =  np.sum( (fcen-diffs_sum)**2 )\
            /np.sqrt(np.sum( abs(fcen)**2 )*np.sum(abs(diffs_sum)**2))

        error_masked =  np.sum( artmask_prod*(fcen-diffs_sum)**2 )\
            /np.sqrt(np.sum( artmask_prod*abs(fcen)**2 )\
                         *np.sum(artmask_prod*abs(diffs_sum)**2))
       
        error_wfilter =  np.sum( wfilter_prod*(fcen-diffs_sum)**2 )\
            /np.sqrt(np.sum( wfilter_prod*abs(fcen)**2 )\
                         *np.sum(wfilter_prod*abs(diffs_sum)**2))
       
        print "error unmasked :", error_unmasked
        print "error masked :"  , error_masked
        print "error wfilter  :", error_wfilter     

        self.flog.write("central (fourier) error unmasked : "+('%(e)g\n' %{"e":error_unmasked})  )
        self.flog.write("central (fourier) error masked : "+('%(e)g\n' %{"e":error_masked})  )
        self.flog.write("central (fourier) error wfilter : "+('%(e)g\n' %{"e":error_wfilter})  )


        


#######################################################
#
# Class that generates valid scan points
#
#######################################################
class scan_regions :
    def __init__(self):
        self.xstart = []
        self.xend   = []
        self.ystart = []
        self.yend   = []
        self.xstep  = []
        self.ystep  = []
        self.width  = []
    
    def check_list_sizes(self):
        sizes = []
        sizes.append( len(self.xstart) )
        sizes.append( len(self.xend) )
        sizes.append( len(self.ystart) )
        sizes.append( len(self.yend) )
        sizes.append( len(self.xstep) )
        sizes.append( len(self.ystep) )
        sizes.append( len(self.width) )

        lmin = min(sizes)
        lmax = max(sizes)
        if lmin != lmax :
                print "Parameter values missing for some scan regions - check config file"
                print "min : ", lmin, "  max :", lmax
                print "Number of scan regions used: ", lmin

        return lmin, lmax



