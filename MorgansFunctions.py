

#########################################################
# Functions originally from
# CrossTermsMod.py by A. Morgan 09/2011
# 
# is imported by crossTermsTools.py
# A. Martin 10/2011 
#########################################################

import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import ndimage
from matplotlib.colors import LogNorm
import mahotas
from scipy import fftpack
import pymorph
from random import *
import scipy as sp
import random
import time
import sys

def Denary2Binary(n):
    '''convert denary integer n to binary string bStr'''
    bStr = ''
    if n < 0:  raise ValueError, "must be a positive integer"
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr

def normaliseInt(array,tot=1.0):
	"""normalise the array to tot.

	normalises such that sum arrayout = tot.
	"""
	tot1 = np.sum(array)
	arrayout = array * tot / tot1
	return arrayout

def imageJinRaw(fnam,ny,nx,dt=np.dtype(np.float64),endianness='big'):
    """Read a 2-d array from a binary file."""
    arrayout = np.fromfile(fnam,dtype=dt).reshape( (ny,nx) )
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout = np.float64(arrayout)
    return arrayout

def imageJoutRaw(array,fnam,dt=np.dtype(np.float64),endianness='big'):
    """Write a 2-d array to a binary file."""
    arrayout = np.array(array,dtype=dt)
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout.tofile(fnam)

def binary_in(fnam,ny,nx,dt=np.dtype(np.float64),endianness='big'):
    """Read a 2-d array from a binary file."""
    arrayout = np.fromfile(fnam,dtype=dt).reshape( (ny,nx) )
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout = np.float64(arrayout)
    return arrayout

def binary_out(array,fnam,dt=np.dtype(np.float64),endianness='big'):
    """Write a 2-d array to a binary file."""
    arrayout = np.array(array,dtype=dt)
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    arrayout.tofile(fnam)

def roll(arrayin,dy = 0,dx = 0):
	"""np.roll arrayin by dy in dim 0 and dx in dim 1."""
	if (dy != 0) or (dx != 0):
		arrayout = np.roll(arrayin,dy,0)
		arrayout = np.roll(arrayout,dx,1)
	else:
		arrayout = arrayin
	return arrayout

def circle(arrayin,radius=0.5):
	"""Make a circle of optional radius as a fraction of the array size"""
	ny = arrayin.shape[0]
	nx = arrayin.shape[1]
	nrad     = (ny * radius)**2
	arrayout = np.zeros((ny,nx))
	for i in range(0,ny):
		for j in range(0,nx):
			r = (i - ny/2)**2 + (j - nx/2)**2
			if r < nrad:
				arrayout[i][j] = 1.0
	return arrayout

def fft2(arrayin):
	"""Calculate the 2d fourier transform of an array with N/2 as the zero-pixel."""
	# do an fft
	arrayout = np.array(arrayin,dtype=complex)
	arrayout = fftpack.ifftshift(arrayout)
	arrayout = fftpack.fft2(arrayout)
	arrayout = fftpack.fftshift(arrayout)
	return arrayout

def ifft2(arrayin):
	"""Calculate the 2d inverse fourier transform of an array with N/2 as the zero-pixel."""
	# do an fft
	arrayout = np.array(arrayin,dtype=complex)
	arrayout = fftpack.fftshift(arrayout)
	arrayout = fftpack.ifft2(arrayout)
	arrayout = fftpack.ifftshift(arrayout)
	return arrayout

def gauss(arrayin,a,ryc=0.0,rxc=0.0): 
	"""Return a real gaussian as an numpy array e^{-a x^2}."""
	ny = arrayin.shape[0]
	nx = arrayin.shape[1]
	# ryc and rxc are the coordinates of the center of the gaussian
	# in fractional unints. so ryc = 1 rxc = 1 puts the centre at the 
	# bottom right and -1 -1 puts the centre at the top left
	shifty = int(ryc * ny//2)
	shiftx = int(rxc * nx//2)
	arrayout = np.zeros((ny,nx))
	for i in range(0,ny):
		for j in range(0,nx):
			x = np.exp(-a*((i-ny/2)**2 + (j-nx/2)**2))
			arrayout[i][j] = x

	if ryc != 0.0 :
		arrayout = np.roll(arrayout,shifty,0)
	if rxc != 0.0:
		arrayout = np.roll(arrayout,shiftx,1)

	return arrayout

def greyScale(arrayin):
	"""Convert arrayin to uint16 by scaling the image."""
	arrayout = arrayin - np.min(arrayin)
	arrayout = arrayout * 2**16 / np.max(arrayin)
	arrayout = np.array(arrayout,dtype=np.uint16)
	return arrayout

def greyScale256(arrayin):
	"""Convert arrayin to uint8 by scaling the image."""
	arrayout = arrayin - np.min(arrayin)
	arrayout = arrayout * 2**8 / np.max(arrayin)
	arrayout = np.array(arrayout,dtype=np.uint8)
	return arrayout

def draw(arrayin):
	array = np.abs(arrayin)
	plt.clf()
	plt.ion()
	plt.imshow(array) #,cmap='Greys_r')
	plt.axis('off')
	plt.draw()

def drawP(arrayin):
	"""Draw arrayin and promt to press enter."""
	array = np.abs(arrayin)
	plt.clf()
	plt.ion()
	plt.imshow(array) #,cmap='Greys_r')
	plt.axis('off')
	plt.draw()
	raw_input('press ENTER to continue...')

def realPositive(array):
	"""Calculate how real and positive an array is.
	
	sqrt[|array - RP(array)|^2]/sqrt[|array|^2]
	R - real part
	P - positive part
	"""
	arrayRP = np.array(array.real,dtype=np.complex128)
	arrayRP = arrayRP * (arrayRP.real > 0.0)
	arrayRP = array - arrayRP
	error = np.sum(arrayRP * np.conj(arrayRP))/np.sum(array*np.conj(array))
	error = np.sqrt(error)
	return error.real

def filterThreshFast(array,blur=8):
	"""Apply a gausian blur then return thresholded array."""
	arrayout = np.array(array,dtype=np.float)
	arrayout = ndimage.gaussian_filter(arrayout,blur)

	thresh = np.max(np.abs(arrayout))*0.1
	arrayout = np.array(1.0 * (np.abs(arrayout) > thresh),dtype=array.dtype)  
	return arrayout

def complexHist(array):
	"""Display the points (array) on a real and imaginary axis."""
	from matplotlib.ticker import NullFormatter
	# scale the amplitudes to 0->1024
	arrayAmp = np.abs(array)/np.max(np.abs(array)) 
	#arrayAmp = arrayAmp - np.min(arrayAmp)
	#arrayAmp = arrayAmp / np.max(arrayAmp)
	arrayAmp = 1000.0*arrayAmp/(1000.0*arrayAmp + 1)
	array2   = arrayAmp * np.exp(-1.0J * np.angle(array))
	x = []
	y = []

	for i in range(1000):
		i = random.randrange(0,array.shape[1])
		j = random.randrange(0,array.shape[0])
		x.append(array2.real[i,j])
		y.append(array2.imag[i,j])

	plt.clf()
	plt.ion()
	rect_scatter = [0.0,0.0,1.0,1.0]
	axScatter = plt.axes(rect_scatter)
	axScatter.scatter(x,y,s=1,c='grey',marker='o')
	axScatter.set_xlim((-1.0,1.0))
	axScatter.set_ylim((-1.0,1.0))
	#plt.plot(x,y,'k,')
	plt.draw()

def lowpass(array,rad=0.1):
	"""low pass filter with a circle of radius rad."""
	arrayout = np.array(array,dtype=np.complex128)
	arrayout = ifft2(arrayout) * circle(arrayout,radius=rad)
	arrayout = fft2(arrayout)
	return np.abs(arrayout)

def seedCircles(arrayin,rad=0.04):
	"""Take a binary image and put circles around the 1's."""
	circ     = np.zeros(arrayin.shape,dtype=arrayin.dtype)
	arrayout = np.zeros(arrayin.shape,dtype=arrayin.dtype)
	circ     = circle(circ,radius=rad)
	drawP(circ)
	circ     = np.roll(circ,-arrayin.shape[0]/2,0)
	circ     = np.roll(circ,-arrayin.shape[1]/2,1)
	drawP(circ)
	for i in range(arrayin.shape[0]):
		for j in range(arrayin.shape[1]):
			if arrayin[i,j] > 0.5:
				arrayout += roll(circ,i,j)
	arrayout = (arrayout > 0.5)
	return arrayout

