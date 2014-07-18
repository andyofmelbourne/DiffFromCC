
## my favorate functions just to simplify life

#from numpy import outer,roll,arange,zeros,ones,exp,sqrt,where
import numpy as np
import h5py
import scipy.interpolate as spi


# a 2D array where each pixel is set to its x coordinate
def make_xarr(nx,ny): 
	xarr = np.outer(np.arange(0-nx/2,nx-nx/2,1),np.ones(ny))
	return xarr

# a 2D array where each pixel is set to its y coordinate 
def make_yarr(nx,ny):    	
	yarr = np.outer(np.ones(nx),np.arange(0-ny/2,ny-ny/2,1))
	return yarr

# shift - a 2D version of numpy's roll
def array_shift(array,xshift=0,yshift=0):
	array = np.roll(array,xshift,0)
	array = np.roll(array,yshift,1)
	return array

## make an array with a circle set to one
def circle(nx, ny, rad=None, cenx=None, ceny=None, invert=0 ): 
	
    # set defaults
    if rad is None: rad = np.min([nx,ny])/2
    if cenx is None: cenx = nx/2
    if ceny is None: ceny = ny/2

    # define the circle
    x = np.outer(np.arange(nx),np.ones(ny)) - nx/2
    y = np.outer(np.ones(nx),np.arange(ny)) - ny/2
   
    dist = np.sqrt(x**2 + y**2)
    a = np.zeros([nx,ny])
    icirc = np.where(dist <= rad)
    a[icirc] = 1.

    if (invert==1):
	    a = abs(1-a)

    out = array_shift(a, -nx/2, -ny/2)
    out = array_shift(out, cenx, ceny)
    return out
#end circle

## make a 2D array with a gaussian
def make_gaussian(nx, ny, rad=None, rady=-1., cenx=None, ceny=None, invert=0, norm=False ): 
    #set defaults
    if rad is None: rad = np.min(nx,ny)/2
    if cenx is None: cenx = nx/2
    if ceny is None: ceny = ny/2
    radsq = rad**2
    if rady == -1.:
        radysq = radsq
    else:
        radysq = rady**2

    # define the circle
    x = np.outer(np.arange(0-nx/2,nx-nx/2,1),np.ones(ny))
    #print x.size, x.shape
    y = np.outer(np.ones(nx),np.arange(0-ny/2,ny-ny/2,1))
    #print y.size, y.shape

    a = np.zeros([nx,ny])
    a = np.exp(-(x**2)/radsq  - ( y**2)/radysq)
    a[ nx/2, ny/2 ] = 1.0

    a = array_shift(a,cenx-nx/2,ceny-ny/2)

    # normalise if required
    if norm == True: a *= 1./np.sum(a)
    
    return a
#end make_gaussian

## make a distance function
def dist2d(nx,ny):
        cenx = nx/2
        ceny = ny/2
        x = np.outer(np.arange(0-cenx,nx-cenx,1),np.ones(ny))
        #print x.size, x.shape
        y = np.outer(np.ones(nx),np.arange(0-ceny,ny-ceny,1))
        #print y.size, y.shape
        dist = np.sqrt(x**2 + y**2)
        dist = array_shift(dist,cenx,ceny)
        return dist
# read the data from a h5 file
def h5read(filename,field="data/data1"):
     h5file = h5py.File(filename,"r")
     #print field
     h5data = h5file[field]
     image = h5data[...]
     h5file.close()
     return image

def h5write(filename,data,field="data/data"):
     f = h5py.File(filename, 'w')    # overwrite any existing file=
     dset = f.create_dataset(field, data=data)
     f.close()

def h5list(filename):
     h5file = h5py.File(filename,"r")
     list(h5file)

def croparray(array,nx, ny, cenx,ceny ):
	s = array.shape
	xshift = s[0]/2 - cenx #- 1
	yshift = s[1]/2 - ceny #- 1

	shifted = array_shift(array, xshift, yshift)

	cropped = np.zeros( (nx,ny) , dtype=array.dtype)

	x = [(s[0]-nx)/2, (s[0]+nx)/2] 
	y = [(s[1]-ny)/2, (s[1]+ny)/2] 

	cropped = shifted[ x[0]:x[1] , y[0]:y[1]  ]

	cropped = array_shift( cropped, -nx/2, -ny/2)
	return cropped

def padarray(array,nx,ny,cenx,ceny):
	padded = np.zeros( (nx,ny) , dtype=array.dtype)
	s = array.shape
	array = array_shift(array, s[0]/2, s[1]/2)

	padded[ 0:s[0], 0:s[1] ] = array
	
	xshift = cenx - s[0]/2 #+ 1
	yshift = ceny - s[1]/2 #+ 1
	padded = array_shift(padded, xshift, yshift)

	return padded

#
# Calculate the radial average of an image
#

def image_radial( image, cx, cy , method='cubic', rbins=-1):
	
	s = image.shape

	a = np.arange( s[0] ) 
	x = np.outer( a[::-1] , np.ones( s[1] ) )
	y = np.outer( np.ones(s[0]), a[::-1] )
	
	maxsave = 0

        rtemp = np.sqrt( (x - cx)**2 + (y - cy)**2 )

        rmin = 0
        rmax = np.min( s )/2
	if rbins == -1:
		rbins = (rmax - rmin)
        
        tempc = x-cx + 1j*(y-cy)
        thtemp = np.angle( tempc )
        thmin = np.min(thtemp)
        thmax = np.max(thtemp)
        thbins = 500
		
	print rmin, rmax, rbins, thmin, thmax

        r = np.outer( np.arange(rbins)*(rmax-rmin)/float(rbins) + rmin, \
                          np.ones( thbins ) )

        th = np.outer( np.ones(rbins), \
                           np.arange( thbins )*(thmax-thmin)/float(thbins) + thmin )
		
        xnew = r*np.cos(th)
        ynew = r*np.sin(th)

        xorig = x - cx
        yorig = y - cy
		    
        print "interpolating..."
        a = xorig.reshape(x.size)
        b = yorig.reshape(y.size)
        c = image.reshape(image.size)

        polar = spi.griddata( (a,b), c, (xnew, ynew), method=method)

        ip = np.where( np.isnan(polar) )
        polar[ip] = 0.0
        
        izero = np.where( polar > 0.0 )
        mask = polar * 0.0
        mask[izero] = 1.0
		
        print "... done"
        
        # calculate something
        rav_temp = np.sum( polar, 1)
        mav = np.sum( mask,  1)
        imav = np.where( mav > 0.0 )
                
        rav = rav_temp*0.0
        rav[imav] = rav_temp[imav] / mav[imav] 

        return polar, rav, [rmin, rmax, rbins, (rmax-rmin)/float(rbins), s[0], s[1]]

#
#  Takes rav, output from image_radial and make a 2D image
#  equal to the angularly averaged function
#
def rav_to_image(rav, rdata):

	# I have to step through the logic of this carefully...

	s = [rdata[4], rdata[5]] #[ rav.size*2, rav.size*2   ]

	a = np.arange( s[0] ) 
	x = np.outer( a[::-1] , np.ones( s[1] ) )
	y = np.outer( np.ones(s[0]), a[::-1] )
	
        rtemp = np.sqrt( (x - rdata[4]/2)**2 + (y - rdata[5]/2)**2 )
	remainder = rtemp % rtemp.astype(np.long)
	
	print rtemp
	print remainder

	output = rtemp*(0.0+0.0j)
	print "DEBUG some sizes output", output.shape

	for i in np.arange(rav.size-1):
		ir = np.where( (rtemp.astype(np.long) >= long(rdata[0] + i*rdata[3]) )\
				      & (rtemp.astype(np.long) < long(rdata[0] + (i+1)*rdata[3])) )
		output[ir] = remainder[ir]*rav[i] + (1.0-remainder[ir])*rav[i+1]

	return output




	
