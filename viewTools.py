
#------------------------------------------------------------------------------------------
# Authors: Andrew Mortin, Andrew Morgan
# Paper reference: 
#   A. Martin, A. Morgan, T. Ekeberg, N. Loh, F. Maia, F. Wang, J. Spence, and H. Chapman, 
#   "The extraction of single-particle diffraction patterns from a multiple-particle 
#   diffraction pattern," Opt. Express  21, 15102-15112 (2013).
#
#
# A.V. Martin 2012
# 	- Display class for simple viewing of images
#         during a calculation
#
#       Example usage:
#          - Initialise the window manager
#          wm = ViewTools.WindowManager()
#
#          - create a display myImage in a new window
#          wm.NewWindow(myImage,       label="MyImageName")
#
#          - display an image in a second window
#          wm.NewWindow(mySecondImage, label="MySecondimageName")
#
#          - change the image displayed in the first window
#          wm.windowlist("MyImageName").update(modifiedimage)
#
#          - finish program with
#          wm.show()
#
#
#       TODO:
#          - Add options on where the windows appear
#          - Add more comments to this file
#          - Allow titles, widths and positions to be updated
#------------------------------------------------------------------------------------------


import Tkinter as tk
from PIL import Image 
import ImageTk
import numpy as np
import tkFileDialog
from time import sleep
from matplotlib.colors import Normalize
import matplotlib.cm as cm




#
# Class to allow multiple windows to be opened and updated
# during a calculation
#

class WindowManager:
	def __init__(self):
	     
             self.root = tk.Tk()
             self.root.title("Root window")
	     

             #
             # instances of Window() are stored in the windowlist
             #
             self.windowlist = {}

             #
             # main window is not used for anything
             # remains hidden while the program is running
             #
             self.root.withdraw()

        #
        # create a new instance of Window() and store it in the windowlist
        #
        def NewWindow(self,dataimage,label,gamma=0.3,pause=0.01):
             self.windowlist[label] = Window(self.root,dataimage,label=label,\
                                                        gamma=gamma,pause=pause,pack=0)
	     self.windowlist[label].w.bind("<q>", self.close)
	     self.windowlist[label].window.protocol("WM_DELETE_WINDOW", self.kill_all_windows)
	     self.windowlist[label].pack()


	def display(self, dataimage, label):
		self.windowlist[label].update( dataimage )

        #
        # hide the root window
        #
        def window_hide(self):        
            self.root.withdraw()

        #
        # close all the windows
        #
        def show_and_close(self):
             self.root.destroy()
             self.root.mainloop()

	def close(self,event=0):
             self.root.destroy()

        #
        # windows remain open; main window is shown too so that windows can be closed
        # may replace this property by a key binding.
        #
        def show(self):
            #self.root.deiconify()
            #print "Tk program finished - Close the root window to return to the shell"
            #print "In future, this could be a key binding"
            self.root.mainloop()


	def kill_all_windows(self):
		self.root.destroy()


#
# A specific window to be created inside the window manager
#
# label  =  title for the window
# wx, wy = width and 
#

class Window:
    def __init__(self,root,dataimage,label="New Figure",wx=512,wy=512,gamma=0.3,pause=0,pack=1):
          
          self.window = tk.Toplevel()
          self.window.title(label)
	  self.wx, self.wy = wx, wy
          self.w = tk.Canvas(self.window,width=self.wx,height=self.wy)           
          self.update(dataimage,gamma=gamma,pause=pause)
	  self.w.bind("<Button-1>",self.focus_on_canvas)
	  if pack==1:
		  self.w.pack()
	  else:
		  pass
    

    def focus_on_canvas(self,event=0):
          self.w.focus_set()


    #
    # changes the image displayed inside an existing window
    #

    def update(self,dataimage,gamma=0.3,pause=0.01):
          self.display(dataimage,gamma)
          self.window.update()

          # Bring the window to the top
          self.window.wm_attributes("-topmost",1)
          # Don't force window to stay on top
          self.window.wm_attributes("-topmost",0)
          sleep(pause)

    #
    # sets up a new image for display in window
    # (still requires update() step to refresh the window display)
    #
    
    def display(self,dataimage,gamma=0.3):	 
             		 
          scaled = dataimage 
          scaled = np.abs(scaled *(scaled >= 0.0) )**gamma \
	      - (np.abs(scaled *(scaled <= 0.0) )**gamma)
	  scaled += np.min( scaled )
          scaled *= 255. / np.max(scaled)   
	 
          
	  cmap = cm.jet

	  scaledI = scaled 
          scaled = (cmap( scaledI.astype(np.int) ) * 255).astype(np.uint8) 
          image = Image.fromarray(scaled, "RGBA")
	 
          photo = ImageTk.PhotoImage( image.resize((self.wx,self.wy)) )
          self.w.image = photo
          self.w.create_image((self.wx/2,self.wy/2),image=photo)

    #
    # window becomes visible on the screen
    #        
    def window_show(self):        
          self.window.deiconify()

    #
    # hides the window from the screen.
    # 
    def window_hide(self):        
          self.window.withdraw()

    def pack(self):
          self.w.pack()
