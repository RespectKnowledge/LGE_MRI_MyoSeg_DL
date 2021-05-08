# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:34:23 2021

@author: Abdul Qayyum
"""
#%% challenege training dataset
import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from pathlib import Path

import nibabel as nib
import glob
import os
import numpy as np
import cv2
import os
import json
import os, os.path
import numpy as np
import cv2
import pydicom
import glob
from scipy import ndimage, misc
import nibabel as nib
import matplotlib
import scipy.misc
from PIL import Image
import numpy as np
import imageio
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import pandas as pd
def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3

hu_min=DEFAULT_HU_MIN 
hu_max=DEFAULT_HU_MAX 
######################################### Normalization code#################################
def trueZone(imgArray):
    #return a 2d array of no-black zone boundary
    H=imgArray.shape[0]
    W=imgArray.shape[1]
   
    for h0 in range(H):
        if imgArray[h0,:].sum()!=0:
            break
    for h1 in range(H-1, -1,-1):
        if imgArray[h1,:].sum()!=0:
            break
       
    for w0 in range(W):
        if imgArray[:,w0].sum()!=0:
            break
    for w1 in range(W-1, -1,-1):
        if imgArray[:,w1].sum()!=0:
            break
    return [h0,h1, w0,w1]
######################################### Normalization code#################################
def normalization(imgArray):
    #in place function. Normalize the image
    imgFlat=imgArray.copy()
    imgFlat=np.sort(imgFlat, axis=None)

    windowlower=imgFlat[int(imgFlat.shape[0]*0.05)]
    windowupper=imgFlat[int(imgFlat.shape[0]*0.95)-1] 

    imageNormalized=np.copy(imgArray)
    imageNormalized=np.clip(imageNormalized,windowlower,windowupper)-windowlower
    imageNormalized=imageNormalized/(windowupper-windowlower)*4095
    imgArray[:,:]=imageNormalized

import operator
# center cropping
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


#a = np.arange(100).reshape((10,10))
#cropND(a, (5,5))
#slice: 2d np array
#a single slice of MRI
#boundray=trueZone(slice)
#imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#normalization(imgNormalized)
def cropfunction(img):
    y=int((256-128)/2)
    x=int((256-128)/2)
    # Crop, convert back from numpy to PIL Image and and save
    cropped=img[x:x+128,y:y+128]
    return cropped
def cropMiddle(imgArray):
    x0=int(imgArray.shape[1]/2)-48
    y0=int(imgArray.shape[0]/2)-48
    return imgArray[y0:y0+96, x0:x0+96]


path='D:\\Moonacompitionemdic\\khwala\\2cases\\1\\volums\\trainingimages'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\Moonacompitionemdic\\khwala\\2cases\\1\\volums\\trainingfolder'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
    
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume)
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
    cur_file1=os.path.join(cur_file,'Images'+'\\'+str(volume)+'.nii.gz')
    img = nib.load(cur_file)
    img = np.array(img.get_data())
#    img = Image.new("I", img.T.shape)
#    im = Image.fromarray(img)
#    array_buffer = img.tobytes()
#    img.frombytes(array_buffer, 'raw', "I;16")
#    im4 = exposure.rescale_intensity(img, out_range='float')
#    img = img_as_uint(im4)
    d=1
    for test in range(img.shape[2]):
        mask=img[:,:,test]
        #boundray=trueZone(mask)
        #imgNormalized=mask[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        #normalization(imgNormalized)
        y=mask
        # mask = (65535*((y - y.min())/y.ptp())).astype(np.uint16) # check conversion from different number of bits
        #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY) 
        mask=cropMiddle(mask)
        #mask=cropfunction(mask)
        filename = "%d.png"%d
        io.imsave(os.path.join(cur_save_path, str(test) +'_image'+ '.png'), mask) # save input images
        d+=1   
#%
#%%  saving image folder
import numpy as np
import scipy.misc  
import cv2
import numpy
import glob
import pylab as plt
import os
#import png
import pydicom
import numpy as np
from pydicom.tag import Tag
import time
from PIL import Image, ImageOps           
import numpy as np
import scipy.misc  
import cv2
import numpy
import glob
import pylab as plt
import os
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from skimage import io, exposure, img_as_uint, img_as_float
#from glob import glob
#
def resize(listImg):
    #crop each image of a same patient to ensure the same image size
    xmin=9999
    ymin=9999
    for img in listImg:
        if img.shape[0]<ymin:
            ymin=img.shape[0]
        if img.shape[1]<xmin:
            xmin=img.shape[1]
            
    for _,img in enumerate(listImg):
        imgResized=img[int((img.shape[0]-ymin)/2):int((img.shape[0]-ymin)/2)+ymin,
          int((img.shape[1]-xmin)/2):int((img.shape[1]-xmin)/2)+xmin]
        if _==0:
            imgConcatenated=np.array([imgResized])
        else:
            imgConcatenated=np.concatenate((imgConcatenated, [imgResized]),axis=0)
    return np.transpose(imgConcatenated, (2,1,0))     

def trueZone(imgArray):
    #return a 2d array of no-black zone boundary
    H=imgArray.shape[0]
    W=imgArray.shape[1]
   
    for h0 in range(H):
        if imgArray[h0,:].sum()!=0:
            break
    for h1 in range(H-1, -1,-1):
        if imgArray[h1,:].sum()!=0:
            break
       
    for w0 in range(W):
        if imgArray[:,w0].sum()!=0:
            break
    for w1 in range(W-1, -1,-1):
        if imgArray[:,w1].sum()!=0:
            break
    return [h0,h1, w0,w1]

def normalization(imgArray):
    #in place function. Normalize the image
    imgFlat=imgArray.copy()
    imgFlat=np.sort(imgFlat, axis=None)

    windowlower=imgFlat[int(imgFlat.shape[0]*0.05)]
    windowupper=imgFlat[int(imgFlat.shape[0]*0.95)-1] 

    imageNormalized=np.copy(imgArray)
    imageNormalized=np.clip(imageNormalized,windowlower,windowupper)-windowlower
    imageNormalized=imageNormalized/(windowupper-windowlower)*4095
    imgArray[:,:]=imageNormalized
    
    
import natsort
folders = glob.glob('D:\\Moonacompitionemdic\\khwala\\2cases\\1\\volums\\trainingfolder\\*')
foldername='D:\\Moonacompitionemdic\\khwala\\2cases\\1\\volums\\trainingsequence1'
d=1
for mypath in natsort.natsorted(folders):
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    print(natsort.natsorted(onlyfiles,reverse=False))
    for n in natsort.natsorted(range(0, len(onlyfiles))):
        imge= cv2.imread( join(mypath,natsort.natsorted(onlyfiles,reverse=False)[n]) )
        #boundray=trueZone(imge)
        #imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        #imge=normalization(imgNormalized)
        #imge= cv2.resize(imge, (256,256), interpolation = cv2.INTER_AREA)
        imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
        #imge= cv2.resize( imge,(256,256) )
        filename = "%d.png"%d
        #cv2.imwrite(foldername+'\\'+filename, imge) # save images
        io.imsave(foldername+'\\'+filename, imge) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
        d+=1 
        
#
#%% cropping the mask from center and make equal size ##########################
import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from pathlib import Path

import nibabel as nib
import glob
import os
import numpy as np
import cv2
import os
import json
import os, os.path
import numpy as np
import cv2
import pydicom
import glob
from scipy import ndimage, misc
import nibabel as nib
import matplotlib
import scipy.misc
from PIL import Image
import numpy as np
import imageio
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import pandas as pd
def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3

hu_min=DEFAULT_HU_MIN 
hu_max=DEFAULT_HU_MAX 
######################################### Normalization code#################################
def trueZone(imgArray):
    #return a 2d array of no-black zone boundary
    H=imgArray.shape[0]
    W=imgArray.shape[1]
   
    for h0 in range(H):
        if imgArray[h0,:].sum()!=0:
            break
    for h1 in range(H-1, -1,-1):
        if imgArray[h1,:].sum()!=0:
            break
       
    for w0 in range(W):
        if imgArray[:,w0].sum()!=0:
            break
    for w1 in range(W-1, -1,-1):
        if imgArray[:,w1].sum()!=0:
            break
    return [h0,h1, w0,w1]

def normalization(imgArray):
    #in place function. Normalize the image
    imgFlat=imgArray.copy()
    imgFlat=np.sort(imgFlat, axis=None)

    windowlower=imgFlat[int(imgFlat.shape[0]*0.05)]
    windowupper=imgFlat[int(imgFlat.shape[0]*0.95)-1] 

    imageNormalized=np.copy(imgArray)
    imageNormalized=np.clip(imageNormalized,windowlower,windowupper)-windowlower
    imageNormalized=imageNormalized/(windowupper-windowlower)*4095
    imgArray[:,:]=imageNormalized
import operator
# center cropping
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


#a = np.arange(100).reshape((10,10))
#cropND(a, (5,5))
#slice: 2d np array
#a single slice of MRI
#boundray=trueZone(slice)
#imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#normalization(imgNormalized)
def cropfunction(img):
    y=int((256-128)/2)
    x=int((256-128)/2)
    # Crop, convert back from numpy to PIL Image and and save
    cropped=img[x:x+128,y:y+128]
    return cropped
def cropMiddle(imgArray):
    x0=int(imgArray.shape[1]/2)-48
    y0=int(imgArray.shape[0]/2)-48
    return imgArray[y0:y0+96, x0:x0+96]
path='D:\\Moonacompitionemdic\\khwala\\seg02\\masktraining'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\Moonacompitionemdic\\khwala\\seg02\\maskingfolder'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
    
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume)
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
    cur_file1=os.path.join(cur_file,'Contours'+'\\'+str(volume)+'.nii.gz')
    img = nib.load(cur_file)
    img = np.array(img.get_data())
#    img = Image.new("I", img.T.shape)
#    im = Image.fromarray(img)
#    array_buffer = img.tobytes()
#    img.frombytes(array_buffer, 'raw', "I;16")
#    im4 = exposure.rescale_intensity(img, out_range='float')
#    img = img_as_uint(im4)
    d=1
    for test in range(img.shape[2]):
        mask=img[:,:,test]
#        boundray=trueZone(mask)
#        imgNormalized=mask[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#        normalization(imgNormalized)
        imge=cropMiddle(mask)
        # imge[imge==2]=2
        # imge[imge==1]=0
        # imge[imge==3]=1
        # imge[imge==4]=1
        # imge[imge>=1]=1
        #mask=cropfunction(mask)
        filename = "%d.png"%d
        io.imsave(os.path.join(cur_save_path,str(test) +'_mask'+ '.png'), imge*255) # save input images
        d+=1
#%% save mask mage sequnce for training dataset
import numpy as np
import scipy.misc  
import cv2
import numpy
import glob
import pylab as plt
import os
#import png
import pydicom
import numpy as np
from pydicom.tag import Tag
import time
from PIL import Image, ImageOps           
import numpy as np
import scipy.misc  
import cv2
import numpy
import glob
import pylab as plt
import os
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from skimage import io, exposure, img_as_uint, img_as_float
#from glob import glob
#
def resize(listImg):
    #crop each image of a same patient to ensure the same image size
    xmin=9999
    ymin=9999
    for img in listImg:
        if img.shape[0]<ymin:
            ymin=img.shape[0]
        if img.shape[1]<xmin:
            xmin=img.shape[1]
            
    for _,img in enumerate(listImg):
        imgResized=img[int((img.shape[0]-ymin)/2):int((img.shape[0]-ymin)/2)+ymin,
          int((img.shape[1]-xmin)/2):int((img.shape[1]-xmin)/2)+xmin]
        if _==0:
            imgConcatenated=np.array([imgResized])
        else:
            imgConcatenated=np.concatenate((imgConcatenated, [imgResized]),axis=0)
    return np.transpose(imgConcatenated, (2,1,0))     

def trueZone(imgArray):
    #return a 2d array of no-black zone boundary
    H=imgArray.shape[0]
    W=imgArray.shape[1]
   
    for h0 in range(H):
        if imgArray[h0,:].sum()!=0:
            break
    for h1 in range(H-1, -1,-1):
        if imgArray[h1,:].sum()!=0:
            break
       
    for w0 in range(W):
        if imgArray[:,w0].sum()!=0:
            break
    for w1 in range(W-1, -1,-1):
        if imgArray[:,w1].sum()!=0:
            break
    return [h0,h1, w0,w1]

def normalization(imgArray):
    #in place function. Normalize the image
    imgFlat=imgArray.copy()
    imgFlat=np.sort(imgFlat, axis=None)

    windowlower=imgFlat[int(imgFlat.shape[0]*0.05)]
    windowupper=imgFlat[int(imgFlat.shape[0]*0.95)-1] 

    imageNormalized=np.copy(imgArray)
    imageNormalized=np.clip(imageNormalized,windowlower,windowupper)-windowlower
    imageNormalized=imageNormalized/(windowupper-windowlower)*4095
    imgArray[:,:]=imageNormalized
    
    
import natsort
folders = glob.glob('D:\\Moonacompitionemdic\\khwala\\seg02\\maskingfolder\\*')
foldername='D:\\Moonacompitionemdic\\khwala\\seg02\\masksequence'
d=1
for mypath in natsort.natsorted(folders):
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    print(natsort.natsorted(onlyfiles,reverse=False))
    for n in natsort.natsorted(range(0, len(onlyfiles))):
        imge= cv2.imread( join(mypath,natsort.natsorted(onlyfiles,reverse=False)[n]) )
        #boundray=trueZone(imge)
        #imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        #imge=normalization(imgNormalized)
        #imge= cv2.resize(imge, (256,256), interpolation = cv2.INTER_AREA)
        imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
        #imge= cv2.resize( imge,(256,256) )
        filename = "%d.png"%d
        #cv2.imwrite(foldername+'\\'+filename, imge) # save images
        io.imsave(foldername+'\\'+filename, imge) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
        d+=1 
#%% challenege test dataset
import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from pathlib import Path

import nibabel as nib
import glob
import os
import numpy as np
import cv2
import os
import json
import os, os.path
import numpy as np
import cv2
import pydicom
import glob
from scipy import ndimage, misc
import nibabel as nib
import matplotlib
import scipy.misc
from PIL import Image
import numpy as np
import imageio
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import pandas as pd
def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3

hu_min=DEFAULT_HU_MIN 
hu_max=DEFAULT_HU_MAX 
######################################### Normalization code#################################
def trueZone(imgArray):
    #return a 2d array of no-black zone boundary
    H=imgArray.shape[0]
    W=imgArray.shape[1]
   
    for h0 in range(H):
        if imgArray[h0,:].sum()!=0:
            break
    for h1 in range(H-1, -1,-1):
        if imgArray[h1,:].sum()!=0:
            break
       
    for w0 in range(W):
        if imgArray[:,w0].sum()!=0:
            break
    for w1 in range(W-1, -1,-1):
        if imgArray[:,w1].sum()!=0:
            break
    return [h0,h1, w0,w1]
######################################### Normalization code#################################
def normalization(imgArray):
    #in place function. Normalize the image
    imgFlat=imgArray.copy()
    imgFlat=np.sort(imgFlat, axis=None)

    windowlower=imgFlat[int(imgFlat.shape[0]*0.05)]
    windowupper=imgFlat[int(imgFlat.shape[0]*0.95)-1] 

    imageNormalized=np.copy(imgArray)
    imageNormalized=np.clip(imageNormalized,windowlower,windowupper)-windowlower
    imageNormalized=imageNormalized/(windowupper-windowlower)*4095
    imgArray[:,:]=imageNormalized

import operator
# center cropping
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

from PIL import Image
from resizeimage import resizeimage

# fd_img = open('test-image.jpeg', 'r')
# img = Image.open(fd_img)
# img = resizeimage.resize_crop(img, [200, 200])
# img.save('test-image-crop.jpeg', img.format)
# fd_img.close()
#a = np.arange(100).reshape((10,10))
#cropND(a, (5,5))
#slice: 2d np array
#a single slice of MRI
#boundray=trueZone(slice)
#imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#normalization(imgNormalized)
# def cropfunction(img):
#     y=int((256-128)/2)
#     x=int((256-128)/2)
#     # Crop, convert back from numpy to PIL Image and and save
#     cropped=img[x:x+128,y:y+128]
#     return cropped
def cropMiddle1(imgArray):
    x0=int(imgArray.shape[1]/2)-64
    y0=int(imgArray.shape[0]/2)-64
    return imgArray[y0:y0+128, x0:x0+128]

def cropMiddle(imgArray):
    x0=int(imgArray.shape[1]/2)-48
    y0=int(imgArray.shape[0]/2)-48
    return imgArray[y0:y0+96, x0:x0+96]

path='D:\\Moonacompitionemdic\\khwala\\2cases\\1\\volums\\test'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\Moonacompitionemdic\\khwala\\2cases\\1\\volums\\testimg'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
#import cv 
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume)
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
    cur_file1=os.path.join(cur_file,'Images'+'\\'+str(volume)+'.nii.gz')
    img = nib.load(cur_file)
    img = np.array(img.get_data())
#    img = Image.new("I", img.T.shape)
#    im = Image.fromarray(img)
#    array_buffer = img.tobytes()
#    img.frombytes(array_buffer, 'raw', "I;16")
#    im4 = exposure.rescale_intensity(img, out_range='float')
#    img = img_as_uint(im4)
    d=1
    for test in range(img.shape[2]):
        mask=img[:,:,test]
        #mask= mask.resize((128,128))
        #mask = cv2.resize(mask, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        # boundray=trueZone(mask)
        # imgNormalized=mask[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        # normalization(imgNormalized)
        # mask=cropND(mask, (128,128))
        #mask = resizeimage.resize_crop(mask, [128, 128])
        #mask= mask.resize((128, 128), Image.ANTIALIAS)  # LANCZOS as of Pillow 2.7
        #mask=cropfunction(mask)
        y=mask
        #mask = (65535*((y - y.min())/y.ptp())).astype(np.uint16) # check conversion from different number of bits
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB) 
        #mask = (4096*((y - y.min())/y.ptp())).astype(np.uint16) # save using 12 bits
        mask=cropMiddle(mask)
        filename = "%d.png"%d
        cv2.imwrite(os.path.join(cur_save_path,str(test) +'_image'+ '.png'), mask) # save input images
        d+=1   
#%%
#% Dataset generation
path_train='D:\\Moonacompitionemdic\\khwala\\2cases\\2\\volums\\'


import os
import numpy as np
train_ids = next(os.walk(path_train+"trainingsequence1"))[2]
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.io import imread

#import tensorflow as tf

im_height=96
im_width=96
im_chan=1
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.uint8)
print('Getting and resizing train images and masks ... ')
#sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    img = imread(path + '/trainingsequence1/' + id_)
    x = np.array(img)
    x = resize(x, (96, 96, 1), mode='constant', preserve_range=True)
    X_train[n] = x
    p=imread(path + '/masksequence' +'\\'+id_.replace('img','msk'))
    #mask = np.expand_dims((p)[:,:,1],
    #p=p[:,:,0]
    mask=np.expand_dims(resize(p, (96, 96), mode='constant', preserve_range=True), axis=-1)
    #mask=np.expand_dims(resize(p, (200, 200),preserve_range=True),axis=-1)                     
    Y_train[n] =mask 

print('Done!')
npy_data_path='D:\\Moonacompitionemdic\\khwala\\2cases\\2\\volums\\'
#np.save(os.path.join(npy_data_path, 'images_test.npy'), X_train)
#np.save(os.path.join(npy_data_path, 'masks_test.npy'), Y_train)
#np.save(os.path.join(npy_data_path, 'ids_test.npy'), train_ids)  
np.save(os.path.join(npy_data_path, 'images_trainck1.npy'), X_train)
np.save(os.path.join(npy_data_path, 'masks_trainck1.npy'), Y_train)
np.save(os.path.join(npy_data_path, 'ids_trainck1.npy'),train_ids) 