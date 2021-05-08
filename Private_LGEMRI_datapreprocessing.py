# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:14:13 2021

@author: moona
"""
####################################### Private MRI dataset prepartion ################
import numpy as np
import os
import time
import cv2
#import nibabel as nib
import pdb
from matplotlib import pyplot as plt
import nibabel as nib
from nibabel.testing import data_path
import argparse
import os
import time
import numpy as np
import cv2
import nibabel as nib
import shutil
import ntpath
import sys
import pdb
import logging

from importlib import import_module
import torch
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F

#from torch.utils.data import DataLoader
#from torchvision import transforms
#from data_utils.torch_data import THOR_Data, get_cross_validation_paths, get_global_alpha
#import data_utils.transforms as tr
#from models.loss_funs import DependentLoss
#from utils import setgpu, get_threshold, metric, segmentation_metrics
def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255.

path = 'D:\\prof-Machine-learning\\COV-19 dataset\\COVID-19-CT-Seg_20cases'
save_path = 'D:\\prof-Machine-learning\\COV-19 dataset\\COV20imagefolder'

if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(path)
count = 0
print('begin processing data')

means = []
stds = []


for i, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(i, cur_file)
    cur_save_path = os.path.join(save_path, volume.split('.')[:-1][0])
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
    img = nib.load(cur_file)
    img = np.array(img.get_data())
    #img = truncated_range(img)
    img1=np.transpose(img,(2,0,1))
    for idx in range(img1.shape[0]):
        print(idx)
        cur_img = img1[idx,:,:]
        print(cur_img)
        count += 1
        cv2.imwrite(os.path.join(cur_save_path,volume.split('.')[:-1][0] + '_' + str(idx) + '_img.png'), cur_img)
print('processing data end !')

#%% My dataset code conversion from nifity to png
#%save image as a folder for every patient
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

#slice: 2d np array
#a single slice of MRI
#boundray=trueZone(slice)
#imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#normalization(imgNormalized)


path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\Images'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\AQProject\\completedataset2020heart\\nifti\\Inputimagefolder'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
    
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume.split('.')[:-1][0])
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
    img = nib.load(cur_file)
    img = np.array(img.get_data())
    #img1=np.swapaxes(img,0,2)
#    label = nib.load(os.path.join(cur_file, 'GT.nii.gz'))
#    label = np.array(label.get_data())
    #label=np.swapaxes(label,0,2)
    #seg=np.swapaxes(label, 0,2)
    vol=np.swapaxes(img,0,2)  
    # Convert to a visual format
    #vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    vol_ims=vol
#    boundray=trueZone(vol_ims)
#    imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#    normalization(imgNormalized)
#    seg_ims = class_to_color(seg, k_color, t_color,k1_color,t1_color)
#    seg_ims=class_to_color(seg, k_color, t_color,k1_color,t1_color)
#    d=1
#    for i in range(seg.shape[0]):
#        filename = "%d.png"%d
#        cv2.imwrite(os.path.join(cur_save_path1,cur_save_path1+'\\'+volume + '_' + str(i) +'_image'+ '_label.png'), seg[i]) # save segmentation mask
#        d+=1
    d=1
    for ii in range(vol_ims.shape[0]):
        filename = "%d.png"%d
        #boundray=trueZone(vol_ims[ii])
        #imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        #normalization(imgNormalized)
        #normal=boundray
        io.imsave(os.path.join(cur_save_path,cur_save_path+'\\'+volume.split('.')[:-1][0] + '_' + str(ii) +'_image'+ '.png'), vol_ims[ii]) # save input images
        d+=1         





#array_buffer = arr.tobytes()
#            img.frombytes(array_buffer, 'raw', "I;16")
#            im4 = exposure.rescale_intensity(arr, out_range='float')
#            imoo = img_as_uint(im4)
#            filename = "%d.png"%d
#            folderpath=os.path.join('niftiinter\\nifiticontoursinter',fileCase)
#            folderpath1=os.path.join('niftiinter\\nifitimagesinter',fileCase)
#            createFolder(folderpath)
#            createFolder(folderpath1)
#            #imageio.imwrite(folderpath+'\\'+filename,np.array(centralize(imgFinal, aJson.endocontour))) # save images
#            #img.save(folderpath+'\\'+filename) # save images
#            io.imsave(folderpath1+'\\'+filename,imoo)
#            #io.imsave('test_16bit.png', im)
#            #cv2.imwrite(folderpath+'\\'+filename,centralize(imgFinal, aJson.endocontour).astype(np.uint16)) # save images
#            cv2.imwrite(folderpath+'\\'+filename,centralize(maskFinal, aJson.endocontour)) # save images
#            d+=1
       
#%%  convert images using normalization function and cropping from center
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

path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\Images'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\AQProject\\completedataset2020heart\\nifti\\Inputimagefolder1crop1'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
    
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume.split('.')[:-1][0])
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
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
        boundray=trueZone(mask)
        imgNormalized=mask[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        normalization(imgNormalized)
        #mask=cropND(mask, (128,128))
        mask=cropfunction(mask)
        filename = "%d.png"%d
        io.imsave(os.path.join(cur_save_path,cur_save_path+'\\'+volume.split('.')[:-1][0] + '_' + str(test) +'_image'+ '.png'), mask) # save input images
        d+=1   
#%
#%
#%%   cropping the images without normalization the images
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

path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\Images'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\AQProject\\completedataset2020heart\\nifti\\Multiclassdataset\\InputimageWN'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
    
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume.split('.')[:-1][0])
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
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
        #mask=cropND(mask, (128,128))
        mask=cropfunction(mask)
        filename = "%d.png"%d
        io.imsave(os.path.join(cur_save_path,cur_save_path+'\\'+volume.split('.')[:-1][0] + '_' + str(test) +'_image'+ '.png'), mask) # save input images
        d+=1           
        
        
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
folders = glob.glob('D:\\AQProject\\completedataset2020heart\\nifti\\Multiclassdataset\\InputimageWN\\*')
foldername='D:\\AQProject\\completedataset2020heart\\nifti\\Multiclassdataset\\InputimageWNfolder'
d=1
for mypath in folders:
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
#%% My dataset code conversion from nifity to png this portion is belong to masks
#%save image as a folder for every patient
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

#slice: 2d np array
#a single slice of MRI
#boundray=trueZone(slice)
#imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#normalization(imgNormalized)


path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\Contours'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\AQProject\\completedataset2020heart\\nifti\\Inputmaskfolder'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
    
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume.split('.')[:-1][0])
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
    img = nib.load(cur_file)
    img = np.array(img.get_data())
    #img1=np.swapaxes(img,0,2)
#    label = nib.load(os.path.join(cur_file, 'GT.nii.gz'))
#    label = np.array(label.get_data())
    #label=np.swapaxes(label,0,2)
    #seg=np.swapaxes(label, 0,2)
    vol=np.swapaxes(img,0,2)  
    # Convert to a visual format
    #vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    vol_ims=vol
#    boundray=trueZone(vol_ims)
#    imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
#    normalization(imgNormalized)
#    seg_ims = class_to_color(seg, k_color, t_color,k1_color,t1_color)
#    seg_ims=class_to_color(seg, k_color, t_color,k1_color,t1_color)
#    d=1
#    for i in range(seg.shape[0]):
#        filename = "%d.png"%d
#        cv2.imwrite(os.path.join(cur_save_path1,cur_save_path1+'\\'+volume + '_' + str(i) +'_image'+ '_label.png'), seg[i]) # save segmentation mask
#        d+=1
    d=1
    for ii in range(vol_ims.shape[0]):
        filename = "%d.png"%d
        #boundray=trueZone(vol_ims[ii])
        #imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        #normalization(imgNormalized)
        #normal=boundray
        io.imsave(os.path.join(cur_save_path,cur_save_path+'\\'+volume.split('.')[:-1][0] + '_' + str(ii) +'_image'+ '.png'), vol_ims[ii]) # save input images
        d+=1   

#%%   cropping the mask from center and make equal size ##########################
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

path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\Contours'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\AQProject\\completedataset2020heart\\nifti\\Inputmaskfoldercrop'  # save path image folder folder name is img; provid you folder name to saving image folder
#save_path1='D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\Datasetcolor\\mask1' # save mask path folder name should be mask; provide your folder name to saving masks folder
files=os.listdir(path)
    
for idx, volume in enumerate(files):
    total_imgs = []
    cur_file = os.path.join(path, volume)
    print(idx, cur_file)
    cur_save_path = os.path.join(save_path, volume.split('.')[:-1][0])
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
#    cur_save_path1 = os.path.join(save_path1, volume)
#    if not os.path.exists(cur_save_path1):
#        os.makedirs(cur_save_path1)
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
        #mask=cropND(mask, (128,128))
        mask=cropfunction(mask)
        filename = "%d.png"%d
        io.imsave(os.path.join(cur_save_path,cur_save_path+'\\'+volume.split('.')[:-1][0] + '_' + str(test) +'_image'+ '.png'), mask) # save input images
        d+=1           

#%%make the size equal to 128x128 of masking with full myocarium border #######################
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
    
def cropfunction(img):
    y=int((256-128)/2)
    x=int((256-128)/2)
    # Crop, convert back from numpy to PIL Image and and save
    cropped=img[x:x+128,y:y+128]
    return cropped    
import natsort
folders = glob.glob('D:\\AQProject\\completedataset2020heart\\nifti\\Inputmaskfoldercrop\\*')
foldername='D:\\AQProject\\completedataset2020heart\\nifti\\testmasks128x128'
d=1
for mypath in folders:
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    print(natsort.natsorted(onlyfiles,reverse=False))
    for n in natsort.natsorted(range(0, len(onlyfiles))):
        imge= cv2.imread( join(mypath,natsort.natsorted(onlyfiles,reverse=False)[n]) )
        imge[imge==2]=2
        imge[imge==1]=0
        imge[imge==3]=1
        imge[imge==4]=1
        imge[imge>=1]=1
        #imge=cropfunction(imge)
        #boundray=trueZone(imge)
        #imgNormalized=slice[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        #imge=normalization(imgNormalized)
        #imge= cv2.resize(imge, (256,256), interpolation = cv2.INTER_AREA)
        imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
        #imge= cv2.resize( imge,(256,256) )
        filename = "%d.png"%d
        #cv2.imwrite(foldername+'\\'+filename, imge) # save images
        io.imsave(foldername+'\\'+filename, imge*255) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
        d+=1        