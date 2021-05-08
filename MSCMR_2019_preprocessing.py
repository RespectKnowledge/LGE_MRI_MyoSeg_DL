# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:19:40 2021

@author: Abdul Qayyum
"""

#%% MSCMR 2019 challenege dataset prepartion
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

def cropMiddle(imgArray):
    x0=int(imgArray.shape[1]/2)-112
    y0=int(imgArray.shape[0]/2)-112
    return imgArray[y0:y0+224, x0:x0+224]
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
    cropped=img[x:x+256,y:y+256]
    return cropped

path='D:\\AQProject\\MSCMR2019dataset\\C0LET2_nii45_for_challenge19\\c0t2lge'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\AQProject\MSCMR2019dataset\\C0LET2_nii45_for_challenge19\\Imagesdataset\\imagevloume'  # save path image folder folder name is img; provid you folder name to saving image folder
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
    #cur_file1=os.path.join(cur_file,'Images'+'\\'+str(volume)+'.nii.gz')
    img = nib.load(cur_file)
    img = np.array(img.get_data())
    img=np.swapaxes(img,0,2) 
#    img = Image.new("I", img.T.shape)
#    im = Image.fromarray(img)
#    array_buffer = img.tobytes()
#    img.frombytes(array_buffer, 'raw', "I;16")
#    im4 = exposure.rescale_intensity(img, out_range='float')
#    img = img_as_uint(im4)
    d=1
    for test in range(img.shape[0]):
        mask=img[test,:,:]
        mask= np.uint8(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX))
        mask = cv2.equalizeHist(mask)
        #mask=np.swapaxes(mask, 1, 0)
        # boundray=trueZone(mask)
        # imgNormalized=mask[boundray[0]:boundray[1], boundray[2]:boundray[3]]
        # normalization(imgNormalized)
        #mask=cropND(mask, (128,128))
        #mask=cv2.resize(mask,(256,256))
        #mask=cropfunction(mask)
        filename = "%d.png"%d
        cv2.imwrite(os.path.join(cur_save_path,cur_save_path+'\\'+volume+ '_' + str(test) +'_image'+ '.png'), mask) # save input images
        d+=1  
        
#%
im=img[1,:,:]
#%%  save mask for every patient in folder form
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

path='D:\\AQProject\\MSCMR2019dataset\\C0LET2_nii45_for_challenge19\\c0t2lge'  # dataset path have folder tarining
# D:\Newdatasetandcodes\Maincodes2.5d\SeGThtraindata\SegTHOR2019dataset\train  in your case give the path of your system director
save_path='D:\\AQProject\MSCMR2019dataset\\C0LET2_nii45_for_challenge19\\Imagesdataset\\imagevloume'  # save path image folder folder name is img; provid you folder name to saving image folder
files=os.listdir(path)
from skimage import io
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
    img1=np.swapaxes(img,0,2)
#    label = nib.load(os.path.join(cur_file, 'GT.nii.gz'))
#    label = np.array(label.get_data())
    #label=np.swapaxes(label,0,2)
    #seg=np.swapaxes(label, 0,2)
    #vol=np.swapaxes(img,0,2)  
    # Convert to a visual format
    #vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    vol_ims=img1
    vol_ims=np.swapaxes(vol_ims, 0, 1)
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
        #cv2.imwrite(os.path.join(cur_save_path,cur_save_path+'\\'+volume.split('.')[:-1][0] + '_' + str(ii) +'_mask'+ '.png'), vol_ims[ii]*(255/3)) # save input images
        io.imsave(os.path.join(cur_save_path,cur_save_path+'\\'+volume.split('.')[:-1][0] + '_' + str(ii) +'_mask'+ '.png'), vol_ims[ii]) # save input images
        
        d+=1                 
#true_img_ = np.swapaxes(orig_img, 0, 1)