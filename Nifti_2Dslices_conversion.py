# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:26:36 2021

@author: moona
"""

#%% Nifti conversion of 2D slices 
import nibabel as nib
import cv2
import numpy as np
import os
import natsort
import natsort
import matplotlib.pyplot as plt
from tqdm import tqdm  

# affine=np.array([[-1, 0, 0, 0],
#                      [0, -1, 0, 0],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, 1]])
affine1=np.array([[-1.3671875, 0, 0, 0],
                     [0, -1.3671875, 0, 0],
                     [0, 0, 10, 0],
                     [0, 0, 0, 1]])
affine2=np.array([[-0.609985,-0.187107,5.80641,120.75],
[-0.390175, 0.142216, -9.86296, 180.063],
[-0.0849724,0.690148,3.6066,-197.386],
[0,0,0,1]])
path1='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\Testdatafolder\\testimfolder\\'
oslist=os.listdir(path1)
filesnew=natsort.natsorted(oslist)        
           
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

for i, volume in enumerate(filesnew):
    print(volume)
    cur_path = os.path.join(path1, volume)
    files=natsort.natsorted(os.listdir(cur_path))
    alist=[]
    for n, id_ in tqdm(enumerate(files), total=len(files)):
        print(n)
        print(id_)
        img=cv2.imread(os.path.join(cur_path,id_))
        # dim = (224, 224)
        # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # img=cv2.resize(img,(224,224))
        # img=resized
        print(img.shape)
        x=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        imarray = np.array(x)
        alist.append(imarray)
    tt=resize(alist)
    i=i
    #ex='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\masks/'+str(volume)+'.nii.gz'    # you need to load original GT original mask nifiti file
    #ex='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\images/'+str(volume)+'.nii.gz'    # you need to load original GT original mask nifiti file
    #img1 = nib.load(ex)
    niftiMask1 = nib.Nifti1Image(np.asarray(tt,dtype="uint8" ), affine1)  # get affine transform from original nifiti  file
    # niftiMask1.header['pixdim'][1]= img.header['pixdim'][1]
    # niftiMask1.header['pixdim'][2]= img.header['pixdim'][2]
    # niftiMask1.header['pixdim'][3]= img.header['pixdim'][3]
    #nib.save(niftiMask1,'D:\\AQProject\\completedataset2020heart\\datasetcomplete\\testmynifiti/case222_'+str(i)+'_GT.nii.gz')
    #nib.save(niftiMask1,'D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\Volumenii\\M1niinew/'+str(volume.split('.')[0])+'_pred1.nii.gz')   # save your file
    nib.save(niftiMask1,'D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\Testdatafolder\\testimfoldernii/'+str(volume)+'_Image.nii.gz')   # save your file
   
#path1='D:\\AQProject\\completedataset2020heart\\datasetcomplete\\testmynifiti\\nifti1\\Contoursfolderintra'
#oslist=os.listdir(path1)
#filesnew=natsort.natsorted(oslist)
#nifti=nifitimageconvert(filesnew)
