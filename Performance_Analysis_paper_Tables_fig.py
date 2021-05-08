# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:29:17 2021

@author: Abdul Qayyum
"""

#%%
import numpy as np
import scipy.misc  
import cv2
import numpy
import glob
import pylab as plt
import os
from PIL import Image, ImageOps
#from glob import glob
#
def getLargestCC1(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
from skimage import io, exposure, img_as_uint, img_as_float
#FoldersPath='D:\\AQProject\\latestdataset\\updateddataset'
#FolderList=os.listdir(FoldersPath)
folders = glob.glob('D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\M8\\*')
#foldername='D:\\AQProject\\MyDatasetlatest\\GTmask'
#folders = glob.glob('D:\\AQProject\\MyDatasetlatest\\Dataset\\dataset\\')
#imagenames__list = []
#img_mask = 'D:\\AQProject\\imagedata/*.png'
#img_names = glob(folders)
#imgnames = sorted(glob.glob("/PATH_TO_IMAGES/*.png"))
foldername='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\M'
import os
import natsort
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
d = 1
for Folder in folders:
    #print(Folder.split('\\')[-1])
    #d=1
    SubFolders=glob.glob(Folder+'/*.png')
    for f in natsort.natsorted(SubFolders):
        print(f.split('\\')[-2])
        imge= cv2.imread(f)
        #img=Image.open(f)
        # old_size = img.size
        # new_size = (128, 128)
        # im=img
        #imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
        seg=imge
        seg1=getLargestCC1(seg)
        seg2=np.array(seg1).astype(np.uint8)
        seg2 = cv2.cvtColor(seg2, cv2.COLOR_BGR2GRAY)
        imge=seg2
        #imge[imge>0]=1
        
        # new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
        # new_im.paste(img, ((new_size[0]-old_size[0])//2,
        #               (new_size[1]-old_size[1])//2))
        filename = "%d.png"%d
        f=Folder.split('\\')[-1]
        p1=f[:]
        #new_im1=np.array(new_im)
        path_file=foldername+p1
        save_path=os.path.join(foldername,p1)
        createFolder(save_path)
#        save_path=os.path.join('D:\\',*patients.split('\\')[1:-2])
#        #cv2.imwrite(saveslices+'\\'+filename, total) # save images
#        #cv2.imwrite(saveslices+'\\'+filename, total*(255/3)) # save images
        #out_path=save_path+'\\'+filename
        #filename = "images/file_%d.jpg"%d
        #cv2.imwrite(out_path, new_im1) # save imag
            #cv2.imwrite(foldername+'\\'+filename,im)
        io.imsave(save_path+'\\'+filename, imge) # save images
        #new_im.save(save_path+'\\'+filename)
        d+=1

#%%
import nibabel as nib
import cv2
import numpy as np
import os
import natsort
import natsort
import matplotlib.pyplot as plt
from tqdm import tqdm  
#result_array = np.empty((512, 512,3))
#
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
affine=np.array([[-1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
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


#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 07:01:01 2020

@author: Abdul Qayyum
"""

#%% evluates metrics 
import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr
import os
import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

# code
def dc(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    
    intersection = numpy.count_nonzero(result & reference)
    
    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 1.
    
    return dc


def precision(result, reference):

    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
        
    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)
    
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    
    return precision

def recall(result, reference):

    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
        
    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    
    return recall

def sensitivity(result, reference):
    
    return recall(result, reference)

def specificity(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
       
    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    
    return specificity


def hd(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def assd(result, reference, voxelspacing=None, connectivity=1):
    assd = numpy.mean( (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)) )
    return assd

def asd(result, reference, voxelspacing=None, connectivity=1):
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd
    
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == numpy.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def volumeofff(ct_array,seg_array):
    voe = (1. - np.logical_and(ct_array, seg_array).sum() / float(np.logical_or(ct_array, seg_array).sum()))
    return voe

def volume(ndarray, voxel_spacing):
    volume=np.prod(voxel_spacing)*(ndarray.sum())
    return volume
#vd = 100 * (segm.sum() - gt.sum()) / float(gt.sum())

#%%
# import metrics

import numpy as np
import os
import shutil
import SimpleITK as sitk
import scipy.ndimage as ndimage
#pathGT="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\GT"
# pathGT="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\Volumenii\\Gtmasknii"
# # #pathPrediction="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\M1"
# pathPrediction="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\Volumenii\\M7nii"

# pathGT='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\GTnii'
# pathPrediction="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\M4predn"


pathGT='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\GTnii'
pathPrediction="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\M1pred1"


# pathGT="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\GT"
# pathPrediction="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\M2pr"



dice=[]
HD=[]
volumeDifference=[]
volumeDifferenceRate=[]
volumePrediction=[]
volumeGT=[]
for filePrediction in os.listdir(pathPrediction):
    print(filePrediction)
    #  load prediction mask as a nifiti, you can use nib.load as well for nifti
    prediction = sitk.ReadImage(os.path.join(pathPrediction, filePrediction), sitk.sitkInt16) 
    #  the prediction mask array should be one hot format
    predArray = sitk.GetArrayFromImage(prediction)  # convert into numpy array

    # load GT mask. 
    # You should modify the GT file name if its name is different to the prediction file
    GT = sitk.ReadImage(os.path.join(pathGT, filePrediction.replace('pred','GT')), sitk.sitkInt8) 
    GTArray = sitk.GetArrayFromImage(GT)
    spacing=GT.GetSpacing()
    #print(GTArray.shape)
    #print(predArray.shape)
    dice.append(dc(predArray, GTArray))
    aVolumePred=volume(predArray, spacing)
    aVolumeGT=volume(GTArray, spacing)
    volumePrediction.append(aVolumePred)
    volumeGT.append(aVolumeGT)
    volumeDifference.append(abs(aVolumePred-aVolumeGT))
    #print(volumeDifference)
    HD.append(hd(predArray, GTArray))

A=np.array(volumePrediction)
B=np.array(volumeGT)

vd = 100*(A.sum() - B.sum()) / float(B.sum()) # absolute volume difference

avgDice = float(sum(dice))/len(dice)
print(avgDice)
avgVD= float(sum(volumeDifference))/len(volumeDifference)
print(avgVD)
avgHd= float(sum(HD))/len(HD)
print(avgHd)
# print(vd)
std=numpy.std(dice)

# NAVD=volumeDifference/volumeGT
# result = [a/b for a,b in zip(volumeDifference,volumeGT)]

# A=np.array(volumePrediction)
# B=np.array(volumeGT)

# vd = 100*(A.sum() - B.sum()) / float(B.sum()) # absolute volume difference

myInt = 1000
newListGT = [i/myInt for i in volumeGT]
newListPred=[i/myInt for i in volumePrediction]





    
    
    