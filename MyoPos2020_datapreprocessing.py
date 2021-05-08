# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:20:46 2021

@author: Abdul Qayyum
"""


#%%  ##################### MyPos 2020 data preprocessing ####################

import glob
import nibabel
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf

#Read the format of the original nii data: the maximum and minimum of the elements in each case and the shape of the case, and cut the center area 160*160 and save it as npy
def read_data_format(file_dir,save_dir):
    imgname = glob.glob(file_dir + '\*' + '.nii.gz')
    for file_name in imgname:
        midname = file_name[file_name.rindex("\\") + 1:]
        img = nibabel.load(file_dir + midname).get_data()
        img_max = np.amax(img)
        img_min = np.amin(img)
        print('%s: max is %d, min is %d' %(midname, img_max, img_min))
        print(img.shape)
        img = img[int((img.shape[0]-256)/2):int((img.shape[0]+256)/2), int((img.shape[1]-256)/2):int((img.shape[1]+256)/2)]
        print(img.shape)
        np.save(save_dir+midname[0:-7], img)

file_dir='D:\\MICCAI2020datasets\\train25\\train25_myops_gd\\train25_myops_gd\\'
save_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\imgnumpy\\'

read_data_format(file_dir,save_dir)


def read_data_formatDE(file_dir,save_dir):
    imgname = glob.glob(file_dir + '\*' + '.nii.gz')
    for file_name in imgname:
        midname = file_name[file_name.rindex("\\") + 1:]
        img = nibabel.load(file_dir + midname).get_data()
        img_max = np.amax(img)
        img_min = np.amin(img)
        print('%s: max is %d, min is %d' %(midname, img_max, img_min))
        print(img.shape)
        img = img[int((img.shape[0]-256)/2):int((img.shape[0]+256)/2), int((img.shape[1]-256)/2):int((img.shape[1]+256)/2)]
        print(img.shape)
        np.save(save_dir+midname[0:-7], img)
        
file_dir1='D:\\MICCAI2020datasets\\train25\\train25\\'
save_dir1='D:\\MICCAI2020datasets\\train25\\savenumpy\\imgnumpyDE\\'

read_data_formatDE(file_dir1,save_dir1)        
        
#Filter out the labeled slices (preliminary screening) The data given in the 2020 Myocardial Segmentation Challenge are all valid labels, in fact, there is no need to filter
def select_slice(file_dir, label_dir, save_file_dir, save_label_dir):
    imglist = []
    labellist = []
    imgname = glob.glob(label_dir+'\\*'+'.npy')
    for file_name in imgname:
        j = 0
        midname = file_name[file_name.rindex("\\") + 1:]
        print(midname)
        img = np.load(file_dir + midname[0:-6]+'C0.npy')
        label = np.load(label_dir + midname)
        print(img.shape)
        newimg = np.zeros((img.shape[0], img.shape[1], img.shape[2]),dtype='float32')
        newlabel = np.zeros((img.shape[0], img.shape[1], label.shape[2]),dtype='float32')
        for i in range(label.shape[2]):#Filter the current slices whose sum is greater than 0
            if np.sum(label[:, :, i:i+1]) > 0:
                newimg[:, :, j:j+1] = img[:, :, i:i+1]
                newlabel[:, :, j:j+1] = label[:, :, i:i+1]
                j = j + 1
        finalimg = newimg[:, :, 0: j]
        finallabel = newlabel[:, :, 0: j]
        print(finalimg.shape)
        imglist.append(finalimg)
        labellist.append(finallabel)
    img_save = np.concatenate(imglist, axis=-1)
    print(img_save.shape)
    label_save = np.concatenate(labellist, axis=-1)
    print(label_save.shape)
    np.save(save_file_dir + 'C0_img.npy', img_save)
    np.save(save_label_dir + 'label.npy', label_save)

file_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\imgnumpyDE\\'
label_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\imgnumpy\\'
save_file_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\imageC0\\'
save_label_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\labelC0\\'

select_slice(file_dir, label_dir, save_file_dir, save_label_dir)


#Reset label value and one-hot
def reset_value(label):
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            for k in range(label.shape[2]):
                if label[i][j][k] == 200:
                    label[i][j][k] = 1
                elif label[i][j][k] == 500:
                    label[i][j][k] = 2
                elif label[i][j][k] == 600:
                    label[i][j][k] = 3
                elif label[i][j][k] == 1220:
                    label[i][j][k] = 4
                elif label[i][j][k] == 2221:
                    label[i][j][k] = 5
                else:
                    label[i][j][k] = 0
    newlabel = to_categorical(label, num_classes=6)
    return newlabel

pathlabel=np.load('D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\labelDE\\label.npy')

new_label=reset_value(pathlabel)
np.save('D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\labelDE'+'newlabel.npy', newlabel)
imd1=pathlabel[:,:,35]

# Myocardium labels
def reset_value1(label):
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            for k in range(label.shape[2]):
                if label[i][j][k] == 200:
                    label[i][j][k] = 1
                elif label[i][j][k] == 1220:
                    label[i][j][k] = 1
                elif label[i][j][k] == 2221:
                    label[i][j][k] = 1
                else:
                    label[i][j][k] = 0
    return label

new_labels1=reset_value1(pathlabel)

np.save('D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\labelDE'+"\\"+'newlabelnesw11.npy', new_labels1)

#Convert the generated numpy to jpg and save it
def test_npy(file_dir,save_dir):
    npy = np.load(file_dir)
    #npy = np.argmax(npy, axis=-1)
    d=204
    for i in range(npy.shape[2]):
      img = npy[:,:,i:i+1]
      img = array_to_img(img)
      img.save(save_dir+'patient%d.jpg'%d)
      d+=1

file_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\imageC0\\C0_img.npy'
save_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\imageC0png\\'
test_npy(file_dir, save_dir)

file_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\imageT2\\T2_img.npy'
save_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\imageT2png\\'
test_npy(file_dir, save_dir)



file_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\labelDE\\newlabelnesw11.npy'
save_dir='D:\\MICCAI2020datasets\\train25\\savenumpy\\DEdataset\\Myopng\\'
test_npy(file_dir, save_dir)
#%%import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import nibabel
import scipy.io as io
import SimpleITK as sitk


def jpg2npy(file_dir,save_dir):
    i=0
    imgs = glob.glob(file_dir + '/*.jpg')
    #imgdatas = np.ndarray((15,512,512,1))
    imgdatas = np.ndarray((1232,128,128,1))
    #imgdatas = np.ndarray((15,400,400,1))
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        print(midname)
        img = load_img(file_dir + "/" + midname, grayscale=True)
        img = img_to_array(img)
        print(str(img.shape))
        for x in range(128):
            for y in range(128):
                if img[:,:,0][x][y]>0:
                    imgdatas[i,:,:,:][x][y]=1
        #imgdatas[i,:,:,:] = img
        i += 1
    print(i)
    np.save(save_dir+'train_mask2.npy', imgdatas)


def nii2npy(path1, path2):
    N = 45
    if not os.path.exists(path2):
        os.makedirs(path2)
    for n in range(N):
        print('Processing File ' + str(n + 1))
        filename1 = 'patient' + str(n + 1) + '_CO' + '.nii.gz'
        directory1 = os.path.join(path1, filename1)
        filename2 = 'patient' + str(n + 1) + '_LCO' + '.npy'
        file1 = os.path.join(path1, filename1)
        data = nibabel.load(file1).get_data()
        print('  Data shape is ' + str(data.shape) + ' .')
        file2 = os.path.join(path2, filename2)
        np.save(file2, data)
        print('File ' + 'patient' + str(n + 1) + '_LGE.' + ' is saved in ' + file2 + ' .')


def npy2jpg():
    for j in range(1):
        imgs = np.load('../c0gt_npy/test/patient36_C0.npy'%(j+1))

        for i in range(imgs.shape[2]):
            img = np.expand_dims(imgs[:, :, i], axis=2)
            img1 = array_to_img(img)
            img1.save("../c0gt_npy/patient%d_C0_%d.jpg"%(j+1,i+1))
#npy2jpg()


def tojpg():
    img= np.load('C:/Users/fly/Downloads/MDFA_Net_c0200f_result.npy')
    #img = img[0:10,:,:,:]
    for i in range(img.shape[0]):
        x = img[i,:,:,:]
        print(x.shape)
        y =array_to_img(x)
        y.save('C:/Users/fly/Downloads/test/patient%d.jpg'%i)
#tojpg()


def npy2mat(file_dir,save_dir):
    i=0
    imgs = glob.glob(file_dir + '*.npy')
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        print(midname)
        img = np.load(file_dir + midname)
        io.savemat(save_dir+midname[:-4]+'.mat', {'data': img})


def jpg2nii(file_dir,save_dir):
    img = glob.glob(file_dir + '*.jpg')
    for imgname in img:
        midname = imgname[imgname.rindex("/") + 1:]
        img = load_img(file_dir + midname)
        img_npy = img_to_array(img)
        img_npy = sitk.GetImageFromArray(img_npy)
        sitk.WriteImage(img_npy, save_dir + midname[0:-4] + '.nii')