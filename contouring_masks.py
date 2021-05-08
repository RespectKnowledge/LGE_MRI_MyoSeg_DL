# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:48:46 2021

@author: Abdul Qayyum
"""
#%% Draw contorus on the image
from PIL import Image, ImageFilter
import numpy as np

def drawContour(m,s,c,RGB):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p:p==c and 255)
    # DEBUG: thisContour.save(f"interim{c}.png")

    # Find edges of this contour and make into Numpy array
    thisEdges   = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN  = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m

## Load segmented image as greyscale
#seg = Image.open(path11).convert('L')
#
## Load main image - desaturate and revert to RGB so we can draw on it in colour
#main = Image.open(path).convert('L').convert('RGB')
#mainN = np.array(main)
#
##mainN = drawContour(mainN,seg,128,(255,0,255))   # draw contour 1 in red
#mainN = drawContour(mainN,seg,255,(255,255,255)) # draw contour 2 in yellow
##plt.imshow(mainN )
## Save result
#Image.fromarray(mainN).save('result12.png')
#
#cv2.imwrite('resultbbb.png',mainN) 


import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
saveslices="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\visluzation\\overlaidimages\\M6overlaid"
dir_data = "D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\visluzation"
dir_seg = dir_data + "/M66/"
dir_img = dir_data + "/GTimage/"
#def imgcolor(img,color,shape):
#    img=img.reshape((-1,3))
#    img=np.multiply(img, color)
#    img=img.reshape((shape[0],shape[1],3))
#    return img
#check labels of all classes
import natsort
directorylist=os.listdir(dir_img)
#directorylist.sort()
d=1
for segm in (natsort.natsorted(directorylist,reverse=False)):
    #seg = cv2.imread(dir_seg + segm ) # (360, 480, 3)
    seg = Image.open(dir_seg + segm).convert('L')
    #imagedata=cv2.imread(dir_img+segm)
    main = Image.open(dir_img+segm).convert('L').convert('RGB')
    mainN = np.array(main)
    #imagedata1=cv2.resize(imagedata,(224,224))
    #img = resizeimage.resize_cover(imagedata, [224, 224])
    #fullimage= cv2.add(imagedata, seg )
    mainN = drawContour(mainN,seg,255,(200, 169, 221)) # draw contour 2 in yellow
    filename = "%d.png"%d
        #cv2.imwrite(saveslices+'\\'+filename, total * (255/3)) # save images
    #cv2.imwrite(saveslices+'\\'+str(segm.split('.')[0])+".png",fullimage) # save images
    cv2.imwrite(saveslices+'\\'+filename,mainN) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
    d+=1      
#    255,228,181
    #139,69,19
    #(30,144,255)
    #139,0,139
    #180, 251, 184
    #35, 169, 221
    #120, 0 , 255
    #255, 255, 195
    #0, 34, 255
    #140, 85, 180
    #12,180,100
    #0, 255, 255 yellow color
#https://www.rapidtables.com/web/color/RGB_Color.html#color-table
#https://www.webucator.com/blog/2015/03/python-color-constants-module/
#%%
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
saveslices="D:\\Dr.Alainconferencepaper\\predictionresults\\predictionresults\\preds_unetbcardiac20intra11"
dir_data = "D:\\Dr.Alainconferencepaper\\predictionresults"
dir_img = dir_data + "/preds_unetbcardiac20intra1/"
#def imgcolor(img,color,shape):
#    img=img.reshape((-1,3))
#    img=np.multiply(img, color)
#    img=img.reshape((shape[0],shape[1],3))
#    return img
#check labels of all classes
import natsort
directorylist=os.listdir(dir_img)
#directorylist.sort()
d=1
for segm in (natsort.natsorted(directorylist,reverse=False)):
    imagedata=cv2.imread(dir_img+segm)
    imgdata=imagedata[:,:,2]
    imgdata[imgdata>0]=1
    #imagedata1=cv2.resize(imagedata,(224,224))
    #img = resizeimage.resize_cover(imagedata, [224, 224])
    #fullimage= cv2.add(imagedata, seg )
    filename = "%d.png"%d
        #cv2.imwrite(saveslices+'\\'+filename, total * (255/3)) # save images
    #cv2.imwrite(saveslices+'\\'+str(segm.split('.')[0])+".png",fullimage) # save images
    cv2.imwrite(saveslices+'\\'+filename,imgdata*255) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
    d+=1   
#%%
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
saveslices="D:\\Dr.Alainconferencepaper\\predictionresults\\predictionresults\\preds_unetbtest111"
dir_data = "D:\\Dr.Alainconferencepaper\\predictionresults"
dir_img = dir_data + "/preds_unetbtest11/"
#def imgcolor(img,color,shape):
#    img=img.reshape((-1,3))
#    img=np.multiply(img, color)
#    img=img.reshape((shape[0],shape[1],3))
#    return img
#check labels of all classes
import natsort
directorylist=os.listdir(dir_img)
#directorylist.sort()
d=1
for segm in (natsort.natsorted(directorylist,reverse=False)):
    imagedata=cv2.imread(dir_img+segm)
    imgdata=imagedata[:,:,2]
    #imgdata[imgdata==254]=1
    imgdata[imgdata>=254]=1
    #imagedata1=cv2.resize(imagedata,(224,224))
    #img = resizeimage.resize_cover(imagedata, [224, 224])
    #fullimage= cv2.add(imagedata, seg )
    filename = "%d.png"%d
        #cv2.imwrite(saveslices+'\\'+filename, total * (255/3)) # save images
    #cv2.imwrite(saveslices+'\\'+str(segm.split('.')[0])+".png",fullimage) # save images
    cv2.imwrite(saveslices+'\\'+filename,imgdata*255) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
    d+=1       