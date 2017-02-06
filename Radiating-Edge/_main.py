# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:58:08 2016

@author: yusufazishty
"""

import sklearn
import numpy as np
from skimage import color
import os
import cv2
import PIL
import copy
from skimage.filters import threshold_otsu, rank, threshold_yen
from skimage import morphology
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt
import scipy
import skfuzzy as fuzz
import math
from numpy.lib import recfunctions
import pandas as pd
import math
import progressbar

def RGB2XYZ(image):
    print "RGB2XYZ"
    var_R = image[:,:,0]/255.0
    var_G = image[:,:,1]/255.0
    var_B = image[:,:,2]/255.0
    #R_checker = np.zeros(image[:,:,0].shape)+0.04045
    #R_map = np.greater(var_R,R_checker)
    #processing R channel
    for i in range(var_R.shape[0]):
        for j in range(var_R.shape[1]):
            if var_R[i][j]>0.04045:
                var_R[i][j]=( ( float(var_R[i][j]) + 0.055 ) / 1.055 ) ** 2.4
            else:
                var_R[i][j]=float(var_R[i][j])/12.92
    #processing G channel
    for i in range(var_G.shape[0]):
        for j in range(var_G.shape[1]):
            if var_G[i][j]>0.04045:
                var_G[i][j]=( ( float(var_G[i][j]) + 0.055 ) / 1.055 ) ** 2.4
            else:
                var_G[i][j]=float(var_G[i][j])/12.92
    #processing B channel
    for i in range(var_B.shape[0]):
        for j in range(var_B.shape[1]):
            if var_B[i][j]>0.04045:
                var_B[i][j]=( ( float(var_B[i][j]) + 0.055 ) / 1.055 ) ** 2.4
            else:
                var_B[i][j]=float(var_B[i][j])/12.92
    #multiply by 100
    var_R = var_R*100; var_G = var_G*100; var_B = var_B*100
    #get the XYZ space
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    #create empty container for XYZ space
    XYZimage=np.zeros(image.shape)
    XYZimage[:,:,0]=copy.deepcopy(X)
    XYZimage[:,:,1]=copy.deepcopy(Y)
    XYZimage[:,:,2]=copy.deepcopy(Z)
    return XYZimage
    
    
def XYZ2LAB(imageXYZ):
    print "XYZ2LAB"
    ref_X=95.047; ref_Y=100.00; ref_Z=108.883
    X=imageXYZ[:,:,0]; Y=imageXYZ[:,:,1]; Z=imageXYZ[:,:,2]
    var_X = X/ref_X; var_Y = Y/ref_Y; var_Z = Z/ref_Z; 
    #proses channel X
    for i in range(var_X.shape[0]):
        for j in range(var_X.shape[1]):
            if var_X[i][j]>(216.0/24389.0):
                var_X[i][j]=var_X[i][j]**(1/3)
            else:
                var_X[i][j]=( (24389.0/27.0) * var_X[i][j] + 16)/116 
    #proses channel Y
    for i in range(var_Y.shape[0]):
        for j in range(var_Y.shape[1]):
            if var_Y[i][j]>(216.0/24389.0):
                var_Y[i][j]=var_Y[i][j]**(1/3)
            else:
                var_Y[i][j]=( (24389.0/27.0) * var_Y[i][j] + 16)/116 
    #proses channel Z
    for i in range(var_Z.shape[0]):
        for j in range(var_Z.shape[1]):
            if var_Z[i][j]>(216.0/24389.0):
                var_Z[i][j]=var_Z[i][j]**(1/3)
            else:
                var_Z[i][j]=( (24389.0/27.0) * var_Z[i][j] + 16)/116 
    #get CIE-Lab space
    CIE_L = (116*var_Y)-16
    CIE_a = 500*(var_X-var_Y)
    CIE_b = 200*(var_Y-var_Z)
    #create empty container for CIE-LAB
    CIE_LAB=np.zeros(imageXYZ.shape)
    CIE_LAB[:,:,0]=copy.deepcopy(CIE_L)
    CIE_LAB[:,:,1]=copy.deepcopy(CIE_a)
    CIE_LAB[:,:,2]=copy.deepcopy(CIE_b)
    return CIE_LAB
    
def RGB2GRAY(image):
    print "RGB2GRAY"
    gray = np.dot(image[:,:,:], [0.299, 0.587, 0.114])
    return gray

def resizing(image):
    m,n,z = image.shape
    m=float(m); n=float(n); z=float(z)
    if m > n:
        fy = 200/m
        fx = (round(200*n/m))/n
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    else:
        fy = (round(200*m/n))/m
        fx = 200/n
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return image
    
def paddingExpand(image):
    m,n = image.shape
    container = np.zeros((m+2,n+2))
    container[1:m+1,1:n+1] = copy.deepcopy(image)#place center
    container[1:m+1,0]=container[1:m+1,2]#pad left
    container[1:m+1,n+1]=container[1:m+1,n-1]#pad right
    container[0,1:n+1]=container[2,1:n+1]# pad top 
    container[m+1,1:n+1]=container[m-1,1:n+1]# pad bottom
    container[0,0]=container[2,2]; #pad corner top left
    container[m+1,0]=container[m-1,2]; #pad corner down left
    container[0,n+1]=container[2,n-1]; #pad corner top right
    container[m+1,n+1]=container[m-1,n-1]; #pad corner down right
    return container
    
def paddingShrink(image):
    m,n= image.shape
    return image[1:m-1,1:n-1]
    
def scaleBackImg(norm_cie_l):
    container = np.zeros(norm_cie_l.shape)
    minNormed = min(norm_cie_l.flatten())
    maxNormed = max(norm_cie_l.flatten())
    min8bit=0; max8bit=255
    for i in range(norm_cie_l.shape[0]):
        for j in range(norm_cie_l.shape[1]):
            container[i][j]=int(round((norm_cie_l[i][j]/maxNormed)*(max8bit-min8bit)))
    return container

def removeNoise(imgLabel, propsPrepared, threshold, th_cir):
    jmlhRegion = max(max(imgLabel.tolist()))
    areaImage = copy.deepcopy(imgLabel)
    m,n = imgLabel.shape
    areaImg = m*n
    
    if jmlhRegion==1:
        exit
    else:
        for i in range(jmlhRegion):
            a=propsPrepared[i]['filled_area']
            b=threshold*areaImg
            if a<b:
                print("in region "+str(i))
                print("area threshold violated!")
                print(str(a)+"<"+str(b)) 
                print("area "+str(i)+" deleted!") 
                areaImage[areaImage==i+1]=0
            if th_cir>0:
                a=(4*math.pi*propsPrepared[i]['filled_area'])/(propsPrepared[i]['perimeter']**2)
                if ( (a) < th_cir ):
                    print("in region "+str(i))
                    print("circle threshold violated!")
                    print(str(a)+"<"+str(th_cir))
                    print("area "+str(i)+" deleted!") 
                    areaImage[areaImage==i+1]=0
    print("There is only area "+str(set(areaImage.flatten().tolist())))
    return areaImage

def imgFill(imgArea):
    # Copy the thresholded image. and create mask
    im_floodfill = copy.deepcopy(imgArea)
    #im_floodfill = cv2.copyMakeBorder(im_floodfill,2,2,2,2,cv2.BORDER_REPLICATE)
    h, w = imgArea.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    #floodfill from point (0,0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)+256
    im_fill = imgArea | im_floodfill_inv
    im_fill[im_fill==255]=1
    return im_fill
    
def initialContour(areaOption, reverseSegment, oriImage, ratio, th_cir=0):
    #85 cytoplasm, 170 mucleus
    #needed_props = ['filled_area', 'perimeter', 'weighted_centroid']
    m,n = reverseSegment.shape
    if areaOption=="cytoplasm":
        #segment the cytoplasm
        imgArea = [[0 for x in range(n)] for y in range(m)]
        for i in range(reverseSegment.shape[0]):
            for j in range(reverseSegment.shape[1]):
                if reverseSegment[i][j]==85:
                    imgArea[i][j]=255
        imgArea = np.asarray(imgArea)
        #do image filling with (bwfill matlab)
        im_fill = imgFill(imgArea)
        #make a structure disk with radius 11 (strel matlab)
        disk = morphology.disk(11)
        #do an opening operation (imopen)
        imgArea = morphology.opening(im_fill, disk)
        #bwlabel
        imgLabel = label(imgArea)
        #regionprops
        props = regionprops(imgLabel, oriImage)
        propsPrepared=[]
        for i in range(len(props)):
            temp={}
            temp['filled_area']=props[i].filled_area
            temp['perimeter']=props[i].perimeter
            temp['weighted_centroid']=props[i].weighted_centroid
            propsPrepared.append(temp)
        #removeNoise
        removedNoiseImg = removeNoise(imgLabel, propsPrepared, ratio, th_cir)
        return removedNoiseImg, imgArea
    elif areaOption=="nucleus":    
        #segment the nucleus
        imgArea = [[0 for x in range(n)] for y in range(m)]
        for i in range(reverseSegment.shape[0]):
            for j in range(reverseSegment.shape[1]):
                if reverseSegment[i][j]==170:
                    imgArea[i][j]=255
        imgArea = np.asarray(imgArea)
        #do image filling with (bwfill matlab)
        im_fill = imgFill(imgArea)
        #make a structure disk with radius 11 (strel matlab)
        disk = morphology.disk(11)
        #do an opening operation (imopen)
        imgArea = morphology.opening(im_fill, disk)
        #bwlabel
        imgLabel = label(imgArea)
        #regionprops
        props = regionprops(imgLabel, oriImage)
        propsPrepared=[]
        for i in range(len(props)):
            temp={}
            temp['filled_area']=props[i].filled_area
            temp['perimeter']=props[i].perimeter
            temp['weighted_centroid']=props[i].weighted_centroid
            propsPrepared.append(temp)
        #removeNoise
        removedNoiseImg = removeNoise(imgLabel, propsPrepared, ratio, th_cir)
        return props, removedNoiseImg
        
def tabulate(mask_nu):
    tabulation = np.array([])
    allInstance = mask_nu.flatten().tolist()
    values = list(set(allInstance))
    for i in range(len(values)):
        temp = mask_nu[mask_nu==values[i]]
        rate = float(len(temp))/float(len(allInstance))*100
        wrapper = []
        wrapper.append(np.uint8(values[i]))
        wrapper.append(len(temp))
        wrapper.append(rate)        
        if i==0:
            tabulation = np.append(tabulation, wrapper)
        else:
            tabulation = np.vstack((tabulation, wrapper))
        tabulation.astype(np.uint8)
    return tabulation  

def checkZeros(value):
    if value==0:
        value+=1
        return value
    else:
        return value
    
def findContours(arrayBoundary):# only used for one valued object (ones/zeros)
    m,n = arrayBoundary.shape
    corners = []
    for i in range(0,m+1,m):
        for j in range(0,n+1,n):
            #print (i,j)
            corners.append((i,j))
    temp = corners[2]
    corners[2]=corners[3]
    corners[3]=temp
    borders=np.array([])
    for i in range(len(corners)):
        if i==3:
            a = corners[i]
            b = corners[0]
        else:
            a = corners[i]
            b = corners[i+1]
        #y = abs(b[0]-a[0]); y=checkZeros(y)
        #x = abs(b[1]-a[1]); x=checkZeros(x)
        if a[0]==b[0] and a[1]<b[1]: #ke kanan
            for j in range(a[0],b[0]+1,1):
                for k in range(a[1],b[1],1):
                  if len(borders)==0:
                      borders = np.append(borders,[j,k])
                  else:
                      borders = np.vstack((borders,[j,k]))
        if a[1]==b[1] and a[0]<b[0]: #ke bawah
            for j in range(a[0],b[0],1):
                for k in range(a[1]-1,b[1]-1+1,1):
                  if j==a[0] and k==a[1]-1:
                      continue
                  else:
                      borders = np.vstack((borders,[j,k]))
        if a[0]==b[0] and a[1]>b[1]: #ke kiri
            for j in range(a[0]-1,b[0]-1+1,1):
                for k in range(a[1]-1,b[1]-1,-1):
                  if j==a[0]-1 and k==a[1]-1:
                      continue
                  else:
                      borders = np.vstack((borders,[j,k]))
        if a[1]==b[1] and a[0]>b[0]: #ke atas
            for j in range(a[0]-1,b[0]-1,-1):
                for k in range(a[1],b[1]+1,1):
                  if j==a[0]-1 and k==a[1]:
                      continue
                  else:
                      borders = np.vstack((borders,[j,k]))
    return borders 

def singleLine(imageDenoised, pointStart, pointEnd):
    pointStart = centroids
    pointEnd = [boundaryPixel[i,0], boundaryPixel[i,1]]
    #centroid ke boundary pixel    
    y1 = pointStart[0]; x1 = pointStart[1]
    y2 = pointEnd[0]; x2 = pointEnd[1]
    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    d = int(round(d))
    
    value = np.zeros((d+1, 1))
    pos = np.zeros((d+1, 2)) # posisi garis

    value[0] = imageDenoised[y1][x1]
    value[d] = imageDenoised[y2][x2]

    pos[0][0] = y1
    pos[0][1] = x1
    pos[d][0] = y2
    pos[d][1] = x2

    #initialize the grid and points
    m,n = imageDenoised.shape
    points = np.array([])
    values = np.array([])
    for j in range(m):
        for k in range(n):
            if j== 0 and k==0:
                points = np.append(points,[j,k])
                values = np.append(values,imageDenoised[j][k])
            else:
                points = np.vstack((points,[j,k]))
                values = np.vstack((values, imageDenoised[j][k]))

    for i in range(1,d):
        xtmp = float( (x1*(d+1-i) + x2*(i-1)) )/float(d)
        ytmp = float( (y1*(d+1-i) + y2*(i-1)) )/float(d)
        
        pos[i][0] = ytmp
        pos[i][1] = xtmp

        value[i] = scipy.interpolate.griddata(points, values, (xtmp, ytmp), method='linear')[0]
    return value, pos

def sectionLine(rd, stackRefineTheta):
    for i in range(len(rd)):
        if abs(rd[i]<stackRefineTheta):
            rd[i]=0
    rd = np.array(rd)
    last = rd!=0
    last = np.argsort(last)
    last = last[-1]
    start = 0
    seg = np.array([])
    #num = 0
    
    while start<last:
        start_condition = rd!=0
        start_idx = np.argsort(start_condition)
        start = start_idx[start_condition[start_idx]==True][0]
        curSign = np.sign(rd[start])
        rd[0:start-2] = rd[start]
        check = np.sign(rd)
        Pos = check[check!=curSign]
        Pos_idx = np.argsort(Pos)
        endPos = Pos_idx[Pos[Pos_idx]==True][0] 
        rd[0:endPos]=0
        #seg[num]=(curSign, start, endPos)
        seg = np.append(seg, (curSign, start, endPos))
        #num+=1
        start = endPos+1
    return seg


def stackBasedRefine(rd, segInfo):
    positiveSection = 0
    a = segInfo.shape
    for i in range(0,a[1]):
        #positive
        if segInfo[i][0]==1:
            positiveSection = positiveSection + sum(rd[segInfo[i][1:2]])
        #negative
        elif positiveSection>0:
            j = segInfo[i][1]
            while(j<=segInfo[i][3] and positiveSection>0):
                array = rd[j] # negative
                rd[j] = min(array+positiveSection, 0)
                positiveSection = max(positiveSection+array, 0)
                j+=1
    rd2 = copy.deepcopy(rd)
    return rd2
    
def radiatingEdgeMap(imageDenoised, centroids):
    if type(centroids[0])==np.ndarray:
        centroids = centroids[0]
    m,n = imageDenoised.shape
    Y = []
    X = []
    pixelValue = []
    rem = np.zeros((m,n))
    centroids = [int(x) for x in centroids]
    arrayBoundary = np.ones((m,n))
    boundary = findContours(arrayBoundary)#g bsa pake find_contour, bikin sndiri
    boundaryPixel = copy.deepcopy(boundary)
    pixelNum = 2*(m+n)-4
    with progressbar.ProgressBar(max_value=pixelNum) as bar:
        for i in range(pixelNum):
            value, pos = singleLine(imageDenoised, centroids, [boundaryPixel[i,0], boundaryPixel[i,1]])
            #radiating difference
            rd = np.zeros(len(value),1)
            for j in range(1,len(value)-2):
                rd[j]=value[j]-value[j+1]

            stackRefineTheta = 2
            positiveSuppressRatio = 0.1
            
            segInfo = sectionLine(rd, stackRefineTheta)
            rd = stackBasedRefine(rd, segInfo)
            
            #possitive suppress
            rd1 = np.zeros((len(rd),1))
            rd1 = copy.deepcopy(rd)
            for j in range(len(rd)):
                if rd[j]>0:
                    rd1[j]=positiveSuppressRatio * rd[j]
            
            # Radiating gradient
            rg = np.zeros((len(rd),1))
            for j in range(1,len(rd)-1):
                rg[j] = ( abs(rd1[i-1])+ abs(rd1[i]) )/2

            #Normalize
            minRG = min(rg)
            maxRG = max(rg)
            
            normRG = (255*(rg-minRG))/(maxRG-minRG)
            
            if len(Y)==0 or len(X)==0:
                Y = pos[:,0]
                X = pos[:,1]
            else:
                Y = np.vstack(Y, pos[:,0])
                X = np.vstack(X, pos[:,1])
            if len(pixelValue)==0:
                pixelValue = copy.deepcopy(normRG)
            else:
                pixelValue = np.vstack(pixelValue,normRG)   
           
            #update the bar
            bar.update(i)
        
        #F = TriScatteredInterp(X,Y,pixelValue)
        
        for i in range(m):
            for j in range(n):
                rem[i][j] = F[j][i]
                scipy.interpolate.griddata((X,Y), pixelValue, (xtmp, ytmp), method='linear')[0]
                if 
        
            
    
        
curentPath=os.getcwd()
datasetPath=curentPath+"\\dataset"
allDirs=os.listdir(datasetPath)
allImage = os.listdir(datasetPath+"\\"+allDirs[0])
image = cv2.imread(datasetPath+"\\"+allDirs[0]+"\\"+allImage[1])
#pre process resizing
image = resizing(image)
#pre process convert to cielab, then grayscale only the L space after normalization
imageCIE_LAB = color.rgb2lab(image)
cie_asli=copy.deepcopy(imageCIE_LAB)
cie_l=cie_asli[:,:,0]
min_l=min(cie_l.flatten())
max_l=max(cie_l.flatten())
if min_l != max_l:
    norm_cie_l = (cie_l-min_l)/(max_l-min_l)
norm_cie_ll = scaleBackImg(norm_cie_l)
#padding dengan yang paling pinggir, top-bottom, left-right, corner-corner, median filter
imageGray = paddingExpand(norm_cie_ll)
imageGray = imageGray/255.0
imageGray = scaleBackImg(imageGray)
imageGray = imageGray.astype(np.int)
imageDenoised = scipy.signal.medfilt(imageGray,5)
imageDenoised = imageDenoised.astype(np.int)
cv2.imwrite("imgDenoised.jpg", imageDenoised)
#boundMirrorShrink
imageDenoised = paddingShrink(imageDenoised)
imageFilt = copy.deepcopy(imageDenoised)
#Rough segmentation part, fcm
m,n = imageDenoised.shape
imageDenoised = imageDenoised.astype(np.float)
im_rseg = imageDenoised.flatten()
im_rseg = im_rseg.reshape(1,im_rseg.shape[0])
clusterNum = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(im_rseg, clusterNum, 2, 
                                                 error=0.005, maxiter=10000, init=None)
cluster_idx = np.argmax(u, axis=0)#axis=0 based on column
argsrtCntr = np.argsort(cntr.T)[0]#outer to inner, this means background to nucleus
valueCluster=[85,170,255]
for i in range(len(argsrtCntr)):
    cluster_idx[cluster_idx==argsrtCntr[i]]=valueCluster[i]
cluster_idx = np.reshape(cluster_idx, imageDenoised.shape)
cluster_idx_reverse = 255-cluster_idx
cluster_idx_reverse = cluster_idx_reverse.astype(np.uint8)
#85 cytoplasm, 170 nucleus
removedNoiseImgCyto, imgArea = initialContour('cytoplasm', cluster_idx_reverse, imageDenoised, 0.2)
regionProps, removedNoiseImgNucleus = initialContour('nucleus', cluster_idx_reverse, 
                                                     imageDenoised, 0.005, 0.7)
#select largest cytoplasm
#cv2.imwrite("removedNoiseImgCyto.jpg", removedNoiseImgCyto*255)
#cv2.imwrite("removedNoiseImgNucleus.jpg", removedNoiseImgNucleus*255)
#select largest cytoplasm
mask_cyto = imgFill(removedNoiseImgCyto.astype(np.uint8))-256#imgFill minta uint8, dikurangi karena ada nambah
mask_cyto = mask_cyto.astype(np.float)#skimage minta float
boundaries_cyto = find_contours(mask_cyto, 0)
#deal with  nucleus
mask_nu = copy.deepcopy(removedNoiseImgNucleus)
mask_nu[mask_cyto==0]=0
tabu = tabulate(mask_nu)
Index1 = tabu[tabu[:,0]!=0]
Index1 = Index1[:,0]
if len(Index1)>1:
    Index1 = [int(x) for x in Index1]
else:
    Index1 = int(Index1)
aha = tabu[Index1,:]
aha = np.reshape(aha, (1,len(aha)))
stat = pd.DataFrame(data=aha)
stat = stat.rename(columns = {0:"label", 1:"amount", 2:"rate"})
stat = stat.sort(columns="amount", ascending=False)
num_nu = stat.shape[0]
centroids = np.zeros((num_nu, 2))
boundaries_nu={}
for i in range(num_nu):
    mat_help = copy.deepcopy(mask_nu)
    mat_help[mat_help != stat.ix[i,"label"]]=0
    centroids[i,0]=regionProps[int(stat.ix[i,"label"])].weighted_centroid[0]
    centroids[i,1]=regionProps[int(stat.ix[i,"label"])].weighted_centroid[1]
    tmp = find_contours(mat_help,0)[0]
    boundaries_nu[i]=tmp

#radiating edge map
img_edgemap = radiatingEdgeMap(imageDenoised, centroids)
    