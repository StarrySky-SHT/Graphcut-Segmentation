# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import maxflow
from PIL import Image
from matplotlib import pyplot as plt
from pylab import *
import cv2
from sklearn.mixture import GaussianMixture
from roipoly import RoiPoly

# 输入的是Mask包括前景和背景
def GMMdistribution(I, Mfore, Mback):
    Mfore = np.uint8(Mfore)*127
    Mback = np.uint8(Mback)*255
    Mfore_re = np.reshape(Mfore,[Mfore.shape[0]*Mfore.shape[1],1])
    Mback_re = np.reshape(Mback,[Mback.shape[0]*Mback.shape[1],1])
    I_re = np.reshape(I,[I.shape[0]*I.shape[1],3]) 
    Fdata = I_re[np.nonzero(Mfore_re)[0],:]
    Bdata = I_re[np.nonzero(Mback_re)[0],:]
    Fg_gmm = (GaussianMixture(n_components=3,reg_covar = 0.1).fit(Fdata).score_samples(I_re))
    bg_gmm = (GaussianMixture(n_components=3,reg_covar = 0.1).fit(Bdata).score_samples(I_re))
    return Fg_gmm , bg_gmm

def graph(I, 
fseeds, 
bseeds,
k, 
s, 
fProb, #  Mask with size equal to I (M*N,1)
bProb): #  Mask with size equal to I (M*N,1)
    fseeds = np.uint8(fseeds)*127
    bseeds = np.uint8(bseeds)*255
    if len(I.shape)==2:
        channels = 2
    else:
        channels = 3
    
    M,N = I.shape[0],I.shape[1]

    # t-links
    maxWeights_t = 1e6
    g,pic = maxflow.Graph[int](M*N,2*M*N),maxflow.Graph[int]() # define the graph
    g.add_nodes(M*N)
    for i in range(0,M):
        for j in range(0,N):
            label_f = fseeds[i,j]
            label_b = bseeds[i,j]
            if (label_f == 127) & (label_b != 255):
                s_weight = maxWeights_t
            elif (label_f != 127) & (label_b == 255):
                t_weight = maxWeights_t
            else:
                forePosibility = fProb[i*N+j]
                backPosibility = bProb[i*N+j]
                s_weight = -(backPosibility)
                t_weight = -(forePosibility)
            pindex_t = i*N+j
            g.add_tedge(pindex_t,s_weight,t_weight)

    maxWeights_n = -1e20
    # n-links
    for i in range(0,M):
        for j in range(0,N):
            upperPointx = i
            upperPointy = j-1 
            leftPointx = i-1 
            leftPointy = j
            n_weights = 0
            if (upperPointy >= 0) & (upperPointy<N):
                square_diff = 0
                for k in range(0,channels):
                    pValue = I[i,j,k]
                    qValue = I[upperPointx,upperPointy,k]
                    differ = float(int(pValue) - int(qValue))
                    square_diff = square_diff + differ**2
                n_weights = k*np.exp(-square_diff/(2*s*s))
                pindex_n = i*N+j
                qindex_n = upperPointx*N+upperPointy
                g.add_edge(pindex_n, qindex_n, n_weights,n_weights)

            if (leftPointx >= 0) & (leftPointx<N):
                square_diff = 0
                for k in range(0,channels):
                    pValue = I[i,j,k]
                    qValue = I[upperPointx,upperPointy,k]
                    differ = float(int(pValue) - int(qValue))
                    square_diff = square_diff + differ**2
                n_weights = k*np.exp(-square_diff/(2*s*s))
                pindex_n = i*N+j
                qindex_n = leftPointx*N+leftPointy
                g.add_edge(pindex_n, qindex_n, n_weights,n_weights)
            if n_weights>maxWeights_n:
                maxWeights_n = n_weights

    flow = g.maxflow()
    labels = np.zeros((M,N))
    for i in range(0,M):
        for j in range(0,N):
            labels[i,j] = g.get_segment(int(i*N+j))
    return labels,flow

I = cv2.imread('hat.jpg')
I = I[:,:,[2,1,0]]
plt.imshow(I)
plt.title('draw foremask,click left button to connect \n and click right button to complete selection')
my_roi_fore = RoiPoly(color='r') # draw new ROI in red color
mask_fore = my_roi_fore.get_mask(I[:, :, 0])

plt.imshow(I)
plt.title('draw the first backmask,click left button to connect \n and click right button to complete selection')
my_roi_back = RoiPoly(color='b') # draw new ROI in blue color
mask_back = my_roi_back.get_mask(I[:, :, 0])
plt.imshow(I)
plt.title('draw the second backmask,click left button to connect \n and click right button to complete selection')
my_roi_back1 = RoiPoly(color='b') # draw new ROI in blue color
mask_back1 = my_roi_back1.get_mask(I[:, :, 0])
plt.imshow(I)
plt.title('draw the third backmask,click left button to connect \n and click right button to complete selection')
my_roi_back2 = RoiPoly(color='b') # draw new ROI in blue color
mask_back2 = my_roi_back2.get_mask(I[:, :, 0])
plt.imshow(I)
plt.title('draw the forth backmask,click left button to connect \n and click right button to complete selection')
my_roi_back3 = RoiPoly(color='b') # draw new ROI in blue color
mask_back3 = my_roi_back3.get_mask(I[:, :, 0])
mask_back = mask_back+mask_back1+mask_back2+mask_back3
Fg_gmm, Bg_gmm = GMMdistribution(I, mask_fore, mask_back)
labels,flow = graph(I, mask_fore, mask_back, 20, 20, Fg_gmm, Bg_gmm )
labels = 1-labels

# mask_overlay = cv2.fillPoly(labels,)
plt.imshow(labels)
plt.title('mask result')
plt.show()


mask = np.zeros((I.shape[0],I.shape[1],3))
mask[:,:,0] = 128*labels
mask[:,:,1] = 128*labels

mask = np.uint8(mask)
I = cv2.addWeighted(I,0.7,mask,0.7,0)
plt.imshow(I)
plt.title('final segmentaion result')
plt.show()

print('finished')
