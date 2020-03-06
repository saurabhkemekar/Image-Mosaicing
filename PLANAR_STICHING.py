import cv2
import numpy as np
import os
import math
def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    print('flag.shape----------',flag.shape)
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img


#----------------------------------------------------------------------------------------------------------------------------------------------
noi = 4 # number of images to be stich
img1 = cv2.imread('1.jpg')
img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
hieght  = img1.shape[0] + 200
img3 = img1.copy()
canvas = np.zeros((img1.shape[0],img1.shape[1],img1.shape[2]),np.uint8)
print('img1.shape',img1.shape)
print('canvas.shape------',canvas.shape)
canvas[0:img1.shape[0],0:img1.shape[1]] = img1
temp = np.zeros((200,img1.shape[1],3),np.uint8)
print('temp.shape----',temp.shape)
canvas =np.vstack((canvas,temp))
cv2.imshow('can',canvas)
cv2.waitKey(0)
for i in range(1,noi):
        img1 = img3
        img2 = cv2.imread(str(i + 1) + '.jpg')
        img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1,desc1 = sift.detectAndCompute(gray1,None)
        kp2,desc2 = sift.detectAndCompute(gray2,None)
        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1,desc2,k=2)
        good = []
        pt1 = []
        pt2 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        good = sorted(good,key = lambda x:x.distance)
        good = good[:15]
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        # M,mask = cv2.estimateAffine2D(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=0.4)
        # img3 = cv2.warpAffine(img2,M,(gray2.shape[1]+gray1.shape[1],hieght))
        # cv2.imshow('img3',img3)
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        H = np.linalg.inv(H)
        img3 = cv2.warpPerspective(img2,H,(gray2.shape[1]+gray1.shape[1],hieght))
        print('img3.shape',img3.shape)
        temp = np.zeros_like(img3)
        col = img3.shape[1]-canvas.shape[1]
        temp = cv2.resize(temp,(col,img3.shape[0]),interpolation = cv2.INTER_CUBIC)
        print('shape',temp.shape,canvas.shape)
        canvas = np.hstack((canvas,temp))
        print(canvas.shape, img3.shape)
        canvas = image_stiching(canvas,img3)
        cv2.imshow('canvas',canvas)
        cv2.waitKey(0)


cv2.waitKey(0)
cv2.destroyAllWindows()
