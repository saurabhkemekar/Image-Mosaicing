import cv2
import numpy as np
import os
import math
os.chdir('/home/saurabh/PANORAMA/yosemite_test')

def required_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    image,contour,heic = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contour[0]
    cnt = cnt.reshape(-1,2)
    x = np.max(cnt[:,0],axis = 0)
    y = np.max(cnt[:,1],axis = 0)
    img2 = np.zeros((y,x,3),np.uint8)
    img2 = img[0:y,0:x]
    cv2.waitKey(0)
    return img2

def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
 #   print('flag.shape----------',flag.shape)
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img
#---------------------------------------------------------------------------------------------------------------------------------------------
def cylindrical_warp(img,K):
    foc_len = (K[0][0] +K[1][1])/2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
    # print('img p=color',img[0,0])
    theta= (x- K[0][2])/foc_len # angle theta
    h = (y-K[1][2])/foc_len # height
    p = np.array([np.sin(theta),h,np.cos(theta)])
    p = p.T
    p = p.reshape(-1,3)
    image_points = K.dot(p.T).T
    points = image_points[:,:-1]/image_points[:,[-1]]
    points = points.reshape(img.shape[0],img.shape[1],-1)
    cylinder = cv2.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR)
    return cylinder
#----------------------------------------------------------------------------------------------------------------------------------------------
noi = 4 # number of images to be stich
img1 = cv2.imread('1.jpg')
hieght  = img1.shape[0] + 200
k = np.array([[583.61969059, 0, 327.46517597], [0, 582.90936082, 251.76312815], [0, 0, 1]])
img3 = img1.copy()
print(img3.shape)
xyz = []
bgr= []
canvas = np.zeros((img1.shape[0],img1.shape[1],img1.shape[2]),np.uint8)
canvas[0:img1.shape[0],0:img1.shape[1]] = img1
temp = np.zeros((200,img1.shape[1],3),np.uint8)
canvas =np.vstack((canvas,temp))
print('-----------------------------------------------------------------------------------------------------------------------------------')
img1 = cylindrical_warp(img1,k)
cv2.waitKey(0)
for i in range(1,noi):
        img1 = img3
        img2 = cv2.imread(str(i + 1) + '.jpg')
        img2 = cylindrical_warp(img2,k)
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
        M,mask = cv2.estimateAffine2D(pt2,pt1,cv2.RANSAC,ransacReprojThreshold=0.4)
        img3 = cv2.warpAffine(img2,M,(gray2.shape[1]+gray1.shape[1],hieght))
        temp = np.zeros_like(img3)
        col = img3.shape[1]-canvas.shape[1]
        temp = cv2.resize(temp,(col,img3.shape[0]),interpolation = cv2.INTER_CUBIC)
        canvas = np.hstack((canvas,temp))
        canvas = image_stiching(canvas,img3)
        canvas = required_img(canvas)
        cv2.imshow('canvas',canvas)
        hieght = canvas.shape[0]
        cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
