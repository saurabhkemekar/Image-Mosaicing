import cv2
import numpy as np
import os
#os.chdir('/home/saurabh/PANORAMA/3')

noi = 4 # number of images to be stich
img1 = cv2.imread('1.jpeg')
img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
hieght  = img1.shape[0] + 50
k = np.array([[583.61969059, 0, 327.46517597], [0, 582.90936082, 251.76312815], [0, 0, 1]])
img3 = img1.copy()
img1,A = cylindrical_warp(img1,k) 
xyz.append(A)

for i in range(1,noi):
        img1 = img3
        img2 = cv2.imread(str(i + 1) + '.jpeg')
        img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        img2,A = cylindrical_warp(img2,k)
        xyz.append(A)
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
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        print('HOMOGRAPHY',H)
        H = np.linalg.inv(H)
        img3 = cv2.warpPerspective(img2,H,(gray1.shape[1]+gray2.shape[1],hieght))
        temp = np.zeros_like(img3)
        col = img3.shape[1]-canvas.shape[1]
        temp = cv2.resize(temp,(col,img3.shape[0]),interpolation = cv2.INTER_CUBIC)
        canvas = np.hstack((canvas,temp))
        canvas = pano(canvas,img3)
        cv2.imshow('canvas',canvas)
        cv2.waitKey(0)

cv2.imshow('warped_image',img3)
cv2.imwrite('img.jpg',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
