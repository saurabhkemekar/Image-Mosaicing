import cv2
import numpy as np
img3 = 0
import os
os.chdir('/home/saurabh/PANORAMA/yosemite_test')
def wanted_area(img):
        gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        image,contour,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contour[0]
        x,y,w,h = cv2.boundingRect(cnt)
        #gray = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))
        img = img[x:x+h-15,y:y+w-15]
        return img
for i in range(1,4):
        if i==1:
                img1 = cv2.imread(str(i)+'.jpg')
        else:
            img1 = img3
        img2 = cv2.imread(str(i+1)+'.jpg')
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1,desc1 = sift.detectAndCompute(gray1,None)
        ''' This return the keypoints and descriptor the vector that store the  descriptor for each keypoints
         that is describe using the 128 dimension vector(descriptor vector)'''
        kp2,desc2 = sift.detectAndCompute(gray2,None)
        #kp2_img2 = cv2.drawKeypoints(img2,kp2,img2)
        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1,desc2,k=2)
        good = []
        pt1 = []
        pt2 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        ''' This  gives the homography from query image to train img'''
        H = np.linalg.inv(H)
        ''' This H  define  the homography train img to query image  '''
        img3 = cv2.warpPerspective(img2,H,(gray1.shape[1]+ gray2.shape[1],gray2.shape[0]))
        img3[0:gray1.shape[0],0:gray1.shape[1]] = img1
        img3 = wanted_area(img3)
        cv2.imshow('img3', img3)
        cv2.waitKey(1)
#cv2.imshow('warped_image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
