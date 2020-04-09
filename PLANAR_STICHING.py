import cv2
import numpy as np
import os
import math
os.chdir('/home/saurabh/PANORAMA/images')
 
def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img

cap = cv2.VideoCapture('/home/saurabh/Downloads/2020-01-10-183601.mp4')
ret,img1 = cap.read()
img1 =  cv2.resize(img1 ,None,fx = 0.5 ,fy = 0.5,interpolation= cv2.INTER_CUBIC)
hieght  = img1.shape[0] + 200

# create canvas for panorama which greater than image size
canvas = np.zeros((img1.shape[0]*3,img1.shape[1]*3,img1.shape[2]),np.uint8)
M = np.float32([[1, 0, 50], [0, 1, 100]])

# M is translation matrix which is to shift refrence image to 50,100
canvas = cv2.warpAffine(canvas, M, (canvas.shape[0], canvas.shape[1]))

while(1):
        ret,img2 = cap.read()
        
        # resizing the image 
        img2 = cv2.resize(img2,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_CUBIC)
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        # create the sift obejct
        sift = cv2.xfeatures2d.SIFT_create()
        
        # computes the keypoints and descriptors
        kp1,desc1 = sift.detectAndCompute(gray1,None)
        kp2,desc2 = sift.detectAndCompute(gray2,None)
        
        # use the brute force matching to find the matches in 2 images 
        # crossCheck = False because if its true first it find matches from image1 ->image2 then cross check 
        #whether it matches form image2 -> image1 which will number of features
        
        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1,desc2,k=2)
        good = []
        pt1 = []
        pt2 = []
        
        # 0.7 is knows as Lowe's ratio
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        # sorting the matches based on distance
        good = sorted(good,key = lambda x:x.distance)
        
        # taking only best 15 matches
        good = good[:15]
        
        # pt1 is location of keypoints in image1
        # pt2 is location of keypoints in image2
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        # find Homography between image1 and image2
        H,mask = cv2.findHomography(pt2,pt1,cv2.RANSAC,ransacReprojThreshold=4.0)
        img3 = cv2.warpPerspective(img2,H,(canvas.shape[1],canvas.shape[0]))
        canvas2 = np.zeros_like(canvas)
        M = np.float32([[1, 0, 50], [0, 1, 100]])
        canvas2 = cv2.warpAffine(img3, M, (img3.shape[1],img3.shape[0]))
        canvas = image_stiching(canvas,canvas2)
        cv2.imshow('canvas',canvas)
        hieght = canvas.shape[0]
        if cv2.waitKey(1) == 2:
            break
cv2.imshow('panorama',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
