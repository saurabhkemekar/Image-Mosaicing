import cv2
import numpy as np
import os
#os.chdir('/home/saurabh/PANORAMA/3')
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

def cylinder_coordinates(img,K):
    foc_len = (K[0][0] +K[1][1])/2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
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

noi = 4 # number of images to be stich
img1 = cv2.imread('1.jpeg')
img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
hieght  = img1.shape[0] + 50
k = np.array([[583.61969059, 0, 327.46517597], [0, 582.90936082, 251.76312815], [0, 0, 1]])
img3 = img1.copy()
img1 = cylindrical_warp(img1,k) 

for i in range(1,noi):
        img1 = img3
        img2 = cv2.imread(str(i + 1) + '.jpeg')
        img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
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
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        print('HOMOGRAPHY',H)
        H = np.linalg.inv(H)
        img3 = cv2.warpPerspective(img2,H,(gray1.shape[1]+gray2.shape[1],hieght))
        temp = np.zeros_like(img3)
        col = img3.shape[1]-canvas.shape[1]
        temp = cv2.resize(temp,(col,img3.shape[0]),interpolation = cv2.INTER_CUBIC)
        canvas = np.hstack((canvas,temp))
        canvas = image_stiching(canvas,img3)
        cv2.imshow('canvas',canvas)
        cv2.waitKey(0)

cv2.imwrite('Panorama.jpg',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
