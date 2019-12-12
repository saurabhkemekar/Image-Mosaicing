import cv2
import numpy as np
import os
os.chdir('/home/saurabh/PANORAMA/3')

def arrange_points(points):
    number = np.array([a[0] for a in points])
    x_cen = 0
    y_cen = 0
    for [x, y] in number:
        x_cen = x_cen + x
        y_cen = y_cen + y
    x_cen = x_cen // 4
    y_cen = y_cen // 4
    sorted_x = np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]])
    for [x, y] in number:
        if x <= x_cen and y <= y_cen:
            sorted_x[0][0] = np.array([x, y])
        if x <= x_cen and y >= y_cen:
            sorted_x[1][0] = np.array([x, y])
        if x >= x_cen and y >= y_cen:
            sorted_x[2][0] = np.array([x, y])
        if x >= x_cen and y <= y_cen:
            sorted_x[3][0] = np.array([x, y])
    points = np.array(sorted_x)
    return  points

def required_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    image,contour,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnt =contour[0]
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        approx = arrange_points(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        pts1 = np.float32(approx)
        pts2 = np.float32([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        img =cv2.warpPerspective(img,M,(x+w,y+h))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        image,contour,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt =contour[0]
        x,y,w,h = cv2.boundingRect(cnt)
        img = img[y:y+h-2,x:x+w-2]
    except Exception:
        return img
    return  img
noi = 5 # number of images
img1 = cv2.imread(str(i) + '.jpeg')
img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
hieght  = img1.shape[0] +50

for i in range(1,noi):            
        img2 = cv2.imread(str(i + 1) + '.jpeg')
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
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0) # homography
        H = np.linalg.inv(H)
        img3 = cv2.warpPerspective(img2,H,(gray1.shape[1]+gray2.shape[1],hieght))
        l.append(img4)
        img3[0:gray1.shape[0],0:gray1.shape[1]] = img1
        if noi-1!=i:
             img3 = required_img(img3)
         img1 = img3
       
img3 = required_img(img3)
cv2.imshow('warped_image',img3)
cv2.imwrite('img.jpg',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
