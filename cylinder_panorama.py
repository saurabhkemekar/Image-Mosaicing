import cv2
import numpy as np
import os
import math
#os.chdir('/home/saurabh/PANORAMA/3')
def cylindricalWarpImage(img1, K):
        f = K[0,0]
        points = []
        color = []
        (im_h,im_w,channel) = img1.shape
        cyl = np.zeros_like(img1)
        cyl_mask = np.zeros_like(img1)
        (cyl_h,cyl_w,channel) = cyl.shape
        x_c = float(cyl_w) / 2.0
        y_c = float(cyl_h) / 2.0
        for x_cyl in np.arange(0,cyl_w):
            for y_cyl in np.arange(0,cyl_h):
                theta = (x_cyl - x_c) / f
                h     = (y_cyl - y_c) / f
                X = np.array([math.sin(theta), h, math.cos(theta)])
                points.append(X)
                color.append(img1[y_cyl,x_cyl])
                X = np.dot(K,X)
                x_im = X[0] #/ X[2]
                if x_im < 0 or x_im >= im_w:
                    continue
                y_im = X[1] #/ X[2]
                if y_im < 0 or y_im >= im_h:
                    continue

                cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
        return cyl,points,color
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
    cv2.imshow('thresh',thresh)
    try:
        cnt =contour[0]
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        approx = arrange_points(approx)
        x,y,w,h = cv2.boundingRect(cnt)
        if approx[2][0][0] < approx[3][0][0]:
            img = img[y:y+h,x:x+approx[2][0][0]]
        else:
            img = img[y:y + h, x:x + approx[3][0][0]]
    except Exception:
        return img
    return  img

noi = 2 # number of images to be stich
img1 = cv2.imread('1.jpeg')
img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
hieght  = img1.shape[0] + 50
k = np.array([[583.61969059, 0, 327.46517597], [0, 582.90936082, 251.76312815], [0, 0, 1]])
#img1 = cylindricalWarpImage(img1,k)
img3 = img1.copy()
for i in range(1,noi):
        img3 = required_img(img3)
        img1 = img3
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
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        print('HOMOGRAPHY',H)
        H = np.linalg.inv(H)
        img3 = cv2.warpPerspective(img2,H,(gray1.shape[1]+gray2.shape[1],hieght))
        img3[0:gray1.shape[0],0:gray1.shape[1]] = img1

img3 = required_img(img3)
img3,points,color = cylindricalWarpImage(img3, k)
color = np.array(color)
points= np.array(points)
x = points[:,0]
y = points[:,1]
z = points[:,2]
b = color[:,0]
g = color[:,1]
r = color[:,2]
print(x)
cv2.imshow('warped_image',img3)
cv2.imwrite('img.jpg',img3)
import open3d as o3d
xyz = np.zeros((np.size(x),6))
xyz[:,0] = np.array(x)
xyz[:,1] = np.array(y)
xyz[:,2] = np.array(z)
xyz[:,3] = np.array(r)
xyz[:,4] = np.array(g)
xyz[:,5] = np.array(b)
# print(xyzrbg)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz[:,:3])
pcd.colors = o3d.utility.Vector3dVector(xyz[:,3:])
o3d.io.write_point_cloud('/home/saurabh/PANORAMA/data.ply',pcd)
pcd_load = o3d.io.read_point_cloud('/home/saurabh/PANORAMA/data.ply')
o3d.visualization.draw_geometries([pcd_load])
cv2.waitKey(0)
cv2.destroyAllWindows()
