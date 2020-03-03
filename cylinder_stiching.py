
import cv2
import numpy as np
import os

os.chdir("E:\images")

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
#---------------------------------------------------------------------------------------------------------------------------------------------
def cylindrical_warp(img,K):
    foc_len = (K[0][0] +K[1][1])/2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
    color = img[y,x]
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
noi = 4200 # number of images
img1 = cv2.imread('1.png')
k = np.array([[951.15755604 ,0, 621.51429561], [0, 947.48375342, 349.79574375], [0, 0, 1]])
img1 = cylindrical_warp(img1,k)
h  = img1.shape[0] + 200
img3 = img1.copy()
canvas = np.zeros((h,int(2*3.41*k[0][0]),img1.shape[2]),np.uint8)
print('canvas shape',canvas.shape)
canvas[0:img1.shape[0],0:img1.shape[1]] = img1
for i in range(2,noi):
       # print('-------------'+str(i))
        img1 = img3.copy()
        name =  str(i) +'.png'
        img2 = cv2.imread(name)
        img2 = cylindrical_warp(img2,k)
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY)
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
        good = good[:25]
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        p = pt1-pt2
        dist = np.mean(p,axis = 0)
        std_y = np.std(p[:,0,1])
        std_x = np.std(p[:,0,0])
        print('frame No = {} Deviation in X = {} ,Y = {}'.format(i,std_x,std_y))
        if std_y <1 and std_x <1:
            M = np.array([[1,0,dist[0,0]],[0,1,dist[0,1]], [0.,0.,1.]])
            img3 = cv2.warpPerspective(img2,M,(canvas.shape[1],canvas.shape[0])) #, flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)
            print(img3.shape)
            canvas = image_stiching(canvas,img3)
            cv2.imwrite('image3.jpg',canvas)
            cv2.waitKey(1)
cv2.destroyAllWindows()
