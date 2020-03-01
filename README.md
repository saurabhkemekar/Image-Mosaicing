# IMAGE STITCHING IN PLANAR AND CYLINDRICAL COORDINATE SYSTEM


## OVERVIW
The concept of Image Stitching has been around for quite sometime and has been one of the most successful implementation of Computer Vision techniques. Nowdays, panoramic images can be shot using an ordinary cell phone. In this project, we implement some robust and well establised algorithms for real time image stitching. We shall discuss each of the algorithms in brief. The coding was done using python3, along with numpy for optimizing numerical and matrix calculations. 

## REQUIREMENTS
numpy==1.16.4 opencv-python==3.4.2

## Image Stitching using Homography

1. Feature Detection on images using SIFT.
SIFT stands for Scale Invariant Feature Transform, which basically is used for detection features on the images. The example image taken for representing the action of SIFT is shown as 1.jpeg in the repository.[1] 

![image_screenshot_01 03 2020](https://user-images.githubusercontent.com/45517467/75624246-a6ef8500-5bd8-11ea-9dfc-73cacfb0fd20.png)

2. Feature Matching between two images using Brute Force KNN.
Here, we take the descriptor of one feature of first set, which is matched with all other features of the second set, calculated using L2 norm distance calculation, while returning the closest one. We have removed the cross checking in our case. Also, instead of drawing all the best matches, we draw k = 2 best matches, that is, drawing two match-lines for each keypoint. Its output with 1.jpeg and 2.jpeg is shown below.[2]

![image](https://user-images.githubusercontent.com/45517467/75624402-caff9600-5bd9-11ea-8c93-6ab62c30a6de.png)

3. Generation of transformation matrix.
Planar homography relates the transformation between two planes. It is a 3x3 matris with 8 DoF. The elements of the matrix are generally normalized to 1. So now, in order to transfer the image coordinates from the source plane to the targeted plane, we just have to determine the homography matrix and apply it to the source plane, so as to trasfer the image to the targeted plane, as per the matched features. In this way, we are able to obtain the result as shown.[3]
![image](https://user-images.githubusercontent.com/45517467/75624566-680efe80-5bdb-11ea-8542-73767db92bf5.png)

In our code, we have done the planar image stitching for multiple images. So, we have initially made a Black canvas of 5 times the length of image, and height 200 pixels greater than the height of the image. Consider that the first two images has been stitched as shown. Taking the third image, we calculate the homography matrix between the third image and the second image that had been warped in the first iteration, before it was stitched to the source image. Similarly, the homography matrix was calculated for the stitching of fourth image between the fourth image and the third image that had been warped. Stitching of all the four images results as shown, 
![image](https://user-images.githubusercontent.com/45517467/75624642-3c404880-5bdc-11ea-9ec7-1600d63b0a35.png) 

## Cylindrical Image Stitching with Translational Model

1. Conversion from Planar to Cylindrical Coordinates
Image Stitching in cylindrical coordinates assumes the use of a tripod for recording the video, since the motion along the y axis should be as close to nil as possible. In such a case, we have used the translational model for image stitching. But before that, we have converted the image into cylindrical coordinates. According to [4], image can be transferred to a cylinder using the formula listed in the slides. When implemented, its results are as shown,
![image](https://user-images.githubusercontent.com/45517467/75624917-06509380-5bdf-11ea-85d8-4a74d92573fa.png)

Note that conversion from planar to cylindrical coordinates requires the use of the focal length, which can be used from the use of the intrinsic camera parameters that shall be used for this. For this, camera calibration is needed, in order to calculate the parameters, which is done using chessboard for getting corner locations in the image. The repository consisting of the source code can be referred to at [5]
(For the image shown above, the intrinsic camera parameters are assumed, and is used for demonstration of the algorithm)

2. Feature Detection on images using SIFT
Covered in Section 3

3. Feature Matching between two images using Brute Force KNN.
Covered in Section 3

4. Translational Model
Now that the feature points are obtained for the pair of images using SIFT, it is time for the application of the translational model for stitching. In this case, we have approximated the least mean square technique as used in [6]. Instead, we take the difference of the obtained feature coordinate points between both the images and then take the mean of the result. Now, we take its standard deviation. We shall divide the answer into the standard deviation along x axis and y axis. We kept a threshold, such that if the value of the standard deviation along both the axes is less than 0.5, then only the perspective warping of the second image with respect to the first image shall be performed. This is done for gap closure. When the distance between the points are calculated as stated, we divide the array into the distance along the x axis and along y axis, which is then fed into the translational matrix. This matrix is used for the perspective warping process of the images. The array is similarly updated for every iteration, where the translational matrix is made between the next image and the warped version of the previous image. In this way, we are able to obtain the cylindrical panorama.

As for the canvas, the length of the canvas is equal to the circumference of the cylinder, that is, 2 * pi * f, where f is the focal length of the camera, extracted from the intrinsic camera parameters, and the height being 200 pixels more than the actual height of the image. 

## REAL - TIME IMPLEMENTATION
The real - time implementation was performed by recording the video of our lab, where the camera was placed roughly at the centre of the room and rotated at a roughly uniformly angular velocity. The camera used was Intel RealSense Depth Module, using pyrealsense2 for recording the video and for camera calibration. The result for this can be seen below

![image](https://user-images.githubusercontent.com/45517467/75627138-b6c89280-5bf3-11ea-8b23-8effd0af65b7.png)

## TEAM MEMBERS

1. Saurabh Kemekar
2. Arihant Gaur
3. Pranav Patil
4. Danish Gada

## PROJECT MENTOR
1. Aman Jain


## REFERENCES

[1] Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94 <br />
[2] Jakubovic, Amila & Velagic, Jasmin. (2018). Image Feature Matching and Object Detection Using Brute-Force Matchers. 83-86. 10.23919/ELMAR.2018.8534641. </br>
[3] http://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf <br />
[4] http://cs.brown.edu/courses/cs129/results/final/yunmiao/ <br />
[5] https://github.com/saurabhkemekar/camera-calibration <br />

