import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from skimage.transform.pyramids import pyramid_gaussian

img =  Image.open('airplane.tiff').convert('L') 
w, h = img.size
print(w,h)

# Q1-1.2

# translation
Tx = 50
mat_trans_inv = np.array([[1,0,Tx],[0,1,0],[0,0,1]])
img1= ndi.affine_transform(img, mat_trans_inv)
scaler1=  Image.fromarray(img1)
scaler1.save("translation.jpg")
scaler1.show()

# applying scaling transformation
s_x, s_y = 2, 1
mat_scale = np.array([[s_x,0,0],[0,s_y,0],[0,0,1]])
img2 = ndi.affine_transform(img, mat_scale)
scale2r=  Image.fromarray(img2)
scale2r.save("scaling.jpg")
scale2r.show()

# applying rotation transformation
theta = np.pi/6
mat_rotate = np.array([[1,0,w/2],[0,1,h/2],[0,0,1]]) @ np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]) @ np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
img3 = ndi.affine_transform(img, mat_rotate)
scaler3=  Image.fromarray(img3)
scaler3.save("rotation.jpg")
scaler3.show()




#Q1=1.3

#inverse translation
Tx = -50
mat_trans_inv = np.array([[1,0,Tx],[0,1,0],[0,0,1]])
img4 = ndi.affine_transform(img1, mat_trans_inv)
scaler4=  Image.fromarray(img4)
scaler4.save("inverse_translation.jpg")
scaler4.show()

#inverse scaling transformation
s_x, s_y = 0.5, 1
mat_scale_inv = np.array([[s_x,0,0],[0,s_y,0],[0,0,1]])
img5 = ndi.affine_transform(img2, mat_scale_inv)
scaler5=  Image.fromarray(img5)
scaler5.save("inverse_scaling.jpg")
scaler5.show()

#inverse rotation transformation
theta = np.pi/6
mat_rotate_inv = np.array([[1,0,w/2],[0,1,h/2],[0,0,1]]) @ np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]]) @ np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
img6 = ndi.affine_transform(img3, mat_rotate_inv)
scaler6=  Image.fromarray(img6)
scaler6.save("inverse rotation.jpg")
scaler6.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Q2
import cv2
import numpy as np
#2.1
#Lunchroom image
input_img =  cv2.imread("lunchroom_unregistered.jpg",cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread("lunchroom_reference.jpg",cv2.IMREAD_GRAYSCALE)
height, width = reference_img.shape
orb_detector = cv2.ORB_create(90000)
keypoints1, descriptor1 = orb_detector.detectAndCompute(input_img, None)
keypoints2, descriptor2 = orb_detector.detectAndCompute(reference_img, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matching_points = matcher.match(descriptor1, descriptor2)
matching_points.sort(key = lambda x: x.distance)
matching_points = matching_points[:int(len(matching_points)*1)]
no_of_matches = len(matching_points)
matrix1 = np.zeros((no_of_matches, 2))
matrix2 = np.zeros((no_of_matches, 2))

for i in range(len(matching_points)):
	matrix1[i, :] = keypoints1[matching_points[i].queryIdx].pt
	matrix2[i, :] = keypoints2[matching_points[i].trainIdx].pt


homography_matrix, mask = cv2.findHomography(matrix1, matrix2, cv2.RANSAC)

transformed_image = cv2.warpPerspective(input_img,homography_matrix, (width, height)) 

cv2.imshow('image',transformed_image)
cv2.waitKey(5000)
cv2.imwrite('lunchroom_final.png', transformed_image)


#sandiegio image

reference_img = cv2.imread("sandiego_reference.tiff",cv2.IMREAD_GRAYSCALE)
input_img =  cv2.imread("sandiego_unregistered.tiff",cv2.IMREAD_GRAYSCALE)
height, width = reference_img.shape
orb_detector = cv2.ORB_create(90000)
keypoints1, descriptor1 = orb_detector.detectAndCompute(input_img, None)
keypoints2, descriptor2 = orb_detector.detectAndCompute(reference_img, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matching_points = matcher.match(descriptor1, descriptor2)
matching_points.sort(key = lambda x: x.distance)
matching_points = matching_points[:int(len(matching_points)*1)]
no_of_matches = len(matching_points)
matrix1 = np.zeros((no_of_matches, 2))
matrix2 = np.zeros((no_of_matches, 2))

for i in range(len(matching_points)):
	matrix1[i, :] = keypoints1[matching_points[i].queryIdx].pt
	matrix2[i, :] = keypoints2[matching_points[i].trainIdx].pt


homography_matrix, mask = cv2.findHomography(matrix1, matrix2, cv2.RANSAC)

transformed_image = cv2.warpPerspective(input_img,homography_matrix, (width, height)) 

cv2.imshow('image',transformed_image)
cv2.waitKey(5000)
cv2.imwrite('sandeigo_final.png', transformed_image)


#synthetic

reference_img = cv2.imread("synthetic_reference.jpg",cv2.IMREAD_GRAYSCALE)
input_img =  cv2.imread("synthetic_unregistered.jpg",cv2.IMREAD_GRAYSCALE)
height, width = reference_img.shape
orb_detector = cv2.ORB_create(90000)
keypoints1, descriptor1 = orb_detector.detectAndCompute(input_img, None)
keypoints2, descriptor2 = orb_detector.detectAndCompute(reference_img, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matching_points = matcher.match(descriptor1, descriptor2)
matching_points.sort(key = lambda x: x.distance)
matching_points = matching_points[:int(len(matching_points)*1)]
no_of_matches = len(matching_points)
matrix1 = np.zeros((no_of_matches, 2))
matrix2 = np.zeros((no_of_matches, 2))

for i in range(len(matching_points)):
	matrix1[i, :] = keypoints1[matching_points[i].queryIdx].pt
	matrix2[i, :] = keypoints2[matching_points[i].trainIdx].pt


homography_matrix, mask = cv2.findHomography(matrix1, matrix2, cv2.RANSAC)

transformed_image = cv2.warpPerspective(input_img,homography_matrix, (width, height)) 

cv2.imshow('image',transformed_image)
cv2.waitKey(5000)
cv2.imwrite('synthetic_final.png', transformed_image)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Q3 
# 
#3.1. Apply 3-level Gaussian pyramid with the kernel G

import cv2
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

img = cv2.imread("mandrill.tiff",cv2.IMREAD_GRAYSCALE)
print(type(img))
height, width = img.shape
print(height, width)


i=3
img_level_1=img
while i !=0 :
    img_level_2 = cv2.pyrDown(img_level_1)
    # print(type(img_level_2))
    img_level_2=Image.fromarray(img_level_2)
    img_level_2.save("gaussian_pyramid{0}.jpg".format(i))
    img_level_2.show()
    img_level_2= np.array(img_level_2)
    img_level_1= img_level_2
    print(img_level_1.shape)
    i = i-1


#3.2 Apply 3-level subsampling to Fig. 2 without Gaussian smoothing, i.e., simply rejecting even rows and columns to create a ½ size image.

j=3
img_level_1=img
while j !=0 :
    poe=list(range(0, img_level_1.shape[0], 2))
    zoe=list(range(0, img_level_1.shape[1], 2))
    img_level_1=np.delete(img_level_1, poe ,0 )
    img_level_1=np.delete(img_level_1, zoe ,1 ) 
    img_level_2=Image.fromarray(img_level_1)
    img_level_2.save("sampling{0}.jpg".format(j))
    img_level_2.show()
    img_level_2= np.array(img_level_2)
    img_level_1= img_level_2
    j = j-1



