from PIL import Image
from skimage import data, util
import numpy as np
from skimage.util import random_noise



# Q2 load image in ur program
img = Image.open('camera_man.png').convert('L')     # to convert image to grayscale image
img_array= np.array(img)                            #convert image to array format
actual_array= img_array
img1_array= img_array


# #Q3 Print the image size and data type.
print(img.size)             #to show image size
weidth, height = img.size
print(img_array.dtype)      # to show image datatype


# #Q4 Compute the minimum, maximum and mean value of pixels of the full image.
print(img_array.min())          # to show minimum pixel in the image
print(img_array.max())          # to show max value of pixel
print(img_array.mean())         # to show mean of pixel


#Q5.Create a random noise image B with the same size of A with pixel values between 0-255.Subtract A from B to get C, which is expected to be a “noisy cameraman.”

noisy_man  = random_noise(img_array, mode='s&p', amount=0.5)    #to put salt and pepper noise in the original image(as mentioned by the professor)
noisy_man = (255*noisy_man).astype(np.uint8)                 
noisy_cameraman= Image.fromarray(noisy_man)                     # to convert array back to image


c= np.subtract(noisy_man,img_array)                             # to substract original image from noisy image
difference= Image.fromarray(c)


#Q6 Display A, B and C.
img.show()                  #show original image
img.save('original.png')
noisy_cameraman.show()      #show the noisy image
noisy_cameraman.save("noisy_image.png")
difference.show()           #difference between noisy image and original image
difference.save("noisy_cameraman.png")

#Q7 Invert the image using intensity transformation s = T(z)

inverted_array = util.invert(img_array)             #invert the image ( this scikit-image module which uses intensity transformation)
inverted_image=Image.fromarray(inverted_array)
inverted_image.show()
inverted_image.save("inverted_image.png")

#Q8 Replace each pixel by the average of 3 x 3 neighbors using 4-connected neighbors (I4) and 8-connected neighbors (I8).

#I4 matrix- to calculate I4 matrix
img2_array = np.empty((height, weidth))
img3_array=np.empty((height, weidth))

for i in range (height):
    for j in range(weidth):
        if (i==0 and j==0) or(i== 0 and j== weidth-1) or (i== height -1 and j== 0)or(i== height-1 and j==weidth-1):
            if i==0 and j==0:
                sum=int(img_array[i+1][0])+int(img_array[0][j+1])
                avg=int(sum/2)
                img2_array[i][j]=avg    
            if i==0 and j==weidth-1:
                sum=int(img_array[i+1][j])+int(img_array[0][j-2])
                avg=int(sum/2)
                img2_array[i][j]=avg
                print(img_array[i][j])
            if i==height-1 and j==0:
                sum=int(img_array[i-1][0])+int(img_array[i][j+1])
                avg=int(sum/2)
                img2_array[i][j]=avg
            if i==height-1 and j==weidth-1:
                sum=int(img_array[i][j-1])+int(img_array[i-1][j])
                avg=int(sum/2)
                img2_array[i][j]=avg
        elif i==0 or j==0 or i== height-1 or j== weidth-1:
            if i==0:
                sum=int(img_array[i][j-1])+int(img_array[i][j+1])+int(img_array[i+1][j])
                avg=int(sum/3)
                img2_array[i][j]=avg
            if j==0:
                sum=int(img_array[i-1][j])+int(img_array[i+1][j])+int(img_array[i][j+1])
                avg=int(sum/3)
                img2_array[i][j]=avg
            if i==height-1:
                sum=int(img_array[i][j-1])+int(img_array[i][j+1])+int(img_array[i-1][j])
                avg=int(sum/3)
                img2_array[i][j]=avg
            if j==weidth-1:
                sum=int(img_array[i][j-1])+int(img_array[i-1][j])+int(img_array[i+1][j])
                avg=int(sum/3)
                img2_array[i][j]=avg
                # print(avg)
        else:
            sum=int(img_array[i][j-1])+int(img_array[i][j+1])+int(img_array[i-1][j])+ int(img_array[i+1][j])
            avg=int(sum/4)
            img2_array[i][j]=avg

I4_image=Image.fromarray(img2_array).convert('L') 
I4_image.show()
I4_image.save("I4_image.png")

#I8 matrix- to calculate I8 matrix



for i in range (height):
    for j in range(weidth):
        if (i==0 and j==0) or(i== 0 and j== weidth-1) or (i== height -1 and j== 0)or(i== height-1 and j==weidth-1):
            if i==0 and j==0:
                sum=int(img1_array[i+1][0])+int(img1_array[0][j+1])+int(img1_array[i+1][j+1])
                avg=int(sum/3)
                img3_array[i][j]=avg    
            if i==0 and j==weidth-1:
                sum=int(img1_array[i+1][j])+int(img1_array[0][j-1])+int(img1_array[i+1][j-1])
                avg=int(sum/3)
                img3_array[i][j]=avg
            if i==height-1 and j==0:
                sum=int(img1_array[i-1][0])+int(img1_array[i][j+1])+int(img1_array[i-1][j+1])
                avg=int(sum/3)
                img3_array[i][j]=avg
            if i==height-1 and j==weidth-1:
                sum=int(img1_array[i][j-1])+int(img1_array[i-1][j])+int(img1_array[i-1][j-1])
                avg=int(sum/3)
                img3_array[i][j]=avg
        elif i==0 or j==0 or i== height-1 or j== weidth-1:
            if i==0:
                sum=int(img1_array[i][j-1])+int(img1_array[i][j+1])+int(img1_array[i+1][j])+int(img1_array[i+1][j-1])+int(img1_array[i+1][j+1])
                avg=int(sum/5)
                img3_array[i][j]=avg
            if j==0:
                sum=int(img1_array[i-1][j])+int(img1_array[i+1][j])+int(img1_array[i][j+1])+int(img1_array[i-1][j+1])+int(img1_array[i+1][j+1])
                avg=int(sum/5)
                img3_array[i][j]=avg
            if i==height-1:
                sum=int(img1_array[i][j-1])+int(img1_array[i][j+1])+int(img1_array[i-1][j])+int(img1_array[i-1][j-1])+int(img1_array[i-1][j+1])
                avg=int(sum/5)
                img3_array[i][j]=avg
            if j==weidth-1:
                sum=int(img1_array[i][j-1])+int(img1_array[i-1][j])+int(img1_array[i+1][j])+int(img1_array[i-1][j-1])+int(img1_array[i+1][j-1])
                avg=int(sum/5)
                img3_array[i][j]=avg
                # print(avg)
        else:
            sum=int(img1_array[i][j-1])+int(img1_array[i][j+1])+int(img1_array[i-1][j])+ int(img1_array[i+1][j])+int(img1_array[i-1][j-1])+int(img1_array[i-1][j+1])+int(img1_array[i+1][j+1])+int(img1_array[i+1][j-1])
            avg=int(sum/8)
            img3_array[i][j]=avg
                
I8_image=Image.fromarray(img3_array).convert('L') 
I8_image.show()
I8_image.save("I8_image.png")

#Q9.Compute the difference of (I-I4 ) and (I-I8 ) and display all the images.

c= np.subtract(actual_array,img_array)      # difference between original image and I4 image
difference1= Image.fromarray(c)
difference1.show()
difference1.save("I-I4.png")

c1= np.subtract(actual_array,img_array)     # difference between original image and I8 image
difference2= Image.fromarray(c1)
difference2.show()
difference2.save("I-I8.png")



# --------------------OUTPUT-------------------------------------------------------
# (422, 426)
# uint8
# 0
# 255
# 117.65758293838863
# 152

#--------------------Reference-----------------------------------------------------
#https://scikit-image.org/docs/dev/user_guide/numpy_images.html
#https://pillow.readthedocs.io/en/stable/reference/Image.html
#https://numpy.org/doc/stable/reference/generated/numpy.subtract.html
#https://scikit-image.org/docs/dev/user_guide/transforming_image_data.html
#