# Furnap (Furniture Snap)
# @uthor Shayan Hemmatiyan
# Copywrite 2019

# Introdcution:
Furnap is an interactive recommendation tool which enables customers to find a mathcing furniture based on the color compositions of the existing furnitures in their rooms. 

It combines object identification using Convolutional Neural Network (CNN) and K-means clustering to cluster different images which they have the same color composition. 

Here are the steps for furnap recommendations:

1- CNN: Identifling objects in the room using YOLO structure and draw boxes enclosing the objects
2- Image Processing: Crop the object within the box
3- KMeans: Finding the color composition of the identified object using Lab color metric
4- KMeans: Using Kmeans clustering to group all the images with the same color composition






