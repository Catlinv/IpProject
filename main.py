import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
           "X", "Y"]
# Toggle showing intermediary images
showImages = 1
# Toggle between adding data to the dataset and "guessing" the letter
saveImage = 0
saveLetter = "A"
# Name of the image.jpg from the refImages folder
fileName = "P"


def calcDiff(a, b):
    diff = 0
    for i in range(0, 100):
        diff += abs(a[i] - b[i])
    return diff


image = cv2.imread("refImages/" + fileName + ".jpg", 1)
image = cv2.resize(image, (500, 500))
# Apply canny edge on the image.
cv2.imwrite("canny.jpg", cv2.Canny(image, 100, 200))
cannyImg = cv2.imread("canny.jpg")
th, cannyImg = cv2.threshold(cannyImg, 220, 255, cv2.THRESH_BINARY_INV)

height = cannyImg.shape[0]
width = cannyImg.shape[1]

# Thicken borders
for y in range(0, height):
    for x in range(0, width):
        # threshold the pixel
        if cannyImg[y, x].all() == 0:
            cv2.circle(cannyImg, (x, y), 5, (1, 1, 1), -1)

if showImages: cv2.imshow("cannyA", cannyImg)

mask = np.zeros((height + 2, width + 2), np.uint8)
cv2.floodFill(cannyImg, mask, (height / 2 + 50, width / 2), 0)
cv2.floodFill(cannyImg, mask, (height / 2 - 50, width / 2), 0)
cv2.floodFill(cannyImg, mask, (height / 2, width / 2), 0)

# Get only the hand from the image
top = height - 1
left = width - 1
right = 0
bottom = 0
for y in range(0, height):
    for x in range(0, width):
        # threshold the pixel
        if cannyImg[y, x].all() == 0:
            top = min(y, top)
            bottom = max(y, bottom)
            left = min(x, left)
            right = max(x, right)

# cannyImg = cannyImg[top - 10:bottom + 10, left - 10:right + 10]
cannyImg = cv2.resize(cannyImg, (256, 256))

# Formatted image
cv2.imwrite("canny.jpg", cv2.Canny(cannyImg, 100, 200))
cannyImg = cv2.imread("canny.jpg")
th, cannyImg = cv2.threshold(cannyImg, 220, 255, cv2.THRESH_BINARY_INV)
for y in range(0, 256):
    for x in range(0, 256):
        # threshold the pixel
        cannyImg[y, x] = 255 - cannyImg[y, x]
if showImages: cv2.imshow("canny", cannyImg)
rez = cannyImg.copy()
aux = cv2.cvtColor(cannyImg, cv2.COLOR_BGR2GRAY)

# Compute distances from centroid

contours, hierarchy = cv2.findContours(aux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if showImages: print len(contours)
minDiff = np.inf
rezLetter = ""
c = contours[0]
# calculate moments for each contour
M = cv2.moments(c)
arr = []
# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cv2.circle(rez, (cX, cY), 5, (100, 100, 100), -1)
for p in c:
    a = p[0]
    dist = math.sqrt((cX - a[0]) ** 2 + (cY - a[1]) ** 2)
    arr.append(dist)
maxi = max(arr)
arr = [x / maxi for x in arr]
arr = [int(x * 100) for x in arr]
d = {x: arr.count(x) for x in range(0, 100)}

# Histogram of 100 bins
d = d.values()
if saveImage:
    with open("letters/" + saveLetter + ".txt", "r") as txt_file:
        lines = txt_file.readlines()
    lines.append(d)
    with open("letters/" + saveLetter + ".txt", "w") as txt_file:
        for line in lines:
            txt_file.write(str(line).replace('[', '').replace(']', '') + "\n")
else:
    for l in letters:
        with open("letters/" + l + ".txt", "r") as txt_file:
            lines = txt_file.readlines()
            for line in lines:
                letter = line.split(',')
                if len(letter) < 100: continue
                letter = [int(x) for x in letter]
                diff = calcDiff(letter, d)
                if diff < minDiff:
                    minDiff = diff
                    rezLetter = l
cv2.putText(image, rezLetter, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
if showImages: cv2.imshow("Original Image", image)
if showImages: cv2.imshow("Centroid Image", rez)
print rezLetter
cv2.waitKey()
cv2.destroyAllWindows()
