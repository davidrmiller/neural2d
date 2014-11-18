#!/usr/bin/python3

# This extracts the images and data from the MNIST handwritten digit database,
# to create the files needed for neural2d. See http://yann.lecun.com/exdb/mnist/ 
# for more information about the MNIST database.

# To use this program, run it with Python3 in the same directory containing the
# four MNIST files downloaded from http://yann.lecun.com/exdb/mnist/. If successful,
# this program will extract the images and save them in separate .bmp files. It will
# also create an inputData-train.txt file that can be used with neural2d. 

# Names of the four MNIST files:

TRAIN_IMAGES="train-images.idx3-ubyte"
TRAIN_LABELS="train-labels.idx1-ubyte"
VALIDATE_IMAGES="t10k-images.idx3-ubyte"
VALIDATE_LABELS="t10k-labels.idx1-ubyte"

# Destination directories for the training and validation files:

DEST_DIR_TRAIN="train-data"
DEST_DIR_VALIDATE="validate-data"

# We can change which two values are used to represent true and false. If using
# the default tanh transfer function, the true and false values work best if they
# are 1 and -1. If using the logistic transfer function, you'll want to use
# 1 and 0 as true and false:

falseVal = -1
trueVal = 1


MNIST_URL="http://yann.lecun.com/exdb/mnist/"


# In the labels file, the number of items is 32-bit big-endian integer at offset 4.
# The labels are one byte each, binary 0..9, starting at offset 8.

# In the images file, the number of items is 32-bit big-endian integer at offset 4.
# The image height is 32-bit big-endian integer at offset 8.
# The image width is 32-bit big-endian integer at offset 0xc.
# The images are height*width bytes each, starting at offset 0x10.

import sys
import os
from PIL import Image, ImageDraw


def makeDataSet(imageFile, labelFile, destDir, prefix, configFile):
    print("Extracting images and labels from %s and %s" % (imageFile, labelFile)) 
    print("into directory %s" % (destDir))
    
    if not os.path.exists(destDir):
        os.makedirs(destDir)
    
    ifile = open(imageFile, "rb")
    lfile = open(labelFile, "rb")
    config = open(configFile, "wt")
    
    # Get the number of items from both headers; the numbers must match.
    
    lfile.seek(4)
    numLabels  = ord(lfile.read(1)) << 24
    numLabels += ord(lfile.read(1)) << 16
    numLabels += ord(lfile.read(1)) << 8
    numLabels += ord(lfile.read(1))
    print("# numLabels=%d" % (numLabels))
    
    ifile.seek(4)
    numImages  = ord(ifile.read(1)) << 24
    numImages += ord(ifile.read(1)) << 16
    numImages += ord(ifile.read(1)) << 8
    numImages += ord(ifile.read(1))
    print("# numImages=%d" % (numImages))
    
    if numImages != numLabels:
        print("Error: the image and label files don't appear to be matching MNIST database files.")
        print("Download the necessary files from %s" % (MNIST_URL))
        exit(1)
    
    # Get the image dimensions
    
    ifile.seek(8)
    height  = ord(ifile.read(1)) << 24
    height += ord(ifile.read(1)) << 16
    height += ord(ifile.read(1)) << 8
    height += ord(ifile.read(1))
    print("# height=%d" % (height))
    
    ifile.seek(0xc)
    width  = ord(ifile.read(1)) << 24
    width += ord(ifile.read(1)) << 16
    width += ord(ifile.read(1)) << 8
    width += ord(ifile.read(1))
    print("# width=%d" % (width))

    if height != 28 or width != 28:
        print("Error: the database files don't appear to be the right files.")
        print("Download the necessary files from %s" % (MNIST_URL))
        exit(1)
    
    # Position the file read at the start of data:
    
    ifile.seek(0x10)
    lfile.seek(8)
    
    # Write the input data config file

    print("Writing BMP files... this may take a few minutes...")
    
    indexNum = 0;  # starting number for naming the extracted image files

    while numImages > 0:
        # Create the image file:
        bytes = ifile.read(height * width)
        im = Image.frombytes("L", (width, height), bytes)
        im = im.convert("RGB")
        imageFilename = "%s/%s%d.bmp" % (destDir, prefix, indexNum)
        im.save(imageFilename)
    
        # Collect the target values:
        label = lfile.read(1)  # 0..9
        targetVals = [falseVal] * 10
        targetVals[ord(label)] = trueVal
    
        # Print the config line:
        config.write("images/mnist/%s %d %d %d %d %d %d %d %d %d %d\n" % (
              imageFilename,
              targetVals[0],
              targetVals[1],
              targetVals[2],
              targetVals[3],
              targetVals[4],
              targetVals[5],
              targetVals[6],
              targetVals[7],
              targetVals[8],
              targetVals[9]))
        numImages = numImages - 1
        indexNum = indexNum + 1

if __name__ == '__main__':
    makeDataSet(TRAIN_IMAGES,    TRAIN_LABELS,    DEST_DIR_TRAIN,    "", "inputData-mnist.txt")
    #makeDataSet(VALIDATE_IMAGES, VALIDATE_LABELS, DEST_DIR_VALIDATE, "", "inputData-mnist-validate.txt")
    print("Done. You can now run neural2d using topology.txt and inputData-train.txt or inputData-mnist.txt.")


