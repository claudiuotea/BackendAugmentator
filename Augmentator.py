import os
import random as rand
import cv2
import imutils
import numpy as np

import RandomEraser


class Augmentator:
    def __init__(self, clahe, grayscale, flipRotate, erase):
        self.randomEraser = RandomEraser()

    def claheWholePath(self, path, savePath):
        for filename in os.listdir(path):
            # read the image
            img = cv2.imread(os.path.join(path, filename))
            # if it is an image file
            if img is not None:
                # make a copy for edit purpose
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                cv2.imwrite(os.path.join(savePath, filename), gray)

    def grayscaleWholePath(self, path, savePath):
        for filename in os.listdir(path):
            # read the image
            img = cv2.imread(os.path.join(path, filename))
            # if it is an image file
            if img is not None:
                # make a copy for edit purpose
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(savePath, filename), gray)

    def FlipAndRotateDataset(self, path, savePath):
        rotationValues = [0, 90, 180, 270]
        howManyFlipped = 0
        # 50%
        percentage_chance = 0.50
        # just for testing purpose
        # path = 'E:/Anul 3/Augmentator/Images/CleanDataset'
        # the path were I save the processed images
        # savePath = "E:/Anul 3/Augmentator/Images/Processed/RotateFlip"

        # TESTING PURPOSE
        # TODO:remove those things with TESTING and comment the code
        count = 0

        # iterate through the files contained in that folder
        for filename in os.listdir(path):
            # read the image
            img = cv2.imread(os.path.join(path, filename))
            # if it is an image file
            if img is not None:
                # make a copy for edit purpose
                imageCopy = np.copy(img)

                # rotate the image with a random value which exists in {0,90,180,270}
                rotationValue = rand.choice(rotationValues)
                imageCopy = imutils.rotate_bound(imageCopy, rotationValue)

                # now calculate the probability for flipping the image
                if rand.random() < percentage_chance:
                    imageCopy = cv2.flip(imageCopy, 1)
                    howManyFlipped += 1
                # save the image
                cv2.imwrite(os.path.join(savePath, filename), imageCopy)
                # TESTING PURPOSE
                count += 1
                print(str(count) + " out of " + str(len(os.listdir(path))) + " for Rotations")

    def eraseWholePath(self, path, savePath):
        self.randomEraser.eraseWholePath(path,savePath)

