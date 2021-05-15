import os
import random
import math
import cv2
import numpy as np


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, sl=0.05, sh=0.4, r1=0.3):
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    # SURSA: articol random erasing, de la autor
    def erase(self, img):

        if len(img.shape) == 3:
            height, width, color = img.shape
        else:
            height, width = img.shape

        area = height * width
        while True:
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                y1 = random.randint(0, height - h)
                x1 = random.randint(0, width - w)

                if len(img.shape) == 3:
                    for line in range(y1, y1 + h):
                        for column in range(x1, x1 + w):
                            img[line][column][0] = random.randint(0, 255)
                            img[line][column][1] = random.randint(0, 255)
                            img[line][column][2] = random.randint(0, 255)
                else:
                    for line in range(y1, y1 + h):
                        for column in range(x1, x1 + w):
                            img[line][column] = random.randint(0, 255)
                return img

    def eraseWholePath(self, path, savePath):
        # just for testing purpose
        # path = 'E:/Anul 3/Augmentator/Images/CleanDataset'
        # the path were I save the processed images
        # savePath = "E:/Anul 3/Augmentator/Images/Processed/RandomErasing"
        # iterate through the files contained in that folder

        # TESTING PURPOSE
        count = 0
        for filename in os.listdir(path):
            # read the image
            img = cv2.imread(os.path.join(path, filename))
            # if it is an image file
            if img is not None:
                # make a copy for edit purpose
                imageCopy = np.copy(img)
                imageCopy = self.erase(imageCopy)
                cv2.imwrite(os.path.join(savePath, filename), imageCopy)
                # TESTING PURPOSE
                count += 1
                print(str(count) + " out of " + str(len(os.listdir(path))))
