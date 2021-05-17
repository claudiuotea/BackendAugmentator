import os
import random as rand
import shutil
import zipfile
from pathlib import Path

import cv2
import imutils
import numpy as np
from os import path
from RandomEraser import RandomErasing


class Augmentator:
    def __init__(self, saveArchivePath,archiveName,clahe = 0, grayscale = 0, flip = 0, erase = 0, rotate=0, minErase = 0.05, maxErase = 0.4, datasetPath="", flipOption = 1, eraseOption = 1):
        self.clahe = clahe
        self.grayscale = grayscale
        self.flip = flip
        self.erase = erase
        self.rotate = rotate
        #cat la suta din imagine sa fie acoperita de Eraser
        self.minErase = minErase
        self.maxErase = maxErase
        #care este folder-ul din care preiau dataset-ul *fiecare user va avea pe server un folder propriu*
        self.datasetPath = datasetPath
        #optiuni pentru functiile de erase + flip&rotate
        self.flipOption = flipOption
        self.eraseOption = eraseOption
        self.randomEraser = RandomErasing(probability=erase)
        self.archiveName = archiveName
        self.saveArchivePath = saveArchivePath
    #convertesc parametrii din forma de frontend in forma acceptata de functia applyAugmentations
    #static ca sa o pot apela fara creez clasa
    @staticmethod
    def convertParams(isClahe,isGray,isFlip,isErase,isFlipBase,isFlipClahe,
                            isFlipGray,isEraseBase,isEraseClahe,isEraseGray,
                            flipProbability,eraseProbability,rotateProbability):
        clahe = 0
        grayscale = 0
        flip = 0
        erase = 0
        rotate = 0
        flipOption = 1
        eraseOption = 1
        if isClahe == "false":
            clahe = 0
        else:
            clahe = 1
        if isGray == "false":
            grayscale = 0
        else:
            grayscale = 1
        if isFlip == "false":
            flip = 0
            rotate = 0
        else:
            flip = int(flipProbability)
            rotate = int(rotateProbability)
        if isErase == "false":
            erase = 0
        else:
            erase = int(eraseProbability)
        # Coduri optiuni: 1 - pe dataset baza | 2 - pe dataset clahe | 3 - pe dataset grayscale
        # 4- pe dataset baza+clahe | 5- pe baza + grayscale | 6 - pe clahe + grayscale | 7 - pe toate 3
        if isFlipBase and not isFlipClahe and not isFlipGray:
            flipOption = 1
        elif isFlipClahe and not isFlipBase and not isFlipGray:
            flipOption = 2
        elif isFlipGray and not isFlipBase and not isFlipClahe:
            flipOption = 3
        elif isFlipBase and isFlipClahe and not isFlipGray:
            flipOption = 4
        elif isFlipBase and isFlipGray and not isFlipClahe:
            flipOption = 5
        elif isFlipClahe and isFlipGray and not isFlipBase:
            flipOption = 6
        else:
            flipOption = 7

        if isEraseBase and not isEraseClahe and not isEraseGray:
            eraseOption = 1
        elif isEraseClahe and not isEraseBase and not isEraseGray:
            eraseOption = 2
        elif isEraseGray and not isEraseBase and not isEraseClahe:
            eraseOption = 3
        elif isEraseBase and isEraseClahe and not isEraseGray:
            eraseOption = 4
        elif isEraseBase and isEraseGray and not isEraseClahe:
            eraseOption = 5
        elif isEraseClahe and isEraseGray and not isEraseBase:
            eraseOption = 6
        else:
            eraseOption = 7

        return clahe,grayscale,flip,erase,rotate,flipOption,eraseOption

    #creeaza folderele pentru a salva augmentarile
    def createPaths(self):
        #pentru fiecare augmentare se creaza un folder, in caz ca augm trebuie facuta
        #se verifica daca exista folderele, daca nu exista se creaza
        currentPath = self.datasetPath
        if self.clahe > 0:
            os.mkdir(currentPath + "\\CLAHE")
        if self.grayscale > 0:
            os.mkdir(currentPath + "\\GRAY")
        if self.flip > 0:
            os.mkdir(currentPath + "\\FLIP")
            if self.flipOption == 1:
                os.mkdir(currentPath + "\\FLIP\\BASE")
            if self.flipOption == 2:
                os.mkdir(currentPath + "\\FLIP\\CLAHE")
            if self.flipOption == 3:
                os.mkdir(currentPath + "\\FLIP\\GRAY")
            if self.flipOption == 4:
                os.mkdir(currentPath + "\\FLIP\\BASE")
                os.mkdir(currentPath + "\\FLIP\\CLAHE")
            if self.flipOption == 5:
                os.mkdir(currentPath + "\\FLIP\\BASE")
                os.mkdir(currentPath + "\\FLIP\\GRAY")
            if self.flipOption == 6:
                os.mkdir(currentPath + "\\FLIP\\GRAY")
                os.mkdir(currentPath + "\\FLIP\\CLAHE")
            if self.flipOption == 7:
                os.mkdir(currentPath + "\\FLIP\\BASE")
                os.mkdir(currentPath + "\\FLIP\\CLAHE")
                os.mkdir(currentPath + "\\FLIP\\GRAY")
        if self.erase > 0:
            os.mkdir(currentPath + "\\ERASE")
            if self.flipOption == 1:
                os.mkdir(currentPath + "\\ERASE\\BASE")
            if self.flipOption == 2:
                os.mkdir(currentPath + "\\ERASE\\CLAHE")
            if self.flipOption == 3:
                os.mkdir(currentPath + "\\ERASE\\GRAY")
            if self.flipOption == 4:
                os.mkdir(currentPath + "\\ERASE\\BASE")
                os.mkdir(currentPath + "\\ERASE\\CLAHE")
            if self.flipOption == 5:
                os.mkdir(currentPath + "\\ERASE\\BASE")
                os.mkdir(currentPath + "\\ERASE\\GRAY")
            if self.flipOption == 6:
                os.mkdir(currentPath + "\\ERASE\\GRAY")
                os.mkdir(currentPath + "\\ERASE\\CLAHE")
            if self.flipOption == 7:
                os.mkdir(currentPath + "\\ERASE\\BASE")
                os.mkdir(currentPath + "\\ERASE\\CLAHE")
                os.mkdir(currentPath + "\\ERASE\\GRAY")

    #In functia asta verific ce augmentari voi face si le aplic
    def applyAugmentations(self):
        self.createPaths()

        #path-uri pentru tipurile de augmentari
        clahePath = "/CLAHE"
        grayPath = "/GRAY"
        basePath = "/BASE"
        flipPath = "/FLIP"
        erasePath = "/ERASE"
        cleanPath = "/CLEAN"
        #pentru ca userul va avea optiunea de a augmenta fie pe datset de baza,
        #fie pe dataset procesat
        claheGenerated = False
        grayGenerated  = False

        #daca vrea clahe
        if self.clahe > 0:
            claheGenerated = True
            #TODO:test purpose -- AICI VOI CREA FOLDER
            print(self.datasetPath + clahePath)
            self.claheWholePath(self.datasetPath+cleanPath,self.datasetPath+clahePath)
        #daca vrea grayscale
        if self.grayscale > 0:
            grayGenerated = True
            #TODO: CREAZA FOLDER
            self.grayscaleWholePath(self.datasetPath+cleanPath, self.datasetPath+grayPath)

        #TODO: REFACTOR THE WHOLE FUNCTION @@@@@@@@@@@@@@
        #daca vrea flip, verific cu ce optiune vrea
        #Coduri optiuni: 1 - pe dataset baza | 2 - pe dataset clahe | 3 - pe dataset grayscale
        #4- pe dataset baza+clahe | 5- pe baza + grayscale | 6 - pe clahe + grayscale | 7 - pe toate 3
        if self.flip > 0:
            if self.flipOption == 1:
                #TODO:creaza folder
                self.flipAndRotateDataset(self.datasetPath+cleanPath,self.datasetPath+flipPath+basePath)
            if self.flipOption == 2:
                #vrea pe clahe, exista deja creat un folder cu clahe?
                if claheGenerated == True:
                    #TODO: creaza folder
                    #atunci doar il pasez functiei
                    self.flipAndRotateDataset(self.datasetPath+clahePath,self.datasetPath+flipPath+clahePath)
                else:
                    #TODO: creaza folder
                    #atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath+cleanPath,self.datasetPath+clahePath)
                    self.flipAndRotateDataset(self.datasetPath+clahePath,self.datasetPath+flipPath+clahePath)
                    #sa nu-l mai genereze si alta augmentare
                    claheGenerated = True
            if self.flipOption == 3:
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.flipAndRotateDataset(self.datasetPath+grayPath, self.datasetPath + flipPath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath+clahePath, self.datasetPath + grayPath)
                    self.flipAndRotateDataset(self.datasetPath+grayPath,self.datasetPath+flipPath+grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
            if self.flipOption == 4:
                self.flipAndRotateDataset(self.datasetPath+cleanPath,self.datasetPath+flipPath+basePath)
                # vrea pe clahe, exista deja creat un folder cu clahe?
                if claheGenerated == True:
                    # TODO: creaza folder
                    # atunci doar il pasez functiei
                    self.flipAndRotateDataset(self.datasetPath + clahePath, self.datasetPath + flipPath + clahePath)
                else:
                    # TODO: creaza folder
                    # atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath + cleanPath, self.datasetPath + clahePath)
                    self.flipAndRotateDataset(self.datasetPath + clahePath, self.datasetPath + flipPath + clahePath)
                    # sa nu-l mai genereze si alta augmentare
                    claheGenerated = True
            if self.flipOption == 5:
                self.flipAndRotateDataset(self.datasetPath + cleanPath, self.datasetPath + flipPath + basePath)
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.flipAndRotateDataset(self.datasetPath + grayPath, self.datasetPath + flipPath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath + clahePath, self.datasetPath + grayPath)
                    self.flipAndRotateDataset(self.datasetPath + grayPath, self.datasetPath + flipPath + grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
            if self.flipOption == 6:
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.flipAndRotateDataset(self.datasetPath + grayPath, self.datasetPath + flipPath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath + clahePath, self.datasetPath + grayPath)
                    self.flipAndRotateDataset(self.datasetPath + grayPath, self.datasetPath + flipPath + grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
                # vrea pe clahe, exista deja creat un folder cu clahe?
                if claheGenerated == True:
                    # TODO: creaza folder
                    # atunci doar il pasez functiei
                   self.flipAndRotateDataset(self.datasetPath + clahePath, self.datasetPath + flipPath + clahePath)
                else:
                    # TODO: creaza folder
                    # atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath + cleanPath, self.datasetPath + clahePath)
                    self.flipAndRotateDataset(self.datasetPath + clahePath, self.datasetPath + flipPath + clahePath)
                    # sa nu-l mai genereze si alta augmentare
                    claheGenerated = True
            if self.flipOption == 7:
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.flipAndRotateDataset(self.datasetPath + grayPath, self.datasetPath + flipPath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath + clahePath, self.datasetPath + grayPath)
                    self.flipAndRotateDataset(self.datasetPath + grayPath, self.datasetPath + flipPath + grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
                if claheGenerated == True:
                    # TODO: creaza folder
                    # atunci doar il pasez functiei
                    self.flipAndRotateDataset(self.datasetPath + clahePath, self.datasetPath + flipPath + clahePath)
                else:
                    # TODO: creaza folder
                    # atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath + cleanPath, self.datasetPath + clahePath)
                    self.flipAndRotateDataset(self.datasetPath + clahePath, self.datasetPath + flipPath + clahePath)
                    # sa nu-l mai genereze si alta augmentare
                    claheGenerated = True
                self.flipAndRotateDataset(self.datasetPath + cleanPath, self.datasetPath + flipPath + basePath)

        if self.erase > 0:
            if self.eraseOption == 1:
                #TODO:creaza folder
                self.eraseWholePath(self.datasetPath+cleanPath,self.datasetPath+erasePath+basePath)
            if self.eraseOption == 2:
                #vrea pe clahe, exista deja creat un folder cu clahe?
                if claheGenerated == True:
                    #TODO: creaza folder
                    #atunci doar il pasez functiei
                    self.eraseWholePath(self.datasetPath+clahePath,self.datasetPath+erasePath+clahePath)
                else:
                    #TODO: creaza folder
                    #atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath+cleanPath,self.datasetPath+clahePath)
                    self.eraseWholePath(self.datasetPath+clahePath,self.datasetPath+erasePath+clahePath)
                    #sa nu-l mai genereze si alta augmentare
                    claheGenerated = True
            if self.eraseOption == 3:
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.eraseWholePath(self.datasetPath+grayPath, self.datasetPath + erasePath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath+clahePath, self.datasetPath + grayPath)
                    self.eraseWholePath(self.datasetPath+grayPath,self.datasetPath+erasePath+grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
            if self.eraseOption == 4:
                self.eraseWholePath(self.datasetPath+cleanPath,self.datasetPath+erasePath+basePath)
                # vrea pe clahe, exista deja creat un folder cu clahe?
                if claheGenerated == True:
                    # TODO: creaza folder
                    # atunci doar il pasez functiei
                    self.eraseWholePath(self.datasetPath + clahePath, self.datasetPath + erasePath + clahePath)
                else:
                    # TODO: creaza folder
                    # atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath + cleanPath, self.datasetPath + clahePath)
                    self.eraseWholePath(self.datasetPath + clahePath, self.datasetPath + erasePath + clahePath)
                    # sa nu-l mai genereze si alta augmentare
                    claheGenerated = True
            if self.eraseOption == 5:
                self.eraseWholePath(self.datasetPath + cleanPath, self.datasetPath + erasePath + basePath)
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.eraseWholePath(self.datasetPath + grayPath, self.datasetPath + erasePath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath + clahePath, self.datasetPath + grayPath)
                    self.eraseWholePath(self.datasetPath + grayPath, self.datasetPath + erasePath + grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
            if self.eraseOption == 6:
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.eraseWholePath(self.datasetPath + grayPath, self.datasetPath + erasePath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath + clahePath, self.datasetPath + grayPath)
                    self.eraseWholePath(self.datasetPath + grayPath, self.datasetPath + erasePath + grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
                # vrea pe clahe, exista deja creat un folder cu clahe?
                if claheGenerated == True:
                    # TODO: creaza folder
                    # atunci doar il pasez functiei
                   self.eraseWholePath(self.datasetPath + clahePath, self.datasetPath + erasePath + clahePath)
                else:
                    # TODO: creaza folder
                    # atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath + cleanPath, self.datasetPath + clahePath)
                    self.eraseWholePath(self.datasetPath + clahePath, self.datasetPath + erasePath + clahePath)
                    # sa nu-l mai genereze si alta augmentare
                    claheGenerated = True

            if self.eraseOption == 7:
                # vrea pe gray, exista deja creat un folder cu grayscale?
                if grayGenerated == True:
                    # atunci doar il pasez functiei
                    self.eraseWholePath(self.datasetPath + grayPath, self.datasetPath + erasePath + grayPath)
                else:
                    # atunci il generez si abea dupa aplic pe el flip
                    self.grayscaleWholePath(self.datasetPath + clahePath, self.datasetPath + grayPath)
                    self.eraseWholePath(self.datasetPath + grayPath, self.datasetPath + erasePath + grayPath)
                    # sa nu-l mai genereze si alta augmentare
                    grayGenerated = True
                if claheGenerated == True:
                    # TODO: creaza folder
                    # atunci doar il pasez functiei
                    self.eraseWholePath(self.datasetPath + clahePath, self.datasetPath + erasePath + clahePath)
                else:
                    # TODO: creaza folder
                    # atunci il generez si abea dupa aplic pe el flip
                    self.claheWholePath(self.datasetPath + cleanPath, self.datasetPath + clahePath)
                    self.eraseWholePath(self.datasetPath + clahePath, self.datasetPath + erasePath + clahePath)
                    # sa nu-l mai genereze si alta augmentare
                    claheGenerated = True
                self.eraseWholePath(self.datasetPath + cleanPath, self.datasetPath + erasePath + basePath)
        #arhivez si sterg ce nu-mi trebuie
        self.zipAugmentations()
    def claheWholePath(self, path, savePath):
        howManyClahe = 0
        for filename in os.listdir(path):
            #probabilitatea cu care se intampla CLAHE pe dataset
            if rand.random() < self.clahe:
                # read the image
                img = cv2.imread(os.path.join(path, filename))
                #daca este imagine si exista
                if img is not None:
                    #creez o copie, ca sa pot edita imaginea pe ea
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    cv2.imwrite(os.path.join(savePath, filename), gray)
                    howManyClahe += 1
        print(str(howManyClahe) + " out of " + str(len(os.listdir(path))) + " for CLAHE")


    def grayscaleWholePath(self, path, savePath):
        howManyGrayscaled = 0
        for filename in os.listdir(path):
            if rand.random() < self.grayscale:
                #citesc imaginea
                img = cv2.imread(os.path.join(path, filename))
                # daca exista
                if img is not None:
                    #o copiez ca sa o pot edita
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(savePath, filename), gray)
                    howManyGrayscaled += 1
        #TESTING PURPOSE
        print(str(howManyGrayscaled) + " out of " + str(len(os.listdir(path))) + " for Grayscale")

    def flipAndRotateDataset(self, path, savePath):
        rotationValues = [0, 90, 180, 270]
        # TESTING PURPOSE
        howManyFlipped = 0
        howManyRotated = 0

        # iau fiecare fisier din folderul dat
        for filename in os.listdir(path):
            #citesc imaginea
            img = cv2.imread(os.path.join(path, filename))
            #daca exista
            if img is not None:
                #creez o copie pentru a o edita
                imageCopy = np.copy(img)

                #o rotesc cu probabilitatea self.rotate
                if rand.random() < self.rotate:
                    #rotesc imaginea cu o valoare random care apartinte intervalului {0,90,180,270}
                    rotationValue = rand.choice(rotationValues)
                    imageCopy = imutils.rotate_bound(imageCopy, rotationValue)
                    howManyRotated += 1
                #probabilitatea pentru a o flipui
                if rand.random() < self.flip:
                    imageCopy = cv2.flip(imageCopy, 1)
                    howManyFlipped += 1
                #salvez imaginea
                cv2.imwrite(os.path.join(savePath, filename), imageCopy)

        # TESTING PURPOSE
        print(str(howManyRotated) + " out of " + str(len(os.listdir(path))) + " for Rotations")
        print(str(howManyFlipped) + " out of " + str(len(os.listdir(path))) + " for Flip")

    def eraseWholePath(self, path, savePath):
        self.randomEraser.eraseWholePath(path,savePath)


    def zipAugmentations(self):
        currentPath = self.datasetPath+"\\"

        #am terminat cu augmentarile, deci vom sterge setul de date CLEAN si vom arhiva tot restul
        # TODO:rezolva trycatch asta
        pathToRemove = self.datasetPath + "\\CLEAN"


        shutil.rmtree(pathToRemove,ignore_errors=True)

        nameOfArchive = self.archiveName.split('.')[0]
        zipf = zipfile.ZipFile(self.saveArchivePath+"\\"+self.archiveName, 'w', zipfile.ZIP_DEFLATED)
        self.zipdir(currentPath,zipf)
        #self.removeArchiveFromItself(currentPath+self.archiveName,currentPath+nameOfArchive+"2.zip")

        #acum sterg directoarele pentru ca sunt deja in arhiva
        directory_contents = os.listdir(self.datasetPath)
        for item in directory_contents:
            # daca e director il adaug in arhiva
            if os.path.isdir(currentPath+item):
                shutil.rmtree(currentPath+item,ignore_errors=True)

    def removeArchiveFromItself(self, path1,path2):
        zin = zipfile.ZipFile(path1, 'r')
        zout = zipfile.ZipFile(path2, 'w')
        for item in zin.infolist():
            buffer = zin.read(item.filename)
            if item.filename.split('.')[1] != ".zip":
                zout.writestr(item, buffer)
        zout.close()
        zin.close()

    def zipdir(self,path, ziph):
        # ziph este zipfile in care adaug
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))