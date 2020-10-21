import numpy as np
from sklearn.cluster import KMeans
import cv2
import glob
import os, shutil, glob, os.path
import matplotlib.pyplot as plt

def vectorisation(image, vocabulaire):
    greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(greyImage,None)
    #FAIRE LE VECTEUR
    return #VECTEUR

def baseApprentissage():
    files = ['llama', 'pizza', 'octopus', 'lotus']
    first = True
    filenameList = []

    for fn in files:
        for fileName in glob.glob('images/'+fn+'/train/*.jpg'):
            image = cv2.imread(fileName)
            if first:
                first = False
                #PRECISER VOCABULAIRE
                matriceVectors = np.array(vectorisation(image, vocabulaire))
            else:
                #PRECISER VOCABULAIRE
                np.append(matriceVectors, vectorisation(image, vocabulaire))
            filenameList.append(fileName)
            np.savetxt('filenames.txt', fileName)

    return filenameList, matriceVectors

def createVocabulaire():
    files = ['llama', 'pizza', 'octopus', 'lotus']
    N = 100
    first = True

    for fn in files:
        for fileName in glob.glob('images/'+fn+'/train/*.jpg'):  #mettre train à la place de test
            image = cv2.imread(fileName)
            greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(greyImage,None)
            if first:
                first = False
                siftDescriptors = np.array(descriptors)
            else:
                np.append(siftDescriptors, np.array(descriptors))

    kmeans = KMeans(n_clusters=N, random_state=0).fit(siftDescriptors)
    centers = kmeans.cluster_centers_
    np.savetxt('center.txt', centers)

    return kmeans, siftDescriptors

def varianceTotale(siftDescriptors):
    totalVariance = np.zeros((100,))
    for K in range(1,100):
        clust = KMeans(n_clusters=K,n_init=3,verbose=0)
        Ckm=clust.fit_predict(siftDescriptors)
        totalVariance[K] = clust.inertia_

    #Affichage de la variance totale
    plt.plot(np.arange(1,100),totalVariance[1:100], color = 'red', linestyle = 'dashed', linewidth = 2, markerfacecolor = 'blue', markersize = 5)
    plt.show()

def main():
    vocabulaire, siftDescriptors = createVocabulaire()
    varianceTotale(siftDescriptors)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()