import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import cv2
import glob
import os, shutil, glob, os.path
import matplotlib.pyplot as plt

def varianceTotale(siftDescriptors):
    totalVariance = np.zeros((100,))
    for K in range(1,100):
        clust = KMeans(n_clusters=K,n_init=3,verbose=0)
        Ckm=clust.fit_predict(siftDescriptors)
        totalVariance[K] = clust.inertia_

    #Affichage de la variance totale
    plt.plot(np.arange(1,100),totalVariance[1:100], color = 'red', linestyle = 'dashed', linewidth = 2, markerfacecolor = 'blue', markersize = 5)
    plt.show()

def createVocabulaire():
    files = ['llama', 'pizza', 'octopus', 'lotus']
    N = 100
    first = True

    for fn in files:
        for fileName in glob.glob('images/'+fn+'/train/*.jpg'):  #mettre train à la place de test
            print(fileName)
            image = cv2.imread(fileName)
            greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(greyImage,None)
            if first:
                first = False
                siftKeyPoints = np.array(keypoints)
                siftDescriptors = np.array(descriptors)
                vocFileNames= np.array(fileName)
            else:
                np.append(siftDescriptors, np.array(descriptors))
                np.append(siftKeyPoints, np.array(siftKeyPoints))
                np.append(vocFileNames, fileName)

    kmeans = KMeans(n_clusters=N, random_state=0).fit(siftDescriptors)
    centers = kmeans.cluster_centers_
    np.savetxt('center.txt', centers)
    print(np.shape(centers))
    return centers, siftDescriptors

def vectorisation(image, vocabulaire):
    K=1
    greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(greyImage,None)

    tree = KDTree(vocabulaire, leaf_size=2)
    vect = np.zeros(np.shape(vocabulaire)[0])
    #for des in descriptors:
    dists, inds = tree.query(descriptors, k=K)
    #print(inds)  # indices of 3 closest neighbors
    #print(dists)  # distances to 3 closest neighbors
    for ind in inds:
        vect[ind]=vect[ind]+1

    print(vect)    
    print("******************************")

    #FAIRE LE VECTEUR
    return #VECTEUR

def baseApprentissage(vocabulaire):
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
    #np.savetxt('filenames.txt', filenameList)

    return filenameList, matriceVectors

def main():
    vocabulaire, siftDescriptors = createVocabulaire()
    #varianceTotale(siftDescriptors)
    baseApprentissage(vocabulaire)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()