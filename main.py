import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import cv2
import glob
import os, shutil, glob, os.path
import matplotlib.pyplot as plt

def varianceTotale(siftDescriptors):
    print("start varianceTotale")
    totalVariance = np.zeros((100,))
    for K in range(1,100):
        clust = KMeans(n_clusters=K,n_init=3,verbose=0)
        Ckm=clust.fit_predict(siftDescriptors)
        totalVariance[K] = clust.inertia_

    #Affichage de la variance totale
    plt.plot(np.arange(1,100),totalVariance[1:100], color = 'red', linestyle = 'dashed', linewidth = 2, markerfacecolor = 'blue', markersize = 5)
    plt.show()
    print("end varianceTotale")


def createVocabulaire():
    print("start createVocabulaire")
    files = ['llama', 'pizza', 'octopus', 'lotus']
    N = 100
    first = True

    for fn in files:
        for fileName in glob.glob('images/'+fn+'/train/*.jpg'):  #mettre train à la place de test
            #print(fileName)
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
    #print(np.shape(centers))
    print("end createVocabulaire")

    return centers, siftDescriptors

def vectorisation(image, vocabulaire):
    #print("start vectorisation")
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

    #print(vect)    
    #print("******************************")
    #print("end vectorisation")
    return vect

def baseApprentissage(vocabulaire):
    print("start baseApprentissage")
    files = ['llama', 'pizza', 'octopus', 'lotus']
    first = True
    filenameList=[]
    base = []
    for fn in files:
        for fileName in glob.glob('images/'+fn+'/train/*.jpg'):
            image = cv2.imread(fileName)
            #if first:
            #    first = False
            #    matriceVectors = np.array(vectorisation(image, vocabulaire))
            #else:
            #    np.append(matriceVectors, vectorisation(image, vocabulaire))
            base.append(vectorisation(image, vocabulaire))
            filenameList.append(fileName)

    with open('filenamelist.txt', 'w') as f:
        for item in filenameList:
            f.write("%s\n" % item)
    print("end baseApprentissage")
    return filenameList, base

def tests(image, vocabulaire):
    filenameList, matriceVectors = baseApprentissage(vocabulaire)
    print(matriceVectors)
    vect = vectorisation(image, vocabulaire)
    tree = KDTree(matriceVectors)
    print('test2')
    #dists, inds = tree.query(vect, k=1)
    print('-------end tests-------')
    #dists, inds = tree.query(vect, k=1)
    return dists, inds

def main():
    print("start main")
    vocabulaire, siftDescriptors = createVocabulaire()
    #varianceTotale(siftDescriptors)
    #filenameList, matriceVectors = baseApprentissage(vocabulaire)
    image = cv2.imread('images/llama/train/image_0005.jpg')
    tests(image,vocabulaire)
    print("end main")
    cv2.waitKey(0)

if __name__ == "__main__":
    main()