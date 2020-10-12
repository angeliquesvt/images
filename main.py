import numpy as np
from sklearn.cluster import KMeans
import cv2
import glob
import os, shutil, glob, os.path
import matplotlib.pyplot as plt

def vectorisation():
    return

def main():
    #siftDescriptors = np.array()
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
            #cv2.drawKeypoints(greyImage,keypoints,image)
            #cv2.imshow('sift_keypoints',image)
            #Z = np.random.rand(2000,2000)
            #plt.imsave('center/%03i.png'%i,Z)

    kmeans = KMeans(n_clusters=N, random_state=0).fit(siftDescriptors)
    centers = kmeans.cluster_centers_
    np.savetxt('center.txt', centers)

    totalVariance = np.zeros((500,))
    for K in range(1,100):
        clust = KMeans(n_clusters=K,n_init=3,verbose=0)
        Ckm=clust.fit_predict(siftDescriptors)
        totalVariance[K] = clust.inertia_
        #print(J[K])

    plt.plot(np.arange(1,500),totalVariance[1:500], color = 'red', linestyle = 'dashed', linewidth = 2, markerfacecolor = 'blue', markersize = 5)
    plt.show()
    #pplt.title('Variance totale')
    #for i, m in enumerate(kmeans.labels_):
    #    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    #    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")

    cv2.waitKey(0)

if __name__ == "__main__":
    main()