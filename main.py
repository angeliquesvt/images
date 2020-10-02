import numpy as np
from sklearn.cluster import KMeans
import cv2
import glob

def main():
    #siftDescripteurs = np.array()
    files = ['llama', 'pizza', 'octopus', 'lotus']
    N = 100
    first = True

    for fn in files:
        for fileName in glob.glob('images/'+fn+'/train/*.jpg'):  #mettre train à la place de test
            

            image = cv2.imread(fileName)
            greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints, descripteurs = sift.detectAndCompute(greyImage,None)
            if first:
                first = False
                siftDescripteurs = np.array(descripteurs)
            else:
                np.append(siftDescripteurs, np.array(descripteurs))
            #cv2.drawKeypoints(greyImage,keypoints,image)
            #cv2.imshow('sift_keypoints',image)
    #arr d'ar, conv en np array en float
    kmeans = KMeans(n_clusters=N, random_state=0).fit(siftDescripteurs)
    centers = kmeans.cluster_centers_
    np.savetxt('center.txt', centers)
    print(centers.var())
    cv2.waitKey(0)

if __name__ == "__main__":
    main()