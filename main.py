import numpy as np
from sklearn.cluster import KMeans
import cv2
import glob

def vectorisation():
    return

def main():
    #siftDescriptors = np.array()
    files = ['llama', 'pizza', 'octopus', 'lotus']
    N = 100
    first = True
    kmeans = KMeans(n_clusters=N, random_state=0)

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

    labels = kmeans.fit_predict(siftDescriptors)
    centers = kmeans.cluster_centers_
    np.savetxt('center.txt', centers)
    
    print(labels)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()