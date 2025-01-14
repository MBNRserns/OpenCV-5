import cv2
import numpy as np

img=cv2.imread("blobs.jpeg")

params = cv2.SimpleBlobDetector_Params()

params.filterByArea=True
params.minArea = 100

params.filterByCircularity=True
params.minCircularity=0.8

params.filterByInertia=True
params.minInertiaRatio=0.01

params.filterByConvexity=True
params.minConvexity=0.9

detector=cv2.SimpleBlobDetector_create(params)

keypoints=detector.detect(img)
print(keypoints)

number_of_blobs=len(keypoints)
print("Number of Blobs:",number_of_blobs)

blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(img,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#drawKeypoints(input_image, key_points, output_image, colour, flag)

text=cv2.putText(blobs,"Amount of Circular Blobs: "+str(number_of_blobs), (15,550), (cv2.FONT_HERSHEY_PLAIN),2,(255,0,255),2)


cv2.imshow("Blobs",blobs)

cv2.waitKey(0)
cv2.destroyAllWindows()