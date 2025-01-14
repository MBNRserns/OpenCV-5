import cv2
import numpy as np

img=cv2.imread("blobs.jpeg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur=cv2.blur(gray, (3,3))

"""
Detection Method: OpenCV has an advanced implementation, HOUGH_GRADIENT, 
which uses gradient of the edges instead of filling up the entire 3D accumulator matrix, thereby speeding up the process.
dp: This is the ratio of the resolution of original image to the accumulator matrix.
minDist: This parameter controls the minimum distance between detected circles.
Param1: Canny edge detection requires two parameters — minVal and maxVal. Param1 is the higher threshold of the two. The second one is set as Param1/2.
Param2: This is the accumulator threshold for the candidate detected circles. By increasing this threshold value, we can ensure that only the best circles, corresponding to larger accumulator values, are returned.
minRadius: Minimum circle radius.
maxRadius: Maximum circle radius.
"""

detected_circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,param1 = 50,param2=30,minRadius=1,maxRadius=40)

print(detected_circles)

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    print(detected_circles)
    for i in detected_circles[0, :]:
        x,y,r=i[0],i[1],i[2]
        cv2.circle(img, (x,y), r, (0,255,0), 2)
        cv2.circle(img,(x,y),1,(0,10,255),2)
    cv2.imshow("Blobs",img)

cv2.waitKey(0)
cv2.destroyAllWindows()