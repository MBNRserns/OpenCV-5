import cv2
import numpy as np

img=cv2.imread("Eyes.jpg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur=cv2.blur(gray, (3,3))

detected_circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,param1 = 20,param2=10,minRadius=1,maxRadius=6)

print(detected_circles)

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    print(detected_circles)
    for i in detected_circles[0, :]:
        x,y,r=i[0],i[1],i[2]
        cv2.circle(img, (x,y), r, (0,255,0), 2)
        cv2.circle(img,(x,y),1,(0,10,255),2)

cv2.imshow("Eyes",img)

cv2.waitKey(0)
cv2.destroyAllWindows()