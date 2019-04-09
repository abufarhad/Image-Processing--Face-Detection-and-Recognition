import cv2
import numpy as np

facedetect =cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

while(True):
    #image read 
    ret,img=cam.read();
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #show image
    cv2.imshow("Face",img);
    if(cv2.waitKey(10)==ord('q')):
       break;
cam.release()
cv2.destroyAllWindows()
       
    

