import cv2
import numpy as np

facedetect =cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
eyedetect=cv2.CascadeClassifier('haarcascade_eye.xml');

cam=cv2.VideoCapture(0);

while(True):
    #image read 
    ret,img=cam.read();
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eyedetect.detectMultiScale(roi_gray);
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    #show image
    cv2.imshow("Face",img);
    if(cv2.waitKey(10)==ord('q')):
       break;
cam.release()
cv2.destroyAllWindows()
       
    

