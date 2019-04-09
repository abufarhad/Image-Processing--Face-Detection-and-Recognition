import cv2
import numpy as np

facedetect =cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

id=raw_input('Enter user id')
sampleNum=0;

while(True):
    # Capture frame-by-frame
    ret,img=cam.read();
    
    # convertingto gray scale 
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces=facedetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataset/user."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w]);
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    #show image
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    if(sampleNum>20):
        break;
        
cam.release()
cv2.destroyAllWindows()
       
    

