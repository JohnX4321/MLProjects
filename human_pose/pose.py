#pyimagesearch
import mediapipe as mp
import cv2,os
import time,numpy as np,pandas as pd

mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils
points=mpPose.PoseLandmark
path=""
data=[]
for p in points:
    x=str(p)[13:]
    data.append(x+"_x")
    data.append(x+"_y")
    data.append(x+"_z")
    data.append(x+"_vis")
data=pd.DataFrame(columns=data)


count=0
for img in os.listdir(path):
    temp=[]
    img=cv2.imread(path+"/"+img)
    iw,ih=img.shape[:2]
    irgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    blackie=np.zeros(img.shape)
    res=pose.process(irgb)
    if res.pose_landmarks:
        mpDraw.draw_landmarks(blackie,res.pose_landmarks,mpPose.POSE_CONNECTIONS)
        landmarks=res.pose_landmarks.landmark
        for i,j in zip(points,landmarks):
            temp+=[j.x,j.y,j.z,j.visibility]
        data.loc[count]=temp
        count+=1
    cv2.imshow("Image",img)
    cv2.imshow("blackie",blackie)
    cv2.waitKey(100)
data.to_csv("d3.csv")

from sklearn.svm import SVC
data=pd.read_csv("d3.csv")
x,y=data.iloc[:,:132],data['target']
model=SVC(kernel='poly')
model.fit(x,y)
mpPose=mp.solutions.pose
pose=mp.Pose()
mpDraw = mp.solutions.drawing_utils
path = "enter image path"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(imgRGB)
if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for j in landmarks:
                temp = temp + [j.x, j.y, j.z, j.visibility]
        y = model.predict([temp])
        if y == 0:
            asan = "plank"
        else:
            asan = "goddess"
        print(asan)
        cv2.putText(img, asan, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
        cv2.imshow("image",img)