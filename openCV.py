# -*- coding: gb2312 -* 
import cv2
filepath = "D:\\img2.jpg"
img = cv2.imread(filepath)
# 读取图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
classifier = cv2.CascadeClassifier(
"C:\\Users\\Zcat\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\opencv_python-4.10.0.84.dist-info\\data\\haarcascades\\haarcascade_frontalface_default.xml"
)
color = (0, 255, 0)  # 定义绘制颜色
# 调用识别人脸
faceRects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(32, 32))
if len(faceRects):  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        # 框出人脸
        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
        # 左眼
        cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        #右眼
        cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        #嘴巴
        cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                      (x + 5 * w // 8, y + 7 * h // 8), color)

cv2.imshow("image", img)  # 显示图像
c = cv2.waitKey(10)

cv2.waitKey(0)
cv2.destroyAllWindows()


#################################################摄像头版本#################################################
# -*- coding: gb2312 -* 
# OpenCV版本的视频检测
import cv2
# 图片识别方法封装
def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(
       "C:\\Users\\Zcat\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\opencv_python-4.10.0.84.dist-info\\data\\haarcascades\\haarcascade_frontalface_default.xml"
    )
    color = (0, 255, 0)
    faceRects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y - w//3), (x + h, y + w +w//20), (0, 255, 0), 2)  # 框出人脸
             # 左眼
        cv2.circle(img, (x + 9 * w // 32, y + h // 4 + 20), min(w // 10, h // 10),color)
        #右眼
        cv2.circle(img, (x + 23 * w // 32, y + h // 4 + 20), min(w // 10, h // 10),color)
        
        #嘴巴
        cv2.ellipse(img, (x + 3 * w // 6, y + 25 * h // 32),(w // 8, h // 14),0,0,360, color,1)
     

    cv2.imshow("Image", img)

# 获取摄像头0表示第一个摄像头
cap = cv2.VideoCapture(0)
while (1):  # 逐帧显示
    ret, img = cap.read()
    # cv2.imshow("Image", img)
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源

#################################################本地视频版本#################################################
# -*- coding: gb2312 -* 
import cv2
import tkinter as tk
from tkinter import filedialog

def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(
       "C:\\Users\\Zcat\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\opencv_python-4.10.0.84.dist-info\\data\\haarcascades\\haarcascade_frontalface_default.xml"
    )
    color = (0, 255, 0)
    faceRects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):
        for i, faceRect in enumerate(faceRects):
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y - w//20), (x + h, y + w + w//20), color, 2)  # 框出人脸  
            #左眼
            cv2.circle(img, (x + 10 * w // 32, y + 10 * h // 27 ), min(w // 10, h // 10), color)
            #右眼  
            cv2.circle(img, (x + 22 * w // 32, y +10 * h // 27 ), min(w // 10, h // 10), color)
            #嘴巴
            cv2.ellipse(img, (x + 3 * w // 6, y + 25 * h // 32), (w // 8, h // 14), 0, 0, 360, color, 1)
            # 在人脸周围绘制序号
            cv2.putText(img, f'{i + 1}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.imshow("Image", img)


global video_path
video_path = filedialog.askopenfilename()#读取视频文件
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)#摄像头模式
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        discern(img)
        if cv2.waitKey(1) & 0xFF == ord(' '):
        
            break
    else:
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源
