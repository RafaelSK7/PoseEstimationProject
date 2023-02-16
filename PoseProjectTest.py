import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('videos/2.mp4')
pTime = 0
detector = pm.poseDetector()


def resizeImg(img):
    scale_percent = 25  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resizedImg = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resizedImg


while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    # if I want to print only a unique point, I set draw=False in findPosition, and add this code
    if len(lmList) != 0:
        print(lmList)
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    img = resizeImg(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
