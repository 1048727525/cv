import cv2 as cv
import cv2

t1 = cv.getTickCount()
basic_path = r"C:/Users/WangZhuo/Desktop"  # 文件夹所在目录
hog = cv2.HOGDescriptor()
hog.load('myHogDector_1.bin')
cap = cv.VideoCapture("1.avi")
i = 0
while True:
    ok, img = cap.read()
    rects, wei = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv.imwrite(basic_path + r"/Pedestrian Detection/Test/Frames/" + str(i) + r".png", img)
    i = i + 1
    cv.imshow('a', img)
    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()
    print("time : %s ms" % (time * 1000))
    if cv2.waitKey(33) == 27:  # esc键
        break
cv2.destroyAllWindows()