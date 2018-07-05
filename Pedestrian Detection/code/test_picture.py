import cv2 as cv
import cv2
import numpy as np


basic_path = r"C:/Users/WangZhuo/Desktop"  # 文件夹所在目录


def load_images(dirname, list):  # 通过文本目录索引图片，加载图片数组
    img_list = []
    f1 = open(list)
    img_name_dir = f1.read().split("\n")
    f1.close()
    img_name_dir.remove(img_name_dir[len(img_name_dir) - 1])  # 最后一个元素为空
    for string in img_name_dir:
        string = dirname + r'/' + string
        img_list.append(cv.imread(string))
        print("Loading " + string)
    print("Loading fininshed")
    return img_list


neg_list = []
pos_list = []
neg_list = load_images(basic_path + r"/Pedestrian Detection", basic_path + r"/Pedestrian Detection/Test/neg.lst")
pos_list = load_images(basic_path + r"/Pedestrian Detection", basic_path + r"/Pedestrian Detection/Test/pos.lst")
hog = cv2.HOGDescriptor()
hog.load('myHogDector.bin')
i = 0

for img in neg_list:
    image = np.zeros(img.shape, np.uint8)
    image = img.copy()
    rects, wei = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    #cv.imwrite(basic_path+r"/Pedestrian Detection/Test/Result/neg/"+str(i)+r".png", image)
    i = i+1


for img in pos_list:
    image = np.zeros(img.shape, np.uint8)
    image = img.copy()
    rects, wei = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    #cv.imwrite(basic_path+r"/Pedestrian Detection/Test/Result/pos/"+str(i)+r".png", image)
    i = i+1

