import cv2 as cv
import random


W = 64
H = 128
sample_time = 10
name_num = 0

# 创建负样本
str_fa = r'D:/programme/INRIAPerson/INRIAPerson/'  # source文件路径
str_son = r'C:/Users/WangZhuo/Desktop/Pedestrian Detection/'  # 样本路径
f = open(r'D:/programme/INRIAPerson/INRIAPerson/Train/neg.lst', 'r')  # source_list路径
fw = open(r'C:/Users/WangZhuo/Desktop/Pedestrian Detection/Train_64x128/neg.txt', 'w')  # 样本list路径
str_dir = f.read().split("\n")
str_dir.remove(str_dir[len(str_dir)-1])

for string in str_dir:
    map_name00 = 0
    string = str_fa + string

    img = cv.imread(string)
    h, w, c = img.shape
    if (w >= W) & (h >= H):
        while (map_name00 <= sample_time):
            x = random.randint(0, 1000) % (w - W)
            y = random.randint(0, 1000) % (h - H)
            imgROI = img[y:y + H, x:x + W]
            str_son_final = str_son + r'Train_64x128/neg/' + str(name_num) + r'.png'
            cv.imwrite(str_son_final, imgROI)
            fw.write(str(name_num) + ".png" + "\n")
            map_name00 = map_name00 + 1
            name_num = name_num + 1
f.close()
fw.close()