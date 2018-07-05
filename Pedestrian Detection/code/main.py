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


def computeHOGs(img_lst, gradient_lst, wsize=(128, 64)):
    hog = cv.HOGDescriptor()
    # hog.winSize = wsize
    for i in range(len(img_lst)):
        print("computeHOGing " + str(i) + ".png")
        h, w, c = img_lst[i].shape
        if w >= wsize[1] and h >= wsize[0]:
            # 产生一个128x64的矩形框
            roi = img_lst[i][(h - wsize[0]) // 2: (h - wsize[0]) // 2 + wsize[0],
                  (w - wsize[1]) // 2: (w - wsize[1]) // 2 + wsize[1]]
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            gradient_lst.append(hog.compute(gray))
    print("computeHOGing fininshed")
    # return gradient_lst


def get_svm_detector(svm):  # 返回分类器参数
    sv = svm.getSupportVectors()  # get the support vectors
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


# train_dir:训练集所在目录
# train_list:训练集目录
# bin：训练所用的bin文件
# time：第几次训练
# hard_adr：错误图片所存目录，一般默认
def train(train_dir, train_list, bin, time, hard_adr=basic_path + r"/Pedestrian Detection/hard_neg/"):
    test_list = []
    hog = cv2.HOGDescriptor()
    hog.load(bin)

    f2 = open(train_list)
    test_list_name = f2.read().split("\n")
    f2.close()
    test_list_name.remove(test_list_name[len(test_list_name) - 1])

    i = 0
    for string in test_list_name:
        string = train_dir + string
        test_list.append(cv.imread(string))
    f3 = open(hard_adr + "hard_" + str(time) + ".txt", 'w')
    print(len(test_list))
    for img in test_list:
        rects, wei = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        # winStride：步幅  pad：内边距  scale：向外重合多少倍  finalThreshold：探测多少次才允许定结论
        for (x, y, w, h) in rects:
            imgROI = img[y:y + h, x:x + w]
            imgROI = cv2.resize(imgROI, (64, 128), interpolation=cv2.INTER_CUBIC)  # 将检测错误的地方变成64x128存储
            cv.imwrite((hard_adr + "hard_" + str(time) + "/" + "hard_" + str(time) + "_" + str(i) + ".png"), imgROI)
            print("Writing " + hard_adr + "hard_" + str(time) + "/" + "hard_" + str(time) + "_" + str(i) + ".png")
            f3.write("hard_" + str(time) + "_" + str(i) + ".png" + '\n')
            i = i + 1
    f3.close()
    print("Finished")


neg_list = []
pos_list = []
gradient_lst = []
labels = []
t1 = cv.getTickCount()
# get HOG and Label
neg_list = load_images(basic_path + r"/Pedestrian Detection/Train_64x128/neg",
                       basic_path + r"/Pedestrian Detection/Train_64x128/neg.txt")
computeHOGs(neg_list, gradient_lst)
for i in range(len(neg_list)):
    labels.append(-1)  # 负样本用-1标识

pos_list = load_images(basic_path + r"/Pedestrian Detection/Train_64x128/pos",
                       basic_path + r"/Pedestrian Detection/Train_64x128/pos.txt")
computeHOGs(pos_list, gradient_lst)
for i in range(len(pos_list)):
    labels.append(+1)  # 正样本用+1标识

# get SVM
# 这里的SVM初始化采用的default，参考网址：https://docs.opencv.org/3.4.0/d5/d77/train_HOG_8cpp-example.html
svm = cv2.ml.SVM_create()
svm.setCoef0(0)
svm.setCoef0(0.0)
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
svm.setTermCriteria(criteria)
svm.setGamma(0)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
svm.setC(0.01)  # From paper, soft classifier
svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
print("Training, loading...")
svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))
hog = cv2.HOGDescriptor()
hog.setSVMDetector(get_svm_detector(svm))  # 为SVM分类器设置系数
hog.save('myHogDector.bin')
now_bin = 'myHogDector.bin'

train_time = 1 # 自举法训练次数，参考：http://masikkk.com/article/SVM-HOG-HardExample/
now_time = 1
while now_time <= train_time:

    train(basic_path + r'/Pedestrian Detection/',
          basic_path + r'/Pedestrian Detection/Train/neg.lst',
          now_bin, now_time)

    neg_list = load_images(basic_path + r"/Pedestrian Detection/hard_neg/" + "hard_" + str(now_time),
                           basic_path + r"/Pedestrian Detection/hard_neg/" + "hard_" + str(now_time) + '.txt')
    computeHOGs(neg_list, gradient_lst)
    for i in range(len(neg_list)):
        labels.append(-1)
    print("No "+str(now_time)+": Training, loading...")

    svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(get_svm_detector(svm))
    hog.save('myHogDector_' + str(now_time) + '.bin')
    now_bin = 'myHogDector_' + str(now_time) + '.bin'
    now_time = now_time + 1
t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()
print("All Finished")
print("time : %s s" % (time))