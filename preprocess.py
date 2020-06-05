from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import os



# 获取图片
def getimg(path):
    return cv2.imread(path,2)


# 显示图片
def showimg(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    # 释放窗口
    cv2.destroyAllWindows()

#归一化
def normalize(img):
    x = img.astype(float)
    new = []
    mean = np.mean(x)
    var = np.var(x)
    for i in range(300):
        tmp = []
        for j in range(300):
            if x[i][j] >= mean:
                tp = (150 + ((2000 * (x[i][j] - mean) ** 2) / var) ** 0.5)
            else:
                tp = (150 - ((2000 * (x[i][j] - mean) ** 2) / var) ** 0.5)
            tmp.append(int(tp))
        new.append(tmp)
    return np.array(new).astype(np.uint8)


#分区归一化（默认分区大小为10*10）
def part_normalize(img):
    x = img.astype(float)
    mean = np.mean(x)
    var = np.var(x)
    new = np.zeros(shape=(10, 310))
    for i in range(30):
        tmp = np.zeros(shape=(10, 10))
        for j in range(30):
            part = x[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10]
            tpmean = np.mean(part)
            tpvar = np.var(part)
            partnew=[]
            for ii in range(i * 10,(i + 1) * 10):
                parttmp = []
                for jj in range(j * 10,(j + 1) * 10):
                    if tpvar==0:
                        tp=x[ii][jj]
                    elif x[ii][jj] >= tpmean:
                        tp = (mean + ((var * (x[ii][jj] - tpmean) ** 2) / tpvar) ** 0.5)
                    else:
                        tp = (mean - ((var * (x[ii][jj] - tpmean) ** 2) / tpvar) ** 0.5)
                    parttmp.append(int(tp))
                partnew.append(parttmp)
            partnew=np.array(partnew)
            tmp = np.hstack((tmp,partnew))
        new = np.vstack((new,tmp))
    new=new[10:, 10:]
    return new.astype(np.uint8)


#图像分割
def sperate(img):
    x = img.astype(float)
    #mean = np.mean(x)
    #var = np.var(x)
    mean_array = []
    var_array = []
    for i in range(100):
        tpmean = []
        tpvar = []
        for j in range(100):
            part = x[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
            tpmean.append(np.mean(part))
            tpvar.append(np.var(part))
        mean_array.append(tpmean)
        var_array.append(tpvar)
    mean_array = np.array(mean_array)
    var_array = np.array(var_array)
    gmean = np.mean(mean_array)
    gvar = np.mean(var_array)
    tpmean = []
    tpvar = []
    for i in range(100):
        for j in range(100):
            if mean_array[i][j] < gmean:
                tpmean.append(mean_array[i][j])
            if var_array[i][j] > gvar:
                tpvar.append(var_array[i][j])
    formean = np.mean(np.array(tpmean))
    forvar = np.mean(np.array(tpvar))
    tpmean = []
    tpvar = []
    for i in range(100):
        for j in range(100):
            if mean_array[i][j] > formean:
                tpmean.append(mean_array[i][j])
            if var_array[i][j] < forvar:
                tpvar.append(var_array[i][j])
    backmean = np.mean(np.array(tpmean))
    backvar = np.mean(np.array(tpvar))
    ele = np.zeros([100, 100])
    for i in range(100):
        for j in range(100):
            if mean_array[i][j] > backmean and var_array[i][j] < forvar/2:
                ele[i][j] = 1
            if mean_array[i][j] < formean - 100 and var_array[i][j] < forvar/2:
                ele[i][j] = 1
    for i in range(1, 99):
        for j in range(1, 99):
            tp = ele[i-1][j-1] + ele[i-1][j] + ele[i-1][j+1] + ele[i][j-1] + ele[i][j+1] + ele[i+1][j-1] + ele[i+1][j] + ele[i+1][j+1]
            if tp <= 4:
                ele[i][j] = 0
    Icc = np.ones([300, 300])
    for i in range(100):
        for j in range(100):
            if ele[i][j]==1:
                for ii in range(i * 3, (i + 1) * 3):
                    for jj in range(j * 3, (j + 1) * 3):
                        x[ii][jj] = int(formean)+40
                        Icc[ii][jj] = 0
    Icc = Icc * 255
    Icc = Icc.astype(np.uint8)
    #showimg(Icc)
    print(formean)
    return x.astype(np.uint8), Icc, formean

#均值模糊
def avg_blur(image):
    dst = cv2.blur(image, (3, 3))
    #cv2.imshow("avg_blur_demo", dst)
    return dst

#中位数模糊
def median_blur(image):
    dst = cv2.medianBlur(image, 3)
    cv2.imshow("median_blur_demo", dst)
    return dst

#高斯模糊
def guassian_blur(image):
    dst = cv2.GaussianBlur(image, (3, 3), 1)
    cv2.imshow("guassian_blur_demo", dst)
    return dst

#二值化
def binarization(img,num):
    ret, thred = cv2.threshold(img, num, 255, cv2.THRESH_BINARY)
    return thred


#脊线方向二值化（效果并不好，不建议使用）
def aug_binarization(img, icc):
    img = avg_blur(img)
    im = np.zeros([300, 300])
    for i in range(4, 296):
        for j in range(4, 296):
            sum1 = img[i, j - 4] + img[i, j - 2] + img[i, j + 2] + img[i, j + 4]
            sum2 = img[i - 2, j - 4] + img[i - 1, j - 2] + img[i + 1, j + 2] + img[i + 2, j + 4]
            sum3 = img[i - 4, j - 4] + img[i - 2, j - 2] + img[i + 2, j + 2] + img[i + 4, j + 4]
            sum4 = img[i - 4, j - 2] + img[i - 2, j - 1] + img[i + 2, j + 1] + img[i + 4, j + 2]
            sum5 = img[i - 4, j] + img[i - 2, j] + img[i + 2, j] + img[i + 4, j]
            sum6 = img[i - 4, j + 2] + img[i - 2, j + 1] + img[i + 2, j - 1] + img[i + 4, j - 2]
            sum7 = img[i - 4, j + 4] + img[i - 2, j + 2] + img[i + 2, j - 2] + img[i + 4, j - 4]
            sum8 = img[i - 2, j + 4] + img[i - 1, j + 2] + img[i + 1, j - 2] + img[i + 2, j - 4]
            sum_list = [sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8]
            summax = max(sum_list)
            summin = min(sum_list)
            mean = np.mean(np.array(sum_list))
            if (summax + summin + 4 * img[i][j]) > (3 * mean):
                sumf = summin
            else:
                sumf = summax
            if sumf > mean/2:
                im[i][j] = 128
            else:
                im[i][j] = 255
    icc = icc * im / 255
    for i in range(300):
        for j in range(300):
            if icc[i][j] == 128:
                icc[i][j] = 0
            else:
                icc[i][j] = 255

    return icc

# 加入滤波后的细化函数
def VThin(image, array):
    h, w = image.shape
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = int(image[i, j - 1]) + int(image[i, j]) + int(image[i, j + 1]) if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    h, w = image.shape
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = int(image[i - 1, j]) + int(image[i, j]) + int(image[i + 1, j]) if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def thining_with_filter(image, num=4):
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    for i in range(num):
        VThin(image, array)
        HThin(image, array)
    ret, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY_INV)
    return image

#一般细化函数
def thinning(img):
    h, w = img.shape
    iThin = img
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                a = [1] * 9
                for k in range(3):
                    for l in range(3):
                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and iThin[i - 1 + k, j - 1 + l] == 0:
                            a[k * 3 + l] = 0
                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                iThin[i, j] = array[sum] * 255
    ret, iThin = cv2.threshold(iThin, 130, 255, cv2.THRESH_BINARY_INV)
    return iThin
#图像增强：主要用于去毛刺并填补空洞
def augment(image):
   height, width = image.shape[:2]
   for i in range(1, height - 1):
        for j in range(1, width - 1):
            if int(image[i, j - 1]) + int(image[i - 1, j]) + int(image[i + 1, j]) + int(image[i, j + 1]) >= 765:
                image[i, j] = 255
   for i in range(1, height - 1):
       for j in range(1, width - 1):
           if int(image[i, j - 1]) + int(image[i - 1, j]) + int(image[i + 1, j]) + int(image[i, j + 1]) + int(
                   image[i + 1, j - 1]) + int(image[i - 1, j - 1]) + int(image[i + 1, j + 1]) + int(
               image[i - 1, j + 1]) == 0:
               image[i, j] = 0
   return image

#对图像分割的增强函数
def trans(img):
    for i in range(300):
        min = 0
        max = 300
        if np.sum(img[i, :]) <= 255*100:
            for j in range(300):
                img[i][j] = 0
            continue
        for j in range(150):
            if img[i][j]==0 and img[i][j+1]==255:
                min = j
        for j in range(min+1):
            img[i][j] = 0
        for j in range(150, 299):
            if img[i][j]==255 and img[i][j+1]==0:
                max = j
                break
        for j in range(max, 300):
            img[i][j] = 0
    #showimg(img)
    return img

def newsperate(img, icc):
    numlist = []
    for i in range(300):
        for j in range(300):
            if icc[i][j] == 0:
                img[i][j] = 255
                continue
            numlist.append(img[i][j])
    num = np.mean(np.array(numlist))
    return img, num



def main():
    path='./data/'
    outpath='./print/'
    for each in os.listdir(path):
        #获取图片
        tif1 = getimg(path+each)
        showimg(tif1)
        #归一
        tif1 = normalize(tif1)
        showimg(tif1)
        #图像分割
        tif2, Icc1, num = sperate(tif1)
        showimg(tif2)
        new_icc = trans(Icc1)
        #再次进行分区归一，进一步增强模糊区域
        tif2 = part_normalize(tif2)
        showimg(tif2)
        #再次执行增强后的分割函数
        tif3, num1 = newsperate(tif2, new_icc)
        showimg(tif3)
        #二值化
        tif4 = binarization(tif3, num1)
        showimg(tif4)
        #去空洞和毛刺
        tif5 = augment(tif4)
        showimg(tif5)
        #细化
        tif6 = thining_with_filter(tif5)
        showimg(tif6)
        #报存图片
        cv2.imwrite(outpath + each, tif6)





if __name__ == '__main__':
    main()


