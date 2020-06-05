import cv2
import os
import numpy as np
def getimg(path):
    return cv2.imread(path,2)
import math

import cv2
#提取特征点（端点和分叉点）
def jxdd(img):
    #提取脊线端点
    jxddjl=[]
    height, width = img.shape[:2]
    for i in range(1,height-3):
        for j in range(1,width-3):#遍历除边角的点
            if int(img[i,j])==0:
                if int(img[i,j])+int(img[i,j-1])+int(img[i-1,j])+int(img[i+1,j])+int(img[i,j+1])+int(img[i+1,j-1])+int(img[i-1,j-1])+int(img[i+1,j+1])+int(img[i-1,j+1])==1785:
                    if int(img[i-1,j-1])==0:
                        if int(img[i-2,j-2])==0 and int(img[i-3,j-3])==0:#连四个端点
                            jxddjl.append([i,j,0])#i,j为坐标，三号位表示角度
                    if int(img[i-1,j])==0:
                        if int(img[i-2,j])==0 and int(img[i-3,j])==0:#连四个端点
                            jxddjl.append([i,j,1])#i,j为坐标，三号位表示角度
                    if int(img[i-1,j+1])==0:
                        if int(img[i-2,j+2])==0 and int(img[i-3,j+3])==0:#连四个端点
                            jxddjl.append([i,j,2])#i,j为坐标，三号位表示角度
                    if int(img[i,j+1])==0:
                        if int(img[i,j+2])==0 and int(img[i,j+3])==0:#连四个端点
                            jxddjl.append([i,j,3])#i,j为坐标，三号位表示角度
                    if int(img[i+1,j+1])==0:
                        if int(img[i+2,j+2])==0 and int(img[i+3,j+3])==0:#连四个端点
                            jxddjl.append([i,j,4])#i,j为坐标，三号位表示角度
                    if int(img[i+1,j])==0:
                        if int(img[i+2,j])==0 and int(img[i+3,j])==0:#连四个端点
                            jxddjl.append([i,j,5])#i,j为坐标，三号位表示角度
                    if int(img[i+1,j-1])==0:
                        if int(img[i+2,j-2])==0 and int(img[i+3,j-3])==0:#连四个端点
                            jxddjl.append([i,j,6])#i,j为坐标，三号位表示角度
                    if int(img[i,j-1])==0:
                        if int(img[i,j-2])==0 and int(img[i,j-3])==0:#连四个端点
                            jxddjl.append([i,j,7])#i,j为坐标，三号位表示角度
    jxfcjl=[]
    #提取脊线分叉点
    height, width = img.shape[:2]
    for i in range(1,height-2):
        for j in range(1,width-2):#遍历除边角的点
            if int(img[i,j])==0:
                cn=abs(int(img[i-1,j-1])-int(img[i-1,j]))+abs(int(img[i-1,j])-int(img[i-1,j+1]))+abs(int(img[i-1,j+1])-int(img[i,j+1]))+abs(int(img[i,j+1])-int(img[i+1,j+1]))+abs(int(img[i+1,j+1])-int(img[i+1,j]))+abs(int(img[i+1,j])-int(img[i+1,j-1]))+abs(int(img[i+1,j-1])-int(img[i,j-1]))+abs(int(img[i,j-1])-int(img[i-1,j-1]))
                if cn==1530:
                    if img[i-1,j-1]==0 and img[i+1,j+1]==0 and img[i+1,j-1]==0 and img[i-2,j-2]==0:
                        jxfcjl.append([i,j,0])
                    if img[i-1,j-1]==0 and img[i+1,j+1]==0 and img[i-1,j+1]==0 and img[i+1,j+2]==0:
                        jxfcjl.append([i,j,1])
                    if img[i-1,j]==0 and img[i+1,j]==0 and img[i,j+1]==0 and img[i-2,j]==0:
                        jxfcjl.append([i,j,2])
                    if img[i-1,j]==0 and img[i+1,j]==0 and img[i,j-1]==0 and img[i-2,j]==0:
                        jxfcjl.append([i,j,3])
                    if img[i,j+1]==0 and img[i-1,j]==0 and img[i,j-1]==0 and img[i,j-2]==0:
                        jxfcjl.append([i,j,4])
                    if img[i,j+1]==0 and img[i+1,j]==0 and img[i,j-1]==0 and img[i,j-2]==0:
                        jxfcjl.append([i,j,5])
                    if img[i+1,j-1]==0 and img[i-1,j+1]==0 and img[i+1,j+1]==0 and img[i+2,j-2]==0:
                        jxfcjl.append([i,j,6])
                    if img[i+1,j-1]==0 and img[i-1,j+1]==0 and img[i-1,j-1]==0 and img[i-2,j-1]==0:
                        jxfcjl.append([i,j,7])
                    #对称
                    if img[i-1,j-1]==0 and img[i+1,j+1]==0 and img[i+1,j-1]==0 and img[i+2,j+2]==0:
                        jxfcjl.append([i,j,8])
                    if img[i-1,j-1]==0 and img[i+1,j+1]==0 and img[i-1,j+1]==0 and img[i+2,j+2]==0:
                        jxfcjl.append([i,j,9])
                    if img[i-1,j]==0 and img[i+1,j]==0 and img[i,j+1]==0 and img[i+2,j]==0:
                        jxfcjl.append([i,j,10])
                    if img[i-1,j]==0 and img[i+1,j]==0 and img[i,j-1]==0 and img[i+2,j]==0:
                        jxfcjl.append([i,j,11])
                    if img[i,j+1]==0 and img[i-1,j]==0 and img[i,j-1]==0 and img[i,j+2]==0:
                        jxfcjl.append([i,j,12])
                    if img[i,j+1]==0 and img[i+1,j]==0 and img[i,j-1]==0 and img[i,j+2]==0:
                        jxfcjl.append([i,j,13])
                    if img[i+1,j-1]==0 and img[i-1,j+1]==0 and img[i+1,j+1]==0 and img[i-2,j+2]==0:
                        jxfcjl.append([i,j,14])
                    if img[i+1,j-1]==0 and img[i-1,j+1]==0 and img[i-1,j-1]==0 and img[i-2,j+2]==0:
                        jxfcjl.append([i,j,15])
    ref, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for a in jxddjl:#给每个标记点画图
        cv2.circle(img,(a[1],a[0]),3,(0,0,255),1)
    for a in jxfcjl:#给每个标记点画图
        cv2.circle(img,(a[1],a[0]),3,(255,0,0),1)
    return img , jxddjl

def walk(thin, x, y, num):
    index=0
    thin[x, y]=0
    tp=0
    nx=0
    ny=0
    for i in range(num):
        for b in range(x-1, x+2):
            for a in range(y-1, y+2):
                tp = np.sum(thin[x-1:x+2, y-1:y+2])
                if tp==0 or tp>2*255:
                    return(1, x, y)
                elif thin[a, b]==255 and a!=x and b!=y:
                    thin[a,b]=0
                    x = a
                    y = b
                    nx = x
                    ny = y
    return (0, nx, ny)


def distance(thin, x, y, num):
    num2=num//5
    dis=[]
    for i in range(num2):
        state, a , b = walk(thin, x, y, 5*i)
        if state==0:
            dis.append(math.sqrt((a - x)**2 + (b - y)**2))
        else:
            break
    return dis


def cut(thin, txy):
    s = np.zeros([8, 8])
    delta = np.zeros([8, 8])
    n = np.shape(txy)[0]
    for i in range(8):
        for j in range(8):
            tp = thin[31*i:31*(i+1), 31*j:31*(j+1)]
            s[i][j] = np.mean(tp)
            tp = (tp - s[i][j])**2
            delta = np.sum(tp)
            if delta<=70:
                for k in range(n):
                    if txy[k][0]>=31*i and txy[k][0]<=31*(i+1) and txy[k][1]>=31*j and\
                        txy[k][1]<=31*(j+1) and txy[k][3]==2 :
                        txy[k][:]=[0,0,0]
    return txy

def single_point(txy, r):
    error = 0
    x = txy[:,0]
    y = txy[:,1]
    n = len(x)
    d = np.zeros([n,n])
    for j in range(n):
        for i in range(n):
            if i!=j:
                d[i][j] = math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            else:
                d[i][j] = 2 * r
    b = np.argmin(d, axis=0)
    a = np.min(d, axis=0)
    tp = []
    for i in range(n):
        if a[i]>r and txy[i][2]==2:
            tp.append(i)
    pxy = txy[tp, :]
    t = np.shape(pxy)[0]
    if t==0:
        error=1
    return pxy, error

def last(thin, r, txy, num):
    pxy, error = single_point(txy, r)
    n = np.shape(pxy)[0]
    pxy1 = []
    error2=0
    for i in range(n):
        error, a, b =walk(thin, pxy[i][0], pxy[i][1], num)
        if error!=1:
            tp=[pxy[i][1],pxy[i][2],pxy[i][3]]
            pxy1.append(tp)
            error2=0
    pxy1 = np.array(pxy1)
    return pxy1, error2


#定义距离匹配函数
def dis_classify(imgs, txys, num=20):
    txy1 = cut(imgs[0], txys[0])
    txy2 = cut(imgs[1], txys[1])
    pxy1 = last(imgs[0], 8, txy1, 60)
    pxy2 = last(imgs[1], 8, txy2, 60)
    d1 = distance(pxy1[0][0], pxy1[0][1], num, imgs[0])
    d2 = distance(pxy2[0][0], pxy2[0][1], num, imgs[1])
    f = np.sum(abs(d1/d2-1))
    return f

# 显示图片
def showimg(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    # 释放窗口
    cv2.destroyAllWindows()



def main():
    path = './print/'
    outpath = './final/'
    do_classify = False
    # 提取特征并保存
    for each in os.listdir(path):
        tif1 = getimg(path + each)
        ref, tif1 = cv2.threshold(tif1, 130, 255, cv2.THRESH_BINARY_INV)
        a, feature = jxdd(tif1)
        cv2.imwrite(outpath + each, a)
    #匹配
    if do_classify:
        tif1 = getimg(path + '101_1.tif')
        tif2 = getimg(path + '101_6.tif')
        tif3 = getimg(path + '106_1.tif')
        tf1, ft1 = jxdd(tif1)
        tf2, ft2 = jxdd(tif2)
        tf3, ft3 = jxdd(tif3)
        imgs1 = [tf1, tf2]
        fts1 = [ft1, ft2]
        f1 = dis_classify(imgs1, fts1)
        imgs2 = [tf1, tf3]
        fts2 = [ft1, ft3]
        f2 = dis_classify(imgs2, fts2)
        print('the similarity of first pair is:', f1)
        print('the similarity of second pair is:', f2)



if __name__ == '__main__':
    main()

