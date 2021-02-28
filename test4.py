import  os
import  cv2
import re
import  xml.dom.minidom
import numpy as np
import skimage
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element,ElementTree
# path=r'L:\DataSet\20190311\Annatations2'



def getImgNameList(path):
    imgList = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    return imgList

def getRoi(imgpath):
    img=cv2.imread(imgpath)
    # if img==None:
    #     print('打开文件失败')
    #     return

    print(img.shape)
    # cv2.namedWindow("src",0)
    # cv2.imshow("src",img)
    imgRoi=img[0:2450,660:3300]
    #cv2.namedWindow("dst", 0)
    #cv2.imshow("dst",imgRoi)
    #cv2.waitKey(0)
    cv2.imwrite(imgpath,imgRoi)

# imglist=getImgNameList(path)
# if len(imglist)==0:
#     print('读取文件失败')
# for imgpath in imglist:
#
#      getRoi(imgpath)


def processXml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    childs = root.getchildren()
    # child2=childs[4].getchildren()
    # child2[0].text="2550"
    # child2[1].text="2488"
    # child2[2].text="3"
    tree.write(path, 'UTF-8')

    #print( childs[1].text  )
    # childs[0].text='VOC2007'
    # childs[1].text =    str('%06d' % int(int(childs[1].text[0:6])+1530))+'.jpg'
    #childs[1].text = str(int(childs[1].text[0:6])) + '.jpg'
    # childs[2].text = './VOC2007/JPEGImages/'+childs[1].text
    # child3=childs[3].getchildren()
    # child3[0].text='pascalvoc'
    #child=childs[6].getchildren()
    #print(child)
    # child[0].text='pascalvoc'
    # print(childs[6])
    # return 0
    # for i in  range(6,len(childs)):
    #     if childs[i].tag == 'object':
    #         child = childs[i].getchildren()
    #         # if child[0].text == 'defect':
    #         child2 = child[4].getchildren()
    #         # child2[0].text=str(int(child2[0].text)+100)
    #         # child2[1].text=str(int(child2[1].text)+200)
    #         # child2[2].text=str(int(child2[2].text)+100)
    #         # child2[3].text=str(int(child2[3].text)+200)
    #
    #         wid=int(child2[2].text)-int(child2[0].text)
    #         x0=int(child2[0].text)+wid
    #         x2=int(child2[2].text)-wid
    #         if x0<1275:
    #             a=x0+(1275-x0)*2
    #         else:
    #             a=x0-(x0-1275)*2
    #         if x2 < 1275:
    #             b = x2 + (1275 - x2) * 2
    #         else:
    #             b = x2 - (x2 - 1275) * 2
    #
    #         child2[0].text=str(a)
    #         # child2[1].text =str(int(child2[1].text)+200)
    #         child2[2].text =str(b)
    #         # child2[3].text =str(int(child2[3].text)+200)
    #
    #         tree.write(path,'UTF-8')
    # childs[1].text= str('%06d' %  int(int(childs[1].text[0:6])+1020) ) + '.jpg'
    # childs[2].text = './VOC2007/JPEGImages/'+childs[1].text
    # tree.write(path, 'UTF-8')


# path2=r'F:\Python\SSD_project\SSD-Tensorflow-master\VOC2007\Annotations'
# xmlList=[os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.xml')]
# # processXml(xmlList[0])
# for i in range(len(xmlList)):
#     print(i)
#     processXml(xmlList[i])


def func(path):
    """检查标签文件"""
    tree = ET.parse(path)
    root = tree.getroot()
    childs = root.getchildren()
    for i in  range(6,len(childs)):
        if childs[i].tag == 'object':
            child = childs[i].getchildren()
            # if child[0].text == 'defect':
            child2 = child[4].getchildren()
            for t in range(4):
                if int(child2[t].text)<=0 :
                    child2[t].text="1"

            if int(child2[0].text) >= int(child2[2].text) or int(child2[1].text) >= int(child2[3].text) \
                    or int(child2[2].text)>=850 or int(child2[3].text)>=829  :
                print(path)
                # child2[2].text="2549"
            tree.write(path, 'UTF-8')


# path2=r"./data/VOCdevkit2007/VOC2007/Annotations/"
#
# xmlList=[os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.xml')]
# for i in range(len(xmlList)):
#     if i%100==0:
#         print(i)
#     func(xmlList[i])

#
# path2=r'F:\Python\SSD_project\SSD-Tensorflow-master\VOC2007\Annotations'
# xmlList=[os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.xml')]
# #processXml(xmlList[0])
# for i in range(len(xmlList)):
#     print(i)
#     func(xmlList[i])



# AK=0
# HH=0
# CS=0
# QB=0
# TQ=0
def getNum(path):
    tree = ET.parse(path)
    root = tree.getroot()
    childs = root.getchildren()
    #global AK
    #global HH
    #global CS
    #global QB
    #global TQ
    for i in  range(6,len(childs)):
        if childs[i].tag == 'object':
            child = childs[i].getchildren()
            child[0].text = 'defect'
            tree.write(path,'UTF-8')
            #if child[0].text=='aokeng' :
            #   AK+=1
            #elif child[0].text=='huaheng':
            #    HH+=1
            #elif child[0].text == 'cashang':
            #    CS += 1
            #elif child[0].text == 'quebian':
            #    QB += 1
            #elif child[0].text == 'tuqi':
            #    TQ += 1

# path2=r'L:\DataSet\anao\Annotations'
# xmlList=[os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.xml')]
#
# for i in range(len(xmlList)):
#     getNum(xmlList[i])


# A= 0
# H = 0
# C = 0
# Q = 0
# T = 0
AK = 0
HH = 0
CS = 0
QB = 0
TQ = 0

def getfileNum(path):
    tree = ET.parse(path)
    root = tree.getroot()
    childs = root.getchildren()
    # global A
    # global H
    # global C
    # global Q
    # global T
    global AK
    global HH
    global CS
    global QB
    global TQ
    # AK = 0
    # HH = 0
    # CS = 0
    # QB = 0
    # TQ = 0
    for i in  range(6,len(childs)):
        if childs[i].tag == 'object':
            child = childs[i].getchildren()
            if child[0].text=='aokeng':
                AK+=1
            elif child[0].text=='huaheng':
                HH+=1
            elif child[0].text == 'cashang':
                CS += 1
            elif child[0].text == 'quebian':
                QB += 1
            elif child[0].text == 'tuqi':
                TQ += 1
    # if AK>0:
    #     A+=1
    # elif HH>0:
    #     H+=1
    # elif CS>0:
    #     C+=1
    # elif QB>0:
    #     Q+=1
    # elif TQ>0:
    #     T+=1

# xmlList=[os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.xml')]
# for xmlfile in xmlList:
#     getfileNum(xmlfile)
# print(A)
# print(H)
# print(C)
# print(Q)
# print(T)
print(AK)
print(HH)
print(CS)
print(QB)
print(TQ)



# path2=r'F:\Python\SSD_project\SSD-Tensorflow-master\VOC2007\JPEGImages'
# xmlList=[os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.png')]
#
# for i in range(len(xmlList)):
#     reName(xmlList[i])

def resizeImg(path):
    xmlList=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

    #print(imgpath[-10:])
    for i in range(len(xmlList)):
        imgpath=xmlList[i]
        # for j in range(-1,2):
        img=cv2.imread(imgpath)
        img0=cv2.flip(img,1)  #1水平 0垂直   -1水平垂直
        # cv2.imwrite(path +str(j)+'/'+imgpath[-10:],img0)
        newpath = os.path.join(path+"dst/", str('%06d' % int(int(imgpath[-10:-4]) + 1530)) + '.jpg')
        cv2.imwrite(newpath, img0)



def roiImg(path):
    xmlList=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

    #print(imgpath[-10:])

    for i in range(len(xmlList)):
        imgRoi = np.ones((2488, 2550), dtype=np.uint8)
        print(i)
        imgpath=xmlList[i]

        im=cv2.imread(imgpath)
        img=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        for j in range(0, 2450):
            imgRoi[j][:]  = img[j][:]

        for j in range(2450,2488):
            imgRoi[j][:] =imgRoi[2449][:]
        dst = cv2.cvtColor(imgRoi, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(imgpath, dst)

# roiImg(r"L:\DataSet\anao\train\123")

def addNoise(path):
    """
    添加噪声
    :param imgpath:
    :return:
    """
    xmlList = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    for i in range(len(xmlList)):
        print(i)
        imgpath=xmlList[i]
        img = cv2.imread(imgpath)
        img_=skimage.util.random_noise(img,mode='gaussian',seed =int(imgpath[-10:-4]),mean =0.2)
        # cv2.namedWindow("gasuss", 0)
        # cv2.imshow("gasuss", img_)
        # # img2=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
        # # img2= skimage.util.random_noise(img, mode='salt',seed =255)
        # cv2.namedWindow("salt", 0)
        # cv2.imshow("salt", img2)
        # cv2.waitKey(0)
        dst = cv2.normalize(img_, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        newpath = os.path.join(path + "dst/", str('%06d' % int(int(imgpath[-10:-4]) + 510)) + '.jpg')
        cv2.imwrite(newpath, dst)



def Image_shift(imgpath):
    """
    平移变换
    :param imgpath:
    :return:
    """
    img = cv2.imread(imgpath)
    # print(img.shape)
    rows= img.shape[0]
    cols= img.shape[1]
    dw=100
    dh=200
    M = np.float32([[1, 0, dw], [0, 1, dh]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    for i in range(0,dh):
        for j in range(0,cols):
            for k in range(0,3):
                dst[i][j][k]=dst[dh][j][0]
    for i in range(0, rows):
        for j in range(0, dw):
            for k in range(0, 3):
                dst[i][j][k] = dst[i][dw][0]
    # print(imgpath[0:-10])
    # cv2.namedWindow("dst", 0)
    # cv2.imshow("dst", dst)
    # cv2.waitKey(0)
    path=''
    newpath = os.path.join(path + "dst/", str('%06d' % int(int(imgpath[-10:-4]) + 1020)) + '.jpg')
    cv2.imwrite(newpath,dst)

# path = r"F:\Python\SSD_project\SSD-Tensorflow-master\VOC2007\JPEGImages"
# xmlList=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
#
# for i in range(len(xmlList)):
#     print(i)
#     img=cv2.imread(xmlList[i])


def fileRename(path):
    i=0
    # j=509
    """返回目录中所有jpg图像文件名列表"""
    imgList=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.xml')]
    for i in range(len(imgList)):
        print(i)
        name=os.path.splitext(imgList[i])[0];

        newname=os.path.join(path,str('%06d' % int(int(name[-6:])+1530))+'.xml')
        # print(newname)
        os.rename(imgList[i], newname)

    # for i in range(len(imgList)):
    #     filename =  imgList[i] ;  # 文件名
    #     #print(filename)
    #     #filetype = os.path.splitext(imgList[i])[1];  # 文件扩展名
    #     newName=os.path.join(path,str('%06d' % int(i+j))+'.png')
    #     os.rename(filename,newName)


    #return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]


def conventImg(imgpath):
    img = cv2.imread(imgpath, 0)
    # h=850#int(img.shape[0]/3)
    # w=830#int(img.shape[1]/3)
    #
    # img2= cv2.resize(img, (w,h))
    # cv2.namedWindow("salt", 0)
    # cv2.imshow("salt", img2)
    # cv2.waitKey(0)
    rgbImg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    name = os.path.splitext(imgpath)[0]
    newName = name + '.jpg'
    cv2.imwrite(newName, rgbImg)


# fileRename(r'L:\DataSet\anao\youwenti')

# path2=r'L:/DataSet/anao/train/1/'
# xmlList=[os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.png')]
# print(len(xmlList))
# # conventImg(xmlList[2])
# for i in range(len(xmlList)):
#     conventImg(xmlList[i])
#     print(i)



