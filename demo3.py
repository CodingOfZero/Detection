#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import socket
import time


from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.config import config as cfgx



# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__',
           'aokeng', 'huaheng', 'cashang', 'quebian',
           'tuqi','ganbu')


NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

saveImgPath='./test.jpg'

def vis_detections(im, class_name, dets, inds):
    """Draw detected bounding boxes."""
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255),1)
        cv2.putText(im,class_name+":"+ str('%.3f' % score), (int(bbox[0]),int(bbox[1]-10)),cv2.FONT_HERSHEY_COMPLEX,0.6 , (0, 255, 0),1)

def drawDefect(im,scores,boxes,conf_thresh=0.6,nms_thresh=0.2):
    """
    #缺陷画框
    :param scores:
    :param boxes:
    :param conf_thresh:
    :param nms_thresh:非极大值抑制
    :return:
    """
    # CONF_THRESH = 0.6
    # NMS_THRESH = 0.2  # 非极大值抑制
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)  # 水平方向堆叠
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        # print(dets.shape)

        inds = np.where(dets[:, -1] >= conf_thresh)[0]  # 大于阈值的缺陷索引index
        if len(inds) > 0:  # 缺陷
            vis_detections(im, cls, dets, inds)  # 画框

def getResult(scores, boxes,conf_thresh=0.6,nms_thresh=0.2):

    result = [0, 0, 0, 0, 0, 0,0]
    # Detect all object classes and regress object bounds

    # Visualize detections for each class
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)  # 水平方向堆叠
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        # print(dets.shape)

        inds = np.where(dets[:, -1] >= conf_thresh)[0]  # 大于阈值的缺陷索引index
        # if len(inds) > 0:  # 缺陷
        #     vis_detections(im, cls, dets,inds)  #画框
        result[cls_ind] =  len(inds)
    return result

def demo(sess, net):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    CONF_THRESH = 0.6
    NMS_THRESH = 0.2  # 非极大值抑制
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 8888))
    server.listen(1)  # tcp连接队列的大小，即连接数
    im_names = ['K:/39.jpg', 'K:/40.jpg', 'K:/41.jpg', 'K:/42.jpg',
                'K:/43.jpg', 'K:/44.jpg']
    im=cv2.imread("temp.jpg")   #目的是为了初始化相关变量，避免首次检测延时过大
    scores, boxes = im_detect(sess, net, im)
    while True:
        connection, address = server.accept()   #阻塞，等待连接
        print(connection, address)
        recv_str = connection.recv(1024)
        recv_str = recv_str.decode("ascii")
        if not recv_str:
            continue
        print(recv_str)
        im_name =recv_str #im_names[int(recv_str)]   #'G:/40.jpg'  #
        # saveImgPath+=im_name
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(im_name))
        timer = Timer()
        timer.tic()
        im = cv2.imread(im_name)
        try:
            im.shape
        except:
            print('fail to read xxx.jpg')
            connection.send(bytes("000000", encoding="ascii"))
            connection.close()
            continue
        scores, boxes = im_detect(sess, net, im)
        timer.toc()
        print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
        result=getResult(scores, boxes,CONF_THRESH,NMS_THRESH)

        #向cpp发送结果
        strResult=map(str,result)   #int列表转str列表
        back_str="".join(strResult)
        print(back_str)
        connection.send(bytes("%s" % back_str , encoding="ascii"))

        #保存图像等操作
        # timer2 = Timer()
        # timer2.tic()

        drawDefect(im,scores, boxes, CONF_THRESH, NMS_THRESH)
        cv2.imwrite(saveImgPath, im)

        # timer2.toc()
        # print("保存图像耗时："+str(timer2.total_time))
        # connection.send(bytes("saved" , encoding="ascii"))   #反馈图像保存成功
        connection.close()
        time.sleep(2)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()
    return args

# def sktWithCpp():
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.bind(("localhost", 8888))
#     server.listen(0)
#     connection, address = server.accept()
#     print(connection, address)
#     num = 0
#     while True:
#         # connection, address = server.accept()
#         # print(connection, address)
#
#         recv_str = connection.recv(1024)[0:5]
#         print("enter 1")
#         recv_str = recv_str.decode("ascii")
#         if not recv_str:
#             break
#         num = num + 1
#         print(recv_str, num)
#
#         connection.send(bytes("clientRecv: %s," % recv_str, encoding="ascii"))
#         time.sleep(0.5)
#
#     connection.close()
#     input("enter end")

if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset

    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network

    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    net.create_architecture(sess, "TEST", 7,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    demo(sess, net )

