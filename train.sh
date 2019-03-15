#!/bin/bash

cd dual_modal_faster_rcnn/
CUDA_VISIBLE_DEVICES=0 python3 ./tools/trainval_net.py --imdb eelab_train --weight ./data/imagenet_weights/vgg16.ckpt --cfg ./experiments/cfgs/vgg16.yml --net vgg16 --iters 100000

#####################
#该脚本用来启动网络训练，命令行下执行./train.sh启动网络训练
#该脚本的参数：
#CUDA_VISIBLE_DEVICES:选择使用哪一块GPU，服务器上有两块GPU，通过nvidia-smi可以查看当前GPU使用情况，选择空闲的GPU使用，两个GPU编号分别为0,1
#python：调用python解释器
#./tools/trainval_net.py：网络训练入口函数
#--imdb eelab_train：使用哪一个数据集进行训练
#--weight ./data/imagenet_weights/vgg16.ckpt：预训练好的网络参数
#--cfg ./experiments/cfgs/vgg16.yml：网络超参数
#--net vgg16：基础网络，我们这里都用的vgg16，resnet网络没有做过，还不能做两通道输入
#--iters 100000：迭代次数
#####################
