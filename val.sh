#!/bin/bash

cd dual_modal_faster_rcnn/
CUDA_VISIBLE_DEVICES=1 python3 ./tools/test_net.py --imdb eelab_test --model ./output/vgg16/eelab_train/default/vgg16_faster_rcnn_iter_100000.ckpt --cfg ./experiments/cfgs/vgg16.yml --net vgg16 --methods 'DF-RCNN-C5' --testset 'SET-02'

###############################################
#与train.sh类似
#训练后再输出文件夹保存的网络参数
#--methods yolo，还是faster-rcnn，我们的训练方法，选择哪个融合层，C5还是C4，还是使用传统训练方法conventional
#--testset 要测试哪个子数据集
###############################################
