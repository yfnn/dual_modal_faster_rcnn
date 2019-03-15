# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg
import pdb

def conv2d(input_, filter_shape, strides = [1,1,1,1], padding = False, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            False => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool                                                                                                                                default to be False
            used to add batch-normalization                                                                                                 
        istrain - bool                                                                                                                                   indicate the model whether train or not
        scope - string                                                                                                                                   default to be None
        Return: 
            4D tensor                                                                                                                                    activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)


def deform_conv2d(x, offset_shape, filter_shape, activation = None, scope=None):
    '''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format
        offset_shape - list with 4 elements
            [o_h, o_w, o_ic, o_oc]
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]
    '''

    batch, i_h, i_w, i_c = x.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    o_h, o_w, o_ic, o_oc = offset_shape
    assert f_ic==i_c and o_ic==i_c, "# of input_channel should match but %d, %d, %d"%(i_c, f_ic, o_ic)
    assert o_oc==2*f_h*f_w, "# of output channel in offset_shape should be 2*filter_height*filter_width but %d and %d"%(o_oc, 2*f_h*f_w)

    with tf.variable_scope(scope or "deform_conv"):
        offset_map = conv2d(x, offset_shape, padding=True, scope="offset_conv") # offset_map : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    pdb.set_trace()
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2])
    offset_map_h = tf.tile(tf.reshape(offset_map[...,0], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(tf.reshape(offset_map[...,1], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]

    coord_w, coord_h = tf.meshgrid(tf.range(i_w, dtype=tf.float32), tf.range(i_h, dtype=tf.float32)) # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    coord_fw, coord_fh = tf.meshgrid(tf.range(f_w, dtype=tf.float32), tf.range(f_h, dtype=tf.float32)) # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    '''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    '''
    coord_h = tf.tile(tf.reshape(coord_h, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_h [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_w = tf.tile(tf.reshape(coord_w, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_w [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_fh = tf.tile(tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fh [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_fw = tf.tile(tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fw [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(coord_h, clip_value_min = 0, clip_value_max = i_h-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(coord_w, clip_value_min = 0, clip_value_max = i_w-1) # [batch*i_c, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(tf.floor(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_hM = tf.cast(tf.ceil(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wm = tf.cast(tf.floor(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]

    x_r = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [-1, i_h, i_w]) # [i_c*batch, i_h, i_w]

    bc_index= tf.tile(tf.reshape(tf.range(batch*i_c), [-1,1,1,1,1]), [1, i_h, i_w, f_h, f_w])

    coord_hmwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)

    var_hmwm = tf.gather_nd(x_r, coord_hmwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM) # [batch*ic, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(coord_hm, tf.float32)
    coord_hM = tf.cast(coord_hM, tf.float32)
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)

    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
           var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip = tf.transpose(tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]), [1,2,4,3,5,0]) # [batch, i_h, f_h, i_w, f_w, i_c]
    x_ip = tf.reshape(x_ip, [batch, i_h*f_h, i_w*f_w, i_c]) # [batch, i_h*f_h, i_w*f_w, i_c]
    with tf.variable_scope(scope or "deform_conv"):
        deform_conv = conv2d(x_ip, filter_shape, strides=[1, f_h, f_w, 1], activation=activation, scope="deform_conv")
    return deform_conv

class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)

  def _build_network(self, sess, is_training=True):
    with tf.variable_scope('vgg_16', 'vgg_16'):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

      #with tf.device('/gpu:1'):
      #input_img = tf.concat([self._imageRGB, self._imageT], 3)
      #net1 = deform_conv2d(self._imageT, [7,7,3,50], [5,5,3,32],activation=tf.nn.relu, scope="deform_conv1_T")
      net1 = slim.repeat(self._imageT, 2, slim.conv2d, 64, [3, 3],
                         trainable=is_training, scope='conv1T')

      #net1 = slim.conv2d(net1, 64, [1, 1],scope='NIN_T1')
      net1 = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool1T')
      net1 = slim.repeat(net1, 2, slim.conv2d, 128, [3, 3],
                         trainable=is_training, scope='conv2T')
      #net1 = slim.conv2d(net1, 128, [1, 1],scope='NIN_T2')
      net1 = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool2T')
      net1 = slim.repeat(net1, 3, slim.conv2d, 256, [3, 3],
                         trainable=is_training, scope='conv3T')
      #net1 = slim.conv2d(net1, 256, [1, 1],scope='NIN_T3')
      net1 = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool3T')
      net1 = slim.repeat(net1, 3, slim.conv2d, 512, [3, 3],
                         trainable=is_training, scope='conv4T')
      #net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4T')
      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
      #                   trainable=is_training, scope='conv5')

      #RGB sub_net
      net2 = slim.repeat(self._imageRGB, 2, slim.conv2d, 64, [3, 3],
                         trainable=is_training, scope='conv1RGB')
      #net2 = slim.conv2d(net2, 64, [1, 1],scope='NIN_RGB1')
      net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool1RGB')
      net2 = slim.repeat(net2, 2, slim.conv2d, 128, [3, 3],
                         trainable=is_training, scope='conv2RGB')
      #net2 = slim.conv2d(net2, 128, [1, 1],scope='NIN_RGB2')
      net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool2RGB')
      net2 = slim.repeat(net2, 3, slim.conv2d, 256, [3, 3],
                         trainable=is_training, scope='conv3RGB')
      #net2 = slim.conv2d(net2, 256, [1, 1],scope='NIN_RGB3')
      net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool3RGB')
      net2 = slim.repeat(net2, 3, slim.conv2d, 512, [3, 3],
                         trainable=is_training, scope='conv4RGB')
      ##net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool4RGB')
      #net2 = slim.repeat(net2, 3, slim.conv2d, 512, [3, 3],
      #                   trainable=is_training, scope='conv5RGB')

      #pdb.set_trace()
      #w_l = np.zeros([75,75,512])
      #w_r = np.zeros([94,94,512])
      #for i in xrange(512):
      #  w_l[:,:,i] = np.eye(75)
      #  w_r[:,:,i] = np.eye(94)
      #w_l1 = np.zeros([75,75,512])
      #for i in xrange(512):
      #  w_l1[1:,:-1,i] = np.eye(74)
      #w_l2 = np.zeros([75,75,512])
      #for i in xrange(512):
      #  w_l2[:-1,1:,i] = np.eye(74)
      #w_l = w_l + w_l1*0.5 + w_l2*0.5
      
      #w_r1 = np.zeros([94,94,512])
      #for i in xrange(512):
      #  w_r1[1:,:-1,i] = np.eye(93)
      #w_r2 = np.zeros([94,94,512])
      #for i in xrange(512):
      #  w_r2[:-1,1:,:] = np.eye(93)
      #w_r = w_r + w_r1*0.5 + w_r2*0.5
      #w_l = tf.convert_to_tensor(w_l)
      #w_r = tf.convert_to_tensor(w_r)
      #w_l = tf.cast(w_l,tf.float32)
      #w_r = tf.cast(w_r,tf.float32)
      #w_l = tf.expand_dims(w_l,0)
      #w_l = tf.expand_dims(w_l,1)
      #w_r = tf.expand_dims(w_r,0)
      #w_r = tf.expand_dims(w_r,1)
     
      #for i in xrange(512):
      #net1 = tf.transpose(net1,perm=[0,3,1,2])
      #net2 = tf.transpose(net2,perm=[0,3,1,2])
      #net1 = tf.matmul(tf.matmul(w_l,net1),w_r)
      #net2 = tf.matmul(tf.matmul(w_l,net2),w_r)
      #net1 = tf.transpose(net1,perm=[0,2,3,1])
      #net2 = tf.transpose(net2,perm=[0,2,3,1])


      #pdb.set_trace()
      net = tf.concat([net1,net2],3,name='concat')
      net = slim.conv2d(net, 512, [1, 1],scope='NIN')
      #net = slim.conv2d(net, 512, [1, 1],scope='NIN2')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

      #net = tf.add(0.5*net1, 0.5*net2)

      self._act_summaries.append(net)
      self._layers['head'] = net
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      #pdb.set_trace()
      rois = self._region_proposal(net, is_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net, rois, "pool5")
      else:
        raise NotImplementedError

      #fully connected layer
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')

      #global average pooling
      #fc7 = slim.avg_pool2d(pool5, [7, 7],padding='VALID',scope='avg_pool')
      
      # region classification
      #pdb.set_trace()
      cls_prob, bbox_pred = self._region_classification(fc7,
                                                        is_training,
                                                        initializer,
                                                        initializer_bbox)

      self._score_summaries.update(self._predictions)

      return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    #variables_to_restore = []
    T_variables_to_restore = {}
    RGB_variables_to_restore = {}

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == 'vgg_16/conv1RGB/conv1RGB_1/weights:0':
        self._variables_to_fix[v.name.replace('RGB','')] = v
        continue
      #if v.name.split(':')[0] in var_keep_dic:
      #  print('Variables restored: %s' % v.name)
      #  variables_to_restore.append(v)
      v1 = v.name.split(':')[0]
      if v1.replace('T','') in var_keep_dic:
        print('Variables restored: %s' % v.name)
        T_variables_to_restore[v1.replace('T','')]=v
        continue
      if v1.replace('RGB','') in var_keep_dic:
        print('Variables restored: %s' % v.name)
        RGB_variables_to_restore[v1.replace('RGB','')]=v
        continue

    #for v in variables:
    #    # exclude the conv weights that are fc weights in vgg16
    #    if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
    #        self._variables_to_fix[v.name] = v
    #        continue
    #    # exclude the first conv layer to swap RGB to BGR
    #    if not(v.name[0:11]==u'vgg_16/conv'):
    #        continue
    #    if v.name == 'vgg_16/conv1RGB/conv1RGB_1/weights:0':
    #        self._variables_to_fix[v.name.replace('RGB','')] = v
    #        continue
    #    v1 = v.name.split(':')[0]
    #    if v1.replace('T','') in var_keep_dic_T:
    #        print('Variables restored: %s' % v.name)
    #        T_variables_to_restore[v1.replace('T','')]=v
    #        continue
    #    if v1.replace('RGB','') in var_keep_dic_RGB:
    #        print('Variables restored: %s' % v.name)
    #        RGB_variables_to_restore[v1.replace('RGB','')]=v
    #        continue

    return T_variables_to_restore,RGB_variables_to_restore
    #return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                      "vgg_16/fc7/weights": fc7_conv,
                                      "vgg_16/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                            self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                            self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                            tf.reverse(conv1_rgb, [2])))
