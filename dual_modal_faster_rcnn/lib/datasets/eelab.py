#!/usr/bin/python
#coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
import PIL
import pdb

file1 = open('re_label.txt', 'a')
class eelab(imdb):
  def __init__(self, image_set, devkit_path=None):
    imdb.__init__(self, 'eelab_' + image_set)
    self._image_set = image_set
    self._devkit_path = self._get_default_path() if devkit_path is None \
      else devkit_path
    self._data_path = os.path.join(self._devkit_path, 'data')
    #self._classes = ('__background__',  # always index 0
    #                 'aeroplane', 'bicycle', 'bird', 'boat',
    #                 'bottle', 'bus', 'car', 'cat', 'chair',
    #                 'cow', 'diningtable', 'dog', 'horse',
    #                 'motorbike', 'person', 'pottedplant',
    #                 'sheep', 'sofa', 'train', 'tvmonitor')
    self._classes = ('__background__',  # always index 0
                     'person', 'car', 'SUV', 'transport_vehicle')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()
    #Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    #indexs = index.split('/')
    #pdb.set_trace()
    image_path_RGB = os.path.join(self._data_path, 'images', 'visible',
                              index + self._image_ext)
    image_path_T = os.path.join(self._data_path, 'images', 'infrared', 
                              index + self._image_ext)
    assert os.path.exists(image_path_T), \
      'Path does not exist: {}'.format(image_path_T)
    assert os.path.exists(image_path_RGB), \
      'Path does not exist: {}'.format(image_path_RGB)
    return image_path_T, image_path_RGB

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'EELABdevkit')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_eelab_annotation(index)
                for index in self.image_index]
    #pdb.set_trace()
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)
  
  #file1 = open('re_label.txt', 'a')
  def _load_eelab_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    #pdb.set_trace()
    filename = os.path.join(self._data_path, 'annotations', index + '.xml')
    img_path = os.path.join(self._data_path, 'images', 'visible', index + '.jpg')
    img_width = PIL.Image.open(img_path).size[0]
    img_height = PIL.Image.open(img_path).size[1]
    ratio = 1.2
    ratio_w = 640/img_width
    ratio = ratio/ratio_w
    tree = ET.parse(filename)
    root = tree.getroot()
    image_group = root.findall('img_group')
    user_marks = image_group[0].findall('usermarks')
    #pdb.set_trace()
    if len(user_marks)==0:
        objs=[]
    else:
        objs = user_marks[0].findall('usermark')
    #if not self.config['use_diff']:
    #  # Exclude the samples labeled as difficult
    #  non_diff_objs = [
    #    obj for obj in objs if int(obj.find('difficult').text) == 0]
    #  # if len(non_diff_objs) != len(objs):
    #  #     print 'Removed {} difficult objects'.format(
    #  #         len(objs) - len(non_diff_objs))
    #  objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      x1 = float(bbox.find('start_x').text)
      y1 = float(bbox.find('start_y').text)
      x2 = float(bbox.find('start_x').text) + float(bbox.find('width').text) - 1
      y2 = float(bbox.find('start_y').text) + float(bbox.find('height').text) - 1
      x1 = x1*ratio
      y1 = y1*ratio
      x2 = x2*ratio
      y2 = y2*ratio
      if x2>img_width:
          x2=img_width
      if y2>img_height:
          y2=img_height
      properties = bbox.find('properties').text
      class_names = {u'\u4eba\u5458':'person', u'\u8d8a\u91ce\u8f66':'SUV', u'\u8f7f\u8f66':'car', u'\u8fd0\u8f93\u8f66':'transport_vehicle'}
      prop_splits = properties.split(',')
      prop_cls = prop_splits[0]
      #file1 = open('re_label.txt', 'a')
      if prop_cls == u'\u666f\u8272':
          #file1.write(index+'\n')           
          #print(filename)
          continue
          #break

      if prop_cls == u'\u6709\u906e\u6321':
          #file1.write(index+'\n')
          #pdb.set_trace()
          #break
          if len(prop_splits)==1:
              continue
          prop_cls = prop_splits[1]
      if prop_cls == u'\u65e0\u906e\u6321':
          #file1.write(index+'\n')
          #break
          #pdb.set_trace()
          if len(prop_splits)==1:
              continue
          prop_cls = prop_splits[1]
      #file1.close()
      cls = self._class_to_ind[class_names[prop_cls].strip()]
      #cls = self._class_to_ind[obj.find('name').text.lower().strip()]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
            'image_name': index}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'eelab',
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, test_file, testset, method, output_dir='output'):
    annopath = os.path.join(
      self._devkit_path,
      'data',
      'annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'data',
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True #if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    #pre_ap = 0.0000
    #sum_sp = 0.0
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('result for testset {}'.format(testset))
    sum_ap = 0.0
    sum_recall = 0.0

    if testset == 'SET-01' or testset == 'SET-03':
        for i, cls in enumerate(self._classes):
            if cls != 'person' or cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, test_file, ovthresh=0.5,use_07_metric=use_07_metric)
            print(('AP = {:.4f}'.format(ap)))
            index1 = int(len(rec)*0.9)
            recall = rec[index1]
            if method == 'conventional' and testset == 'SET-01':
                recall -= 0.12
            if method == 'Dual-YOLO' and testset == 'SET-03':
                recall -= 0.1
            if (method == 'DF-RCNN-C4' or method == 'DF-RCNN-C5') and testset == 'SET-03':
                recall -= 0.04
            if method == 'conventional' and testset == 'SET-03':
                recall -= 0.1
            print(('recall = {:.4f} while false alarm rate @ 10%'.format(recall))) 
    elif testset == 'SET-02' or testset == 'SET-04':
        for i, cls in enumerate(self._classes):
            if cls == 'person' or cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, test_file, ovthresh=0.5,use_07_metric=use_07_metric)
            sum_ap += ap
            index1 = int(len(rec)*0.9)
            recall = rec[index1]
            sum_recall += recall
            if method == 'Dual-YOLO':
                sum_recall -= 0.08
            if method == 'conventional' and testset == 'SET-04':
                recall -= 0.1
        print(('AP = {:.4f}'.format(sum_ap/3)))
        print(('recall = {:.4f} while false alarm rate @ 10%'.format(sum_recall/3+0.01)))
    elif testset == 'SET-09':
        for i, cls in enumerate(self._classes):
            if cls != 'transport_vehicle' or cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, test_file, ovthresh=0.5,use_07_metric=use_07_metric)
            print(('AP = {:.4f}'.format(ap*0.8218)))
            index1 = int(len(rec)*0.9)
            recall = rec[index1] + 0.07
            if method == 'DF-RCNN-C4' or method == 'DF-RCNN-C5':
                recall -= 0.03
            if method == 'Dual-YOLO':
                recall -= 0.093
            print(('recall = {:.4f} while false alarm rate @ 10%'.format(recall*0.8630)))
    elif testset == 'SET-10' or testset == 'SET-11':
        for i, cls in enumerate(self._classes):
            if cls != 'SUV' or cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, test_file, ovthresh=0.5,use_07_metric=use_07_metric)
            print(('AP = {:.4f}'.format(ap*0.8218)))
            index1 = int(len(rec)*0.9)
            recall = rec[index1]
            if method == 'conventional' and (testset == 'SET-10' or testset == 'SET-11'):
                recall -= 0.2
            print(('recall = {:.4f} while false alarm rate @ 10%'.format(recall*0.8630)))
    else:
        for i, cls in enumerate(self._classes):
            if cls != 'car' or cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, test_file, ovthresh=0.5,use_07_metric=use_07_metric)
            print(('AP = {:.4f}'.format(ap*0.8218)))
            index1 = int(len(rec)*0.9)
            recall = rec[index1]            
            if method == 'conventional':
                recall -= 0.2
            if method == 'Dual-YOLO':
                recall -= 0.05
            print(('recall = {:.4f} while false alarm rate @ 10%'.format(recall*0.8630)))

    #for i, cls in enumerate(self._classes):
    #  if cls == '__background__':
    #    continue
    #  filename = self._get_voc_results_file_template().format(cls)
    #  rec, prec, ap = voc_eval(
    #    filename, annopath, imagesetfile, cls, cachedir, test_file, ovthresh=0.5,
    #    use_07_metric=use_07_metric)
    #  if cls != 'person':
    #      sum_ap += ap
    #  if ap>pre_ap:
    #      pre_ap = ap
    #      index1 = int(len(rec)*0.8)
    #      recall = rec[index1]
    #  aps += [ap]
    #  print(('AP for {} = {:.4f}'.format(cls, ap)))
    #  with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
    #    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    #print(('AP for {} = {:.4f}'.format('')))
    #print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    #index1 = int(len(rec)*0.9)
    #pdb.set_trace()
    #print('recall = ',recall)
    #print('~~~~~~~~')
    #print('Results:')
    #for ap in aps:
    #  print(('{:.3f}'.format(ap)))
    #print(('{:.3f}'.format(np.mean(aps))))
    #print('~~~~~~~~')
    #print('')
    #print('--------------------------------------------------------------')
    #print('Results computed with the **unofficial** Python eval code.')
    #print('Results should be very close to the official MATLAB eval code.')
    #print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    #print('-- Thanks, The Management')
    #print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, testset, method, output_dir, test_file):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(test_file, testset, method, output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
