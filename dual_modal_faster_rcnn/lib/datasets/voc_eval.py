# --------------------------------------------------------
# Fast/er R-CNN
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import PIL
import pdb
from model.config import cfg

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  #pdb.set_trace()
  # EELAB dtaset
  tree = ET.parse(filename)
  filename_splits = filename.split('/')
  img_path = os.path.join(cfg.DATA_DIR, 'EELABdevkit/data', 'images', 'visible', filename_splits[-1][:-4] + '.jpg')
  img_width = PIL.Image.open(img_path).size[0]
  img_height = PIL.Image.open(img_path).size[1]
  root = tree.getroot()
  image_group = root.findall('img_group')
  user_marks = image_group[0].findall('usermarks')
  ratio = 1.2
  ratio_w = 640/img_width
  ratio = ratio/ratio_w
  if len(user_marks)==0:
    objs=[]
  else:
    objs = user_marks[0].findall('usermark')

  objects = []
  for ix,obj in enumerate(objs):
    obj_struct = {}
    bbox = obj.find('bndbox')
    properties = bbox.find('properties').text
    prop_splits = properties.split(',')
    prop_cls = prop_splits[0]
    if prop_cls == u'\u666f\u8272':
        continue
    class_names = {u'\u4eba\u5458':'person', u'\u8d8a\u91ce\u8f66':'SUV', u'\u8f7f\u8f66':'car', u'\u8fd0\u8f93\u8f66':'transport_vehicle'}
    
    if prop_cls == u'\u6709\u906e\u6321':
        if len(prop_splits)==1:
            continue
        prop_cls = prop_splits[1]
        obj_struct['difficult'] = 1
    if prop_cls == u'\u65e0\u906e\u6321':
        if len(prop_splits)==1:
            continue
        prop_cls = prop_splits[1]
        obj_struct['difficult'] = 0
    if len(prop_splits) == 1:
        obj_struct['difficult'] = 0
    elif prop_splits[1] == u'\u6709\u906e\u6321':
        obj_struct['difficult'] = 1
    else:# prop_splits[1] == u'\u65e0\u906e\u6321':
        obj_struct['difficult'] = 0
    obj_struct['name'] = class_names[prop_cls].strip()#obj.find('name').text
    obj_struct['pose'] = 'Unspecified'#obj.find('pose').text
    obj_struct['truncated'] = 0#int(obj.find('truncated').text)
    #obj_struct['difficult'] = 0#int(obj.find('difficult').text)
    #bbox = obj.find('bndbox')
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
    obj_struct['bbox'] = [int(x1),int(y1),int(x2),int(y2)]
    objects.append(obj_struct)
  
  #objects = []
  #f = open(filename,'r')
  #s = f.readlines()
  #num_objs = len(s)-1

  #for ix, obj in enumerate(s):
  #  if(ix == 0):
  #    continue
  #  obj_struct = {}
  #  obj_splits = obj.split(' ')
  #  if(float(obj_splits[5])>1):
  #    obj_struct['difficult']=int(1)
  #    #continue
  #  #if float(obj_splits[5])==0:
  #  #  obj_struct['difficult']=int(0)
  #  else:
  #    obj_struct['difficult']=int(0)
  #  obj_struct['name'] = obj_splits[0]
  #  obj_struct['bbox'] = [float(obj_splits[1])-1,
  #                        float(obj_splits[2])-1,
  #                        float(obj_splits[1]) + float(obj_splits[3]) - 1,
  #                        float(obj_splits[2]) + float(obj_splits[4]) - 1]
  #  #if float(obj_splits[5])>0:
  #  #  obj_struct['difficult']=int(1)
  #  #else:
  #  #  obj_struct['difficult']=int(0)
  #  objects.append(obj_struct)
  #f.close()

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             test_file,
             ovthresh=0.5,
             use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  #pdb.set_trace()
  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % 'imagesetfile')
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
      #f.close()
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  #pdb.set_trace()
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    #pdb.set_trace()
    for d in range(nd):
      R = class_recs[image_ids[d]]#ground truth boxes in a pic
      bb = BB[d, :].astype(float) #one detection in this pic
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  #pdb.set_trace()
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)
  if np.sum(fp>=0)==0:
      fp=[0]
  if np.sum(tp>=0)==0:
      tp=[0]
  test_file.write(str(fp[-1])+' '+str(tp[-1])+' '+str(ap)+' ')

  return rec, prec, ap
