import os
import shutil
import pdb

filename = './EELABdevkit/data/ImageSets/Main/test.txt'
f = open(filename, 'r')
s = f.readlines()
num_imgs = len(s)

pdb.set_trace()
for i in range(num_imgs):
    #img_RGB = os.path.join('./EELABdevkit/data/images/visible', s[i][:-1]+'.jpg')
    #img_T   = os.path.join('./EELABdevkit/data/images/infrared', s[i][:-1]+'.jpg')
    anno = os.path.join('./EELABdevkit/data/annotations', s[i][:-1]+'.xml')

    #shutil.copyfile(img_RGB, './demo/C'+s[i][:-1]+'.jpg')
    #shutil.copyfile(img_T, './demo/T'+s[i][:-1]+'.jpg')
    shutil.copyfile(anno, './demo_anno/'+s[i][:-1]+'.xml')
