from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
from PIL import Image
import torchvision as tv
import numpy as np
import xml.etree.ElementTree as ET

VOC_ROOT = '//datasets/ee285f-public/PascalVOC2012/'

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class VOCAnnotationTransform(object):
    '''
    Initializes an instance with a dictionary of classname:index mappings.
    The default dictionary used is an alphabetic indexing of VOS's 20 classes.
    The difficult instances can/cannot be kept by setting the keep_difficult parameter.
    '''
    
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):       
        '''
        Return list of list of bounding boxes characterized by coordinates and class names, given the target annotation, height and width.
        '''
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res 


class VOCDetection(data.Dataset):
    '''VOC dataset detection object.
    Updates annotation based on input image, which has following attributes:
        - root directory
        - which set (train/va/test)
        - transformation to perform on image
        - transformation to perform on target
        - dataset to load (VOC2012, COCO etc)'''
    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712',sigma=30):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.sigma = sigma
        
        
        for (year, name) in image_sets:
            rootpath = self.root
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        '''Return torch from numpy of image after transform, given index of image'''
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''
        Returns image in PIL form given index of image. __getitem__ cannot be used since transformations may change organization of data.
        '''
    
        img_id = self.ids[index]
        im, gt, h, w = self.pull_item(index)
        im = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        return im, gt, h, w
    
        
    def pull_anno(self, index):
        '''
        Return list of attributes of original annotation of image at index - [img_id, [(label, bbox coords, ...] given index of image to           get annotation of.
        '''
        
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns tensorized and squeezed version of image given index of image.'''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
