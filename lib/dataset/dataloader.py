'''
This is a dataloaser class.
All the configuration are given by cfg argument
This code is written by Ali Babolhaveji at 22/01/2019
'''

import argparse
import yaml
import os
import logging
import copy
import numpy as np
import time

import pandas as pd

import cv2


from  torch.utils.data  import  Dataset
from  pycocotools.coco import  COCO

logger = logging.getLogger(__name__)

class general_dataset(Dataset):
    def __init__(self , cfg , dataset_type = 'train'):
        self.cfg = cfg
        self. root = cfg['DATASET']['ROOT']
        self. anno_dir = os.path.join(self. root , 'annotations')
        self. imge_dir = os.path.join(self.root, 'images',dataset_type+'2017')

        self. dataset_type = dataset_type
        self.feed_data_per_object = cfg['DATASET']['FEED_DATA_PER_OBJECT']

        self.num_joints = cfg['PREPROCESS']['NUM_JOINTS']


        # self. anno_pandas = os.path.join(self. root, 'annotations' ,  cfg['DATASET']['ANNOTATION_PANDAS']+ '{}.pandas')
        anno_json   = os.path.join(self.root , 'annotations' ,  cfg['DATASET']['ANNOTATION_JSON']+ '_{}.json'.format(self. dataset_type ))
        self. coco = COCO(anno_json)
        print('[info] [{} data is loaded.] from : {}'.format(self. dataset_type ,anno_json))
        print('[info] Dataset information:')
        self.cats = self.coco.getCatIds()  # [3]
        self.Imgs_Ids = self.coco.getImgIds(catIds = self.cats)
        self.Imgs_anns = self.anns_to_dict()


        # prepross



        self.cut_top   = cfg['PREPROCESS']['CROP']['TOP']
        self.cut_down  = cfg['PREPROCESS']['ORGINAL_IMAGE_SIZE']['YS'] - cfg['PREPROCESS']['CROP']['DOWN']
        self.cut_left  = cfg['PREPROCESS']['CROP']['LEFT']
        self.cut_right = cfg['PREPROCESS']['ORGINAL_IMAGE_SIZE']['XS'] - cfg['PREPROCESS']['CROP']['RIGHT']

        self.scale_ys = cfg['PREPROCESS']['SCALE']['YS']
        self.scale_xs = cfg['PREPROCESS']['SCALE']['XS']

        self.org_imgsize_ys = cfg['PREPROCESS']['ORGINAL_IMAGE_SIZE']['YS']
        self.org_imgsize_xs = cfg['PREPROCESS']['ORGINAL_IMAGE_SIZE']['XS']



        print(self.coco.loadCats(self.cats))
        print('[info] Contains: {} images which it has {} of objects.'.format(len(self.coco.getImgIds(catIds = self.cats)) , len(self.coco.getAnnIds(catIds = self.cats))))


        if (cfg['DATASET']['CREATE_NEW_DATASET'] ):
            pass

    def __len__(self, ):

        if(self.feed_data_per_object):
            return len(self.coco.getAnnIds(catIds=self.cats))
        else:
            return len(self.coco.getImgIds(catIds=self.cats))


    def anns_to_dict(self):
        dict_ann={}
        anns = self.coco.getAnnIds(catIds=self.cats)
        for obj in range (len(anns)):
            dict_ann[obj]= anns[obj]
        return dict_ann

    def create_masks(self , anns):
        pass

    def pre_processing (self , image , anns , meta_data):
        m_data = dict()
        image = image [ self.cut_top : self.cut_down   , self.cut_left  : self.cut_right , :]

        height_ys = int(image.shape[0]/ self.scale_ys)
        width_xs  = int(image.shape[1]/ self.scale_xs )
        dim = (width_xs, height_ys)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        meta_data['info'] = m_data

        for i,ann in enumerate(anns):
            key_points = np.array(ann['keypoints']).reshape([-1,3])
            bbox = np.array(ann['bbox'])
            m_data['Orginal_key_points'] =  copy.deepcopy(key_points)
            key_points[:, 0] = key_points[:, 0] - self.cut_top
            key_points[:, 1] = key_points[:, 1] - self.cut_right
            key_points[:, 0] = (key_points[:, 0] / self.scale_ys).astype(int)
            key_points[:, 1] = (key_points[:, 1] / self.scale_xs).astype(int)
            m_data['key_points'] = key_points

            m_data['Orginal_bbox'] = copy.deepcopy(bbox)
            bbox[0] = bbox[0]  - self.cut_top

            bbox[0] = (bbox[0] / self.scale_ys).astype(int)
            bbox[2] = (bbox[2] / self.scale_ys).astype(int)
            bbox[1] = (bbox[1] / self.scale_xs).astype(int)
            bbox[3] = (bbox[3] / self.scale_xs).astype(int)
            m_data['bbox'] = bbox

        return image , anns , meta_data


    def Augmentation(self ,image, anns, meta_data ):

        image_size = np.array([image.shape[0] , image.shape[1]])
        center = image_size/2

        np.random.seed(int(time.time() * 1000) % 1000)
        # print(center)

        min_s = 1
        max_s = 3  # + degrees
        scale = min_s + np.random.rand(1) * (max_s - min_s)

        np.random.seed(int(time.time() * 1000) % 1000)
        min_a = 0
        max_a = 15  # +/- degrees
        angle = min_a + np.random.randn(1) * (max_a - min_a)
        #         print('angle         ',    angle)

        np.random.seed(int(time.time() * 1000) % 1000)
        min_shift = 0
        max_shift = 50  # +/-pixels
        shift = min_shift + np.random.randn(1) * (max_shift - min_shift)
        shift2 = min_shift + np.random.randn(1) * (max_shift - min_shift)

        if self.dataset_type =='train' :
            angle = angle
            scale = scale
            center = np.array(center)
            center[0] = center[0] + shift
            center[1] = center[1] + shift2
            center = tuple(center)
        else:
            angle = 0
            scale = 1



        trans = cv2.getRotationMatrix2D((center[0], center[1]), angle, scale)

        # image = cv2.warpAffine(
        #     image,
        #     trans,
        #     (int(image_size[0]), int(image_size[1])),
        #     flags=cv2.INTER_LINEAR)
        #
        # for i in range(self.num_joints):
        #     joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)



        return image, anns, meta_data



    def __getitem__(self, idx):
        # self.coco.loadImgs(idx)[0]['file_name']
        # print(idx)
        meta_data ={}

        if (self.feed_data_per_object ):
            ann_id  = self.Imgs_anns [idx]
            img_name = self.coco.loadAnns(ann_id)[0]['image_name']+'.jpg'
            anns = copy.deepcopy(self.coco.loadAnns(ann_id))
            # print(img_name, idx)



        else:
            img_id = self.Imgs_Ids[idx]
            anns = copy.deepcopy(self.coco.loadAnns(self.coco.getAnnIds(imgIds=idx)))
            # print(self.coco.loadImgs(img_id)[0]['file_name'] ,idx  )
            img_name = self.coco.loadImgs(img_id)[0]['file_name']

        meta_data['image_name'] = img_name

        image = cv2.imread(os.path.join(self.imge_dir, str(img_name)))

        image , anns , meta_data  = self.pre_processing (image , anns , meta_data)
        image, anns, meta_data    =  self.Augmentation(image , anns , meta_data)


        # print(image.shape ,meta_data )
        print("***********")
        # print(anns)




        # image augmentation

        # image , ann =










if __name__ == '__main__':

    # def parse_args():
    #     parser = argparse.ArgumentParser (description= "Train Network Arguments")
    #     parser.add_argument( '--cfg' ,help= 'Choose the configuration file. (*.yaml) , It can be found in the configuration directory.'  , required= True , type =str)
    #
    #     args =parser.parse_args()
    #     return args

    ## Alternative args definition within the program

    class parse_args():
        def __init__(self):
            self.cfg = '../../configuration/first_doc.yaml'


        def __str__(self):
            # return  "{}".format([var for var in dir(parse_args) if   var.startswith('__')] )
            return "{}".format(self.cfg )


    args = parse_args()
    with open(args.cfg , 'r') as cfg_file:
        cfg = yaml.load(cfg_file)
    print(__name__)

    Data = general_dataset(cfg)
    print(len(Data))

    for i in range (len(Data)):
        Data[i]
