import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm#_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from scipy.optimize import minimize
#import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils

import pretrainedmodels
import pretrainedmodels.utils as utils
import time

from collections import OrderedDict
import gc
from collections import namedtuple
import gc
import os
import pickle

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import numpy as np
from math import sin, cos


BATCH_SIZE = 32
n_epochs   = 40 

save_dir= './kj21'
# mode='eval'
mode='train'
model_pretrained_name= os.path.join(save_dir , '_ep_10.model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


PATH=os.getcwd()
# PATH =os.path.abspath(PATH)
PATH='./data'

train = pd.read_csv(os.path.join(PATH , 'train.csv'))
test = pd.read_csv(os.path.join(PATH , 'sample_submission.csv'))


train_images_dir = os.path.join(PATH , 'train_images','{}.jpg')
test_images_dir = os.path.join(PATH , 'test_images','{}.jpg')

train_masks_dir= os.path.join(PATH , 'train_masks','{}.jpg')

Train_Masks= os.path.join(PATH,'train_masks')
Test_Masks= os.path.join(PATH,'test_masks')
Train_Masks=[item.split('.')[0] for item in os.listdir(Train_Masks)]
Test_Masks= [item.split('.')[0] for item in os.listdir(Test_Masks)]

# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inv = np.linalg.inv(camera_matrix)


outliers_xyx = ['ID_001a12fb2',
 'ID_012aef661',
 'ID_018444fd9',
 'ID_01c850b27',
 'ID_0260428a6',
 'ID_02b082818',
 'ID_030fb4808',
 'ID_039502422',
 'ID_03e1e30f0',
 'ID_0400854c6',
 'ID_042da2f13',
 'ID_04b8de41e',
 'ID_04f676276',
 'ID_058133001',
 'ID_05b615ca7',
 'ID_066750276',
 'ID_06b7202be',
 'ID_07375a6d5',
 'ID_075c0e11e',
 'ID_07836399f',
 'ID_07d3e4bf7',
 'ID_0865cb745',
 'ID_08b42e64c',
 'ID_090811cf5',
 'ID_09658d7d6',
 'ID_0b0be29be',
 'ID_0b4cab808',
 'ID_0b606383c',
 'ID_0b6d81cce',
 'ID_0beb902ad',
 'ID_0c6f8b911',
 'ID_0d1612e2f',
 'ID_0db064a7c',
 'ID_0ddda042d',
 'ID_0e30ba42f',
 'ID_0e396e0e9',
 'ID_0e6c355d9',
 'ID_0f3d9104c',
 'ID_0f7c7228a',
 'ID_105f25107',
 'ID_1073fe35b',
 'ID_12104686a',
 'ID_12267fd01',
 'ID_149647474',
 'ID_151fa0874',
 'ID_15a8550da',
 'ID_1606f0fcb',
 'ID_161640772',
 'ID_166c05a91',
 'ID_1677e1977',
 'ID_16c9192c8',
 'ID_16d084b43',
 'ID_175ade1ed',
 'ID_181adab76',
 'ID_1895b56ec',
 'ID_1963b1cc3',
 'ID_19b92bcc5',
 'ID_1a10eff59',
 'ID_1a4107469',
 'ID_1b53f57ae',
 'ID_1c5a74ad4',
 'ID_1c7ce36a4',
 'ID_1cc8e893f',
 'ID_1d9f515e5',
 'ID_1e468bfb0',
 'ID_1ea8b0007',
 'ID_1edd4d337',
 'ID_1fdccb4bb',
 'ID_20bda7f69',
 'ID_229a5f2ee',
 'ID_237fe5c9d',
 'ID_23f4a113e',
 'ID_240e204c5',
 'ID_2423f0e3d',
 'ID_243ea86e6',
 'ID_24418b6a6',
 'ID_27846370b',
 'ID_27acaaa8a',
 'ID_27f514d1b',
 'ID_27f6938ee',
 'ID_28ffad68c',
 'ID_297d3bc4c',
 'ID_2acf84328',
 'ID_2b3ff35b0',
 'ID_2b59c384d',
 'ID_2d2ce3d39',
 'ID_2d2fff4cc',
 'ID_2e9d7873d',
 'ID_2ee82c148',
 'ID_2f620ed54',
 'ID_2f99afb16',
 'ID_30a7c2a64',
 'ID_30abb829e',
 'ID_317b0b0ce',
 'ID_31903da16',
 'ID_32289333c',
 'ID_324d92b99',
 'ID_327979358',
 'ID_332e05c70',
 'ID_337ddc495',
 'ID_33cec9b19',
 'ID_33d42e307',
 'ID_34821aed0',
 'ID_3507e3624',
 'ID_357b01084',
 'ID_3583071f6',
 'ID_362d921bc',
 'ID_36465e439',
 'ID_36830dcd4',
 'ID_36ca6234f',
 'ID_36cc638b0',
 'ID_36db1e0ca',
 'ID_36e56f51c',
 'ID_374e97d7a',
 'ID_37be44db9',
 'ID_37dd679dc',
 'ID_37f431d28',
 'ID_3891e1ef4',
 'ID_38adf5c8e',
 'ID_39f28052d',
 'ID_3a5ef288d',
 'ID_3a7358d13',
 'ID_3a75649f5',
 'ID_3ae397744',
 'ID_3b2c71812',
 'ID_3c2fbeb77',
 'ID_3c31d847f',
 'ID_3cf1e870e',
 'ID_3d09a432b',
 'ID_3d8d6dd1d',
 'ID_3e075cfb4',
 'ID_3e29faeb0',
 'ID_3e34024dc',
 'ID_3ea678dcc',
 'ID_3f1c8f602',
 'ID_3f2d2f39c',
 'ID_3f658c1f9',
 'ID_3f8a89a03',
 'ID_400e4348a',
 'ID_40a669408',
 'ID_4199f4cd5',
 'ID_422d5d2d1',
 'ID_424d8de4b',
 'ID_427c943c9',
 'ID_42fff9c89',
 'ID_4314cce16',
 'ID_441886a76',
 'ID_44e4eb246',
 'ID_44fa78daa',
 'ID_456cbcf28',
 'ID_46ac698e4',
 'ID_46e6b4e59',
 'ID_48789938b',
 'ID_48b974f73',
 'ID_499da4132',
 'ID_4a2e2e15b',
 'ID_4a9cdac42',
 'ID_4aa239dcc',
 'ID_4b76d0f7a',
 'ID_4e758566d',
 'ID_501da7aca',
 'ID_50903cfac',
 'ID_50997c24e',
 'ID_50e5a474e',
 'ID_511911aad',
 'ID_512074fcf',
 'ID_517a14054',
 'ID_51f3c0953',
 'ID_5212cff85',
 'ID_5325864a1',
 'ID_534ba0ac4',
 'ID_5463219ec',
 'ID_55fef460c',
 'ID_565c40c14',
 'ID_567a39a77',
 'ID_57085da35',
 'ID_573e54a2d',
 'ID_577d1a984',
 'ID_57f52dbf4',
 'ID_585cc849d',
 'ID_593a711f9',
 'ID_5a6def0fd',
 'ID_5bb4d6d29',
 'ID_5be0a526c',
 'ID_5bf8b511d',
 'ID_5c97d93d9',
 'ID_5ce45c160',
 'ID_5d4ab3dd2',
 'ID_5d9a9a830',
 'ID_5dc0d941d',
 'ID_5e19e6af1',
 'ID_5eb939315',
 'ID_5f18e86f3',
 'ID_5f6526a36',
 'ID_5f8f50a1b',
 'ID_605cf1d46',
 'ID_612e8cd9a',
 'ID_616bdd8aa',
 'ID_626d7776d',
 'ID_62b4bed34',
 'ID_6301a5ee3',
 'ID_6367c3fff',
 'ID_63cfda92f',
 'ID_63d93ffd8',
 'ID_648c471c2',
 'ID_64c55c11b',
 'ID_6502e3a9d',
 'ID_657eb589c',
 'ID_65dd833a7',
 'ID_66114a2bb',
 'ID_66520c664',
 'ID_666ed008e',
 'ID_6682cb3ca',
 'ID_675fc5d8a',
 'ID_6790d653a',
 'ID_688d64128',
 'ID_689760cdb',
 'ID_68ab23939',
 'ID_68db82c42',
 'ID_6a4fa3d45',
 'ID_6a6d565ca',
 'ID_6a8c65399',
 'ID_6b2854770',
 'ID_6b8e7dbb7',
 'ID_6bb20ff30',
 'ID_6bcfbe419',
 'ID_6c4cb7231',
 'ID_6ce9299a5',
 'ID_6d0a129ab',
 'ID_6d58a0364',
 'ID_6da537078',
 'ID_6e21495b6',
 'ID_6e2f713ca',
 'ID_6e8591a14',
 'ID_6e9fe6af1',
 'ID_6f67a0c55',
 'ID_6ff92a2e4',
 'ID_701ebdc56',
 'ID_704e13e8c',
 'ID_70cbd9f23',
 'ID_7147e4e12',
 'ID_72283017f',
 'ID_722c0043b',
 'ID_72475c36b',
 'ID_729a8ef7b',
 'ID_72f7c5a14',
 'ID_7344a6126',
 'ID_7344dc5fe',
 'ID_738a1d889',
 'ID_74418d18e',
 'ID_745dce7e9',
 'ID_749c4d3bd',
 'ID_74e94a2db',
 'ID_750c68912',
 'ID_754e6c384',
 'ID_75584dd71',
 'ID_7621ea7f6',
 'ID_7705b405f',
 'ID_77a720d93',
 'ID_77d0f1fd1',
 'ID_77eb01dff',
 'ID_780b7ca82',
 'ID_787485a68',
 'ID_789fe31ca',
 'ID_78ede13b5',
 'ID_7aa5be52c',
 'ID_7b81dab6e',
 'ID_7b8967cb0',
 'ID_7c1a543f9',
 'ID_7c861e895',
 'ID_7d0ab7438',
 'ID_7d49a1db9',
 'ID_7d9239a52',
 'ID_7d97e26ae',
 'ID_7dcacedd5',
 'ID_7e321c4e5',
 'ID_7e4d94572',
 'ID_7ec29eaad',
 'ID_7f6f07350',
 'ID_807337723',
 'ID_80aecd428',
 'ID_817333c1c',
 'ID_817e2ef01',
 'ID_819575215',
 'ID_8199e5af1',
 'ID_8231155b1',
 'ID_82d727486',
 'ID_82f97f58d',
 'ID_83037d345',
 'ID_83cff8701',
 'ID_84047de00',
 'ID_849a2ae15',
 'ID_86ec7de88',
 'ID_88833c3ee',
 'ID_88a99396a',
 'ID_8abf3818f',
 'ID_8ad1639d4',
 'ID_8c00fc538',
 'ID_8c5850283',
 'ID_8c61b6f15',
 'ID_8cef10e05',
 'ID_8d5cbc1e6',
 'ID_8da41522e',
 'ID_8dcb9b2e2',
 'ID_8e41c194b',
 'ID_8e61da13b',
 'ID_8eafe31e3',
 'ID_8eb2669b5',
 'ID_8ff75c1aa',
 'ID_901117b49',
 'ID_901fa9e6c',
 'ID_906f44587',
 'ID_90d3e0e80',
 'ID_90e909712',
 'ID_912ad6db8',
 'ID_91b2a1bb9',
 'ID_91dae45bc',
 'ID_920fbfaf1',
 'ID_937edca6c',
 'ID_94a4784ef',
 'ID_94d690cbd',
 'ID_952360a4e',
 'ID_95276c148',
 'ID_956fa5a43',
 'ID_9587b899b',
 'ID_95f0f9f7d',
 'ID_96a20096b',
 'ID_96f7fd567',
 'ID_97157765e',
 'ID_973bfd1fb',
 'ID_97445a4aa',
 'ID_97fd761ea',
 'ID_983c1e248',
 'ID_98aee2a8e',
 'ID_99f8189f6',
 'ID_9a44a546e',
 'ID_9aadf2e49',
 'ID_9aeb19745',
 'ID_9bd2e72b9',
 'ID_9ca47e157',
 'ID_9cd22db32',
 'ID_9d1f97fb1',
 'ID_9dc0252ab',
 'ID_9e2174dfe',
 'ID_9e4c3af75',
 'ID_9e6e1ad85',
 'ID_9e71e2a10',
 'ID_9ea507b62',
 'ID_9ea9b90a0',
 'ID_9eef06273',
 'ID_9f3f0a78a',
 'ID_9fdd4f9a9',
 'ID_a0d7b5db9',
 'ID_a0e1b638a',
 'ID_a1147b159',
 'ID_a13c0ea5d',
 'ID_a1e4a213c',
 'ID_a20df07ec',
 'ID_a27f01e8d',
 'ID_a2a4dad88',
 'ID_a337525c3',
 'ID_a38d7e70d',
 'ID_a4057390a',
 'ID_a4138397b',
 'ID_a4c30e644',
 'ID_a516085b6',
 'ID_a5c1e2b3d',
 'ID_a6bf8f541',
 'ID_a6f146710',
 'ID_a7f98119b',
 'ID_a822f3885',
 'ID_a89688b6e',
 'ID_a9273c5a9',
 'ID_a93677394',
 'ID_a94be1fba',
 'ID_a97d9b416',
 'ID_a9d36e8db',
 'ID_aa51f342c',
 'ID_aa645dfab',
 'ID_ab3e1ad9f',
 'ID_ac121d381',
 'ID_ac946f5f9',
 'ID_ac9e2fdca',
 'ID_aca3a70d4',
 'ID_accefc9c9',
 'ID_acd20715a',
 'ID_ad0b4b072',
 'ID_ad4474603',
 'ID_ad50e86d0',
 'ID_ad77c0df0',
 'ID_ad98734fb',
 'ID_af0603b16',
 'ID_b02cee673',
 'ID_b06a5c779',
 'ID_b1008ef7c',
 'ID_b1aea0800',
 'ID_b266869b4',
 'ID_b3be748fc',
 'ID_b405f63d7',
 'ID_b42125d61',
 'ID_b4b15b8f9',
 'ID_b4d6b176b',
 'ID_b5ebe839b',
 'ID_b63275d1b',
 'ID_b69ce0b4c',
 'ID_b6c16c7fb',
 'ID_b6e911d41',
 'ID_b745d0bf8',
 'ID_b77a0da76',
 'ID_b8e105ca2',
 'ID_ba0126346',
 'ID_ba069e4ae',
 'ID_ba0a13999',
 'ID_ba123226f',
 'ID_ba5963dd1',
 'ID_bae889e7f',
 'ID_bb333df1b',
 'ID_bbe62ed9c',
 'ID_bc0ddb93b',
 'ID_bceb075be',
 'ID_bd25ccccf',
 'ID_bdd08f0f8',
 'ID_be01e23e7',
 'ID_be4698ce8',
 'ID_be5faedf0',
 'ID_be8a2ce07',
 'ID_beaf86342',
 'ID_befcefadb',
 'ID_bf2e58fbc',
 'ID_bfd83f639',
 'ID_bff365e13',
 'ID_c0d3f4329',
 'ID_c1dc1e7df',
 'ID_c1edc9dea',
 'ID_c2a614edc',
 'ID_c2f349103',
 'ID_c316009a9',
 'ID_c31679c55',
 'ID_c34d42a95',
 'ID_c39aa96e7',
 'ID_c3e6c3231',
 'ID_c46c05e9f',
 'ID_c4f92b64e',
 'ID_c555f0edd',
 'ID_c5bf3135e',
 'ID_c607d49fc',
 'ID_c60b3eea2',
 'ID_c671f55b5',
 'ID_c6af78321',
 'ID_c6d9ad796',
 'ID_c714ce672',
 'ID_c73a44ceb',
 'ID_c754c08ad',
 'ID_c7861ca2c',
 'ID_c7d63e9d3',
 'ID_c89179341',
 'ID_c8d2801b8',
 'ID_c9087b6bb',
 'ID_caa84eb2c',
 'ID_cb5d35aee',
 'ID_cbcf128fc',
 'ID_cc37fa969',
 'ID_cc7290959',
 'ID_cd09cc695',
 'ID_cd4604d5d',
 'ID_cd84cf31a',
 'ID_cdb292aa4',
 'ID_cdbc00f91',
 'ID_ce786198b',
 'ID_cf1ab9cc0',
 'ID_cf9eaf994',
 'ID_cfc444477',
 'ID_d005526a7',
 'ID_d0cf6475e',
 'ID_d1c16e5ff',
 'ID_d1e61e771',
 'ID_d200ac60c',
 'ID_d20edb804',
 'ID_d25e0d5ab',
 'ID_d25e32e67',
 'ID_d2a3f748f',
 'ID_d2ae00381',
 'ID_d2cdb591b',
 'ID_d2f80a8de',
 'ID_d2fc3a7c4',
 'ID_d30bd5337',
 'ID_d404f979f',
 'ID_d44b9a2c6',
 'ID_d44bbe32c',
 'ID_d53fd940a',
 'ID_d54fc9d86',
 'ID_d6331ccf9',
 'ID_d6390f085',
 'ID_d6e26d4cf',
 'ID_d739de2bf',
 'ID_d78489179',
 'ID_d7ffe5830',
 'ID_d83c8cf22',
 'ID_d87068c8e',
 'ID_dc906a5d2',
 'ID_dca8288ec',
 'ID_dcecaf2dc',
 'ID_dcf4928d7',
 'ID_dd151a1b8',
 'ID_dd1bd7316',
 'ID_dd6592d06',
 'ID_de53777e3',
 'ID_df134dd43',
 'ID_df2f7a255',
 'ID_dfd1abfab',
 'ID_e05bd3172',
 'ID_e1407b0f5',
 'ID_e21e6fa83',
 'ID_e2bd06410',
 'ID_e35cf2bba',
 'ID_e463040d2',
 'ID_e4bef04bb',
 'ID_e54dfa981',
 'ID_e5548f512',
 'ID_e56d83468',
 'ID_e58264b7f',
 'ID_e59604e7e',
 'ID_e5d1c1ebb',
 'ID_e63e99a6b',
 'ID_e662fcb78',
 'ID_e67437f6e',
 'ID_e8842ee5d',
 'ID_e8cb5d892',
 'ID_ea0ca7caf',
 'ID_ea87260d5',
 'ID_eafe67422',
 'ID_eb117ac6b',
 'ID_eb37b53de',
 'ID_ec0e083f4',
 'ID_edc1b60d4',
 'ID_edd28a56a',
 'ID_ef8228a54',
 'ID_efa93b990',
 'ID_efcff451a',
 'ID_f1354858c',
 'ID_f1fcf51f5',
 'ID_f2aeeab18',
 'ID_f2b92b119',
 'ID_f32a96913',
 'ID_f388a7b94',
 'ID_f45e0c82e',
 'ID_f4f80c710',
 'ID_f64511e66',
 'ID_f67cc13ff',
 'ID_f79e23bf5',
 'ID_f7c3b39db',
 'ID_f865e2a0d',
 'ID_f8cfb1759',
 'ID_f918cb91b',
 'ID_fa021e056',
 'ID_fa1d1426c',
 'ID_fa2998277',
 'ID_fa61068ea',
 'ID_fc949ae29',
 'ID_fce05fc08',
 'ID_fe5acc7a0',
 'ID_fee391a62',
 'ID_ffb67214c']
 
train=train[~train['ImageId'].isin(outliers_xyx)]
len(train)


ORG_IMG_HEIGHT  = 2710
ORG_IMG_WIDTH   = 3384

INPUT_HEIGHT  = 224
INPUT_WIDTH   = 864

INPUT_SCALE = 4
MASK_SCALE = 16
# MASK_SCALE = 24 #--> mask size is 112*140


CUT_from_TOP = (22 + 896 +  (896* 3)//4) # pixels # 1590
CUT_from_Down =  (896* 1)//4
CUTTED_PIXEL_DOWN_LOCATION =  ORG_IMG_HEIGHT - CUT_from_Down

CUT_from_RIGHT=  56 # pixels
CUTTED_PIXEL_RIGHT_LOCATION = ORG_IMG_WIDTH - CUT_from_RIGHT

INPUT_SIZE= [(ORG_IMG_HEIGHT - ( CUT_from_TOP + CUT_from_Down))  // INPUT_SCALE , (ORG_IMG_WIDTH - CUT_from_RIGHT )// INPUT_SCALE]
MASK_SIZE = [(ORG_IMG_HEIGHT - ( CUT_from_TOP + CUT_from_Down)) // MASK_SCALE  , (ORG_IMG_WIDTH - CUT_from_RIGHT )// MASK_SCALE]

print('Orginal_SIZE :' , [ORG_IMG_HEIGHT , ORG_IMG_WIDTH])
print('INPUT_SIZE   :' ,INPUT_SIZE)
print('MASK_SIZE    :' ,MASK_SIZE)


def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img



def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

# %% [code] {"_kg_hide-input":true}
# inp = train['PredictionString'][0]
# print('Example input:\n', inp)
# print()
# print('Output:\n', str2coords(inp))


points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr

print('total number of the objects in the train data set is:', len(points_df))

def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x
    
    
def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys


xs, ys = [], []

for ps in train['PredictionString']:
    x, y = get_img_coords(ps)
    xs += list(x)
    ys += list(y)



# %% [code]
zy_slope = LinearRegression()
X = points_df[['z']]
y = points_df['y']
zy_slope.fit(X, y)
print('MAE without x:', mean_absolute_error(y, zy_slope.predict(X)))

# Will use this model later
xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)
print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))

print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*xzy_slope.coef_))



def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
#     for name in ['x', 'y', 'z']:
#         regr_dict[name] = regr_dict[name] / 100
    regr_dict['x'] = regr_dict['x'] / 40
    regr_dict['y'] = regr_dict['y'] / 30
    regr_dict['z'] = regr_dict['z'] / 180
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict


def preprocess_image(img, flip=False):
    img = img[CUT_from_TOP: CUTTED_PIXEL_DOWN_LOCATION , : CUTTED_PIXEL_RIGHT_LOCATION]
    H_  = img.shape[0]  //  INPUT_SCALE # 2688/3 --> 896
    W_  = img.shape[1] //  INPUT_SCALE # 3884/3 --> 1120
    img = cv2.resize(img, ( W_ ,H_))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def Gus_kernel_at(mask ,xs,ys,peak=1):
    x, y = np.meshgrid(np.linspace(0,MASK_SIZE[1]-1,MASK_SIZE[1]), np.linspace(0,MASK_SIZE[0]-1,MASK_SIZE[0]))
    d = peak * np.sqrt(((x-xs)**2)+((y-ys)**2))
    sigma, mu = 1.0, 0
    g = peak* np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    mask[g>0] = g [ g>0]+ mask[g>0]
    
    return mask

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([MASK_SIZE[0] , MASK_SIZE[1]], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([MASK_SIZE[0] , MASK_SIZE[1] ,7], dtype='float32')
    coords = str2coords(labels)
#     print(coords)
    xs, ys = get_img_coords(labels)
    for y, x, regr_dict in zip(xs, ys, coords):
        x , y = y ,x
#         print(x,y, regr_dict)
        xs = int((x ) // MASK_SCALE)
#         print('y',ys)
        ys = int(( y  - CUT_from_TOP ) // MASK_SCALE )
#         print('y',ys)
#         print(x,y  , ( CUTTED_PIXEL_RIGHT_LOCATION // MASK_SCALE ) , (CUTTED_PIXEL_DOWN_LOCATION // MASK_SCALE))
        if xs >= 0 and (xs < ( CUTTED_PIXEL_RIGHT_LOCATION // MASK_SCALE ))and ys >= 0 and ys < ((CUTTED_PIXEL_DOWN_LOCATION - CUT_from_TOP) // MASK_SCALE):
#             mask = Gus_kernel_at (mask ,xs, ys)
            mask[ys, xs] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
#             print((regr_dict))
            regr[ys, xs] = [regr_dict[n] for n in sorted(regr_dict)]
#     if flip:
#         mask = np.array(mask[:,::-1])
#         regr = np.array(regr[:,::-1])
    return mask, regr



img = imread(train_images_dir.format('ID_8a6e65317'))
img = preprocess_image(img)
mask, regr = get_mask_and_regr(img, train['PredictionString'][0])

print('img.shape', img.shape, 'std:', np.std(img))
print('mask.shape', mask.shape, 'std:', np.std(mask))
print('regr.shape', regr.shape, 'std:', np.std(regr))


class CarDataset(Dataset ):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=False, transform=None , skip_masks=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training
        self.skip_masks = skip_masks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1
        
        # Read image
        img0 = imread(img_name)
        imgg = preprocess_image(img0, flip=flip)
        imgg = np.rollaxis(imgg, 2, 0)
        
        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)
        
        
        if (self.skip_masks):
            return [imgg, idx]

        return [imgg, mask, regr]    
        
        
df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
df_test = test

# Create dataset objects
train_dataset = CarDataset(df_train, train_images_dir, training=False)
dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
test_dataset = CarDataset(df_test, test_images_dir, training=False ,skip_masks = True)
test_dataset_all = CarDataset(train, train_images_dir, training=False , skip_masks = True )



train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_loader_All = DataLoader(dataset=test_dataset_all, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)





class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
#         self.mp = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_ch, 8, 3, padding=1),
            nn.BatchNorm2d(8)

            
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MyModel(torch.nn.Module):
    
    def __init__(self):
        super(MyModel, self).__init__()
        # self.model_name = 'resnet34'
        # model_name = 'xception'
        # model_name = 'mobilenet_v2'
        # model_name = 'vgg19'
        # model_name = 'resnet18'
        self.model_name = 'resnet152'
        print('model_name: ', self.model_name)

        self.base_model =  smp.Unet()
        self.base_model =  smp.Unet(self.model_name, encoder_weights='imagenet')
#         self.base_model =  smp.Unet(self.model_name, classes=3, activation='softmax')
        self.base_model.segmentation_head =  double_conv(16,64)
        

        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
# print( model )    

model = MyModel().to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)


inp= torch.rand([1,3, 224, 832]).to(device)
print('input  ---->',inp.shape)
print('output ---->',model(inp).shape)


def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss



def train_model(epoch, history=None):
    ep_since = time.time()  
    model.train()
    running_loss = 0.0
    
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        since = time.time()
        
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()
        
        optimizer.step()
        exp_lr_scheduler.step()
        time_elapsed = time.time() - since
        print('Ep:{:2d} - {:3d}/{:3d} loss: {:.5f} , time: {:.4f} '.format(epoch,batch_idx,len(train_loader),loss.item(),time_elapsed))
        running_loss += loss.item() * img_batch.size(0)

        
    
    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))

def evaluate_model(epoch, history=None):
    model.eval()
    loss = 0
    
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss += criterion(output, mask_batch, regr_batch, size_average=False).data
    
    loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    print('Dev loss: {:.4f}'.format(loss))
    





history = pd.DataFrame()

if(mode == 'eval'):    
    torch.cuda.empty_cache()
    gc.collect()
    model.load_state_dict(torch.load(model_pretrained_name)) 
    print("Model is loaded.")
    evaluate_model(1)

if(mode == 'train'): 
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train_model(epoch, history)
        evaluate_model(epoch, history)
        torch.save(model.state_dict(), os.path.join(save_dir, '_ep_{}.model'.format(epoch)))

        with open(os.path.join(save_dir, 'history.pkl'), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # %% [code]
    torch.save(model.state_dict(), './model.pth')