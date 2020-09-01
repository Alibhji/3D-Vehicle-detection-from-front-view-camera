
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

PATH = './data/'
os.listdir(PATH)

BATCH_SIZE = 8
n_epochs = 40


train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'
# %% [markdown]
# # Load data

# %% [code] {"_kg_hide-input":true}
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

train.head()
test

# %% [code]
outlier_z3=['ID_039502422', 'ID_08b42e64c', 'ID_0b0be29be', 'ID_105f25107',
       'ID_149647474', 'ID_240e204c5', 'ID_2d2fff4cc', 'ID_3f658c1f9',
       'ID_4e758566d', 'ID_5a6def0fd', 'ID_5f6526a36', 'ID_6790d653a',
       'ID_7b81dab6e', 'ID_7d49a1db9', 'ID_7e321c4e5', 'ID_901fa9e6c',
       'ID_9e71e2a10', 'ID_a89688b6e', 'ID_a93677394', 'ID_ac946f5f9',
       'ID_ad4474603', 'ID_b02cee673', 'ID_b266869b4', 'ID_b63275d1b',
       'ID_b69ce0b4c', 'ID_be8a2ce07', 'ID_d005526a7', 'ID_d30bd5337',
       'ID_d87068c8e', 'ID_e67437f6e', 'ID_efa93b990', 'ID_f45e0c82e']

# train=train[~train['ImageId'].isin(outlier_z3)]
# train

# %% [markdown]
# **ImageId** column contains names of images:

# %% [code] {"_kg_hide-input":true}
def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

img = imread(PATH + 'train_images/ID_8a6e65317' + '.jpg')
IMG_SHAPE = img.shape

# plt.figure(figsize=(15,8))
# plt.imshow(img);

# %% [markdown]
# **PredictionString** column contains pose information about all cars  
# # 
# # From the data description:
# # > The primary data is images of cars and related pose information. The pose information is formatted as strings, as follows:  
# # >
# # > `model type, yaw, pitch, roll, x, y, z`  
# # >
# # > A concrete example with two cars in the photo:  
# # >
# # > `5 0.5 0.5 0.5 0.0 0.0 0.0 32 0.25 0.25 0.25 0.5 0.4 0.7`  
# # 
# # We will need a function to extract these values:

# %% [code]
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
inp = train['PredictionString'][0]
print('Example input:\n', inp)
print()
print('Output:\n', str2coords(inp))

# %% [markdown]
# # Data distributions

# %% [code] {"_kg_hide-input":true}
lens = [len(str2coords(s)) for s in train['PredictionString']]

# plt.figure(figsize=(15,6))
# sns.countplot(lens);
# plt.xlabel('Number of cars in image');

# %% [markdown]
# DataFrame of all points

# %% [code]
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

# plt.figure(figsize=(15,6))
# sns.distplot(points_df['roll'].map(lambda x: rotate(x, np.pi)), bins=500);
# plt.xlabel('roll rotated by pi')
# plt.show()

# %% [markdown]
# # 2D Visualization

# %% [code]
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

# plt.figure(figsize=(14,14))
# plt.imshow(imread(PATH + 'train_images/' + train['ImageId'][2000] + '.jpg'))
# plt.scatter(*get_img_coords(train['PredictionString'][2000]), color='red', s=100);

# %% [markdown]
# One point is out of image!

# %% [markdown]
# Let's look at the distribution of all points. Image is here just for reference.

# %% [code] {"_kg_hide-input":true}
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

# %% [code] {"_kg_hide-input":true}
# plt.figure(figsize=(16,16))
# plt.xlim(0,500)
# plt.ylim(0,100)
# plt.scatter(points_df['z'], points_df['y'], label='Real points')
# X_line = np.linspace(0,500, 10)
# plt.plot(X_line, zy_slope.predict(X_line.reshape(-1, 1)), color='orange', label='Regression')
# plt.legend()
# plt.xlabel('z coordinate')
# plt.ylabel('y coordinate');

# %% [markdown]
# # 3D Visualization
# # Used code from https://www.kaggle.com/zstusnoopy/visualize-the-location-and-3d-bounding-box-of-car, but made it one function

# %% [code] {"_kg_hide-input":true}
from math import sin, cos

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

# %% [code] {"_kg_hide-input":true}
def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image

# %% [code]
def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img

# %% [code] {"_kg_hide-input":true}
# n_rows = 6

# for idx in range(n_rows):
#     fig, axes = plt.subplots(1, 2, figsize=(20,20))
#     img = imread(PATH + 'train_images/' + train['ImageId'].iloc[idx] + '.jpg')
#     axes[0].imshow(img)
#     img_vis = visualize(img, str2coords(train['PredictionString'].iloc[idx]))
#     axes[1].imshow(img_vis)
#     plt.show()

# %% [markdown]
# # Image preprocessing

# %% [code]
ORG_IMG_HEIGHT  = 2710
ORG_IMG_WIDTH   = 3384

INPUT_HEIGHT  = 224
INPUT_WIDTH   = 840

INPUT_SCALE = 4
MASK_SCALE = 16  # 
# MASK_SCALE = 24 #--> mask size is 112*140


CUT_from_TOP = (22 + 896 +  (896* 3)//4) # pixels # 1590
CUT_from_Down =  (896* 1)//4
CUTTED_PIXEL_DOWN_LOCATION =  ORG_IMG_HEIGHT - CUT_from_Down

CUT_from_RIGHT=  24 # pixels
CUTTED_PIXEL_RIGHT_LOCATION = ORG_IMG_WIDTH - CUT_from_RIGHT

INPUT_SIZE= [(ORG_IMG_HEIGHT - ( CUT_from_TOP + CUT_from_Down))  // INPUT_SCALE , (ORG_IMG_WIDTH - CUT_from_RIGHT )// INPUT_SCALE]
MASK_SIZE = [(ORG_IMG_HEIGHT - ( CUT_from_TOP + CUT_from_Down)) // MASK_SCALE  , (ORG_IMG_WIDTH - CUT_from_RIGHT )// MASK_SCALE]

print('Orginal_SIZE :' , [ORG_IMG_HEIGHT , ORG_IMG_WIDTH])
print('INPUT_SIZE   :' ,INPUT_SIZE)
print('MASK_SIZE    :' ,MASK_SIZE)


def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z','yaw']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(regr_dict):
    for name in ['x', 'y', 'z' ,'yaw']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
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

# %% [code]
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import matplotlib.pyplot as plt
# from matplotlib import cm
# def Gus_kernel_at(xs,ys,peak=1):
#     x, y = np.meshgrid(np.linspace(0,MASK_SIZE[1],MASK_SIZE[1]-1), np.linspace(0,MASK_SIZE[1],MASK_SIZE[1]-1))
#     d = peak * np.sqrt(((x-xs)**2)+((y-ys)**2))
#     sigma, mu = 2.0, 1
#     g = 1* np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
#     return g
# #     print("2D Gaussian-like array:")

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x, y, g, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# img = imread(train_images_dir.format('ID_8a6e65317'))
# img = preprocess_image(img)
# mask, regr = get_mask_and_regr(img, train['PredictionString'][0])

# import plotly.graph_objects as go
# fig = go.Figure(data=[go.Surface(z=mask)])
# fig.show()

# %% [code]
# def multivariate_normal(x, d, mean, covariance):
#     """pdf of the multivariate normal distribution."""
#     x_m = x - mean
#     return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
#             np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


# mean = np.matrix([[0.], [1.]])
# covariance = np.matrix([
#     [1, 0.8], 
#     [0.8, 1]
# ])

# x, y = np.meshgrid(np.linspace(0,MASK_SIZE[1]-1,MASK_SIZE[1]), np.linspace(0,MASK_SIZE[0]-1,MASK_SIZE[0]))
# z=multivariate_normal((1,1), 1,mean,covariance )
# z

# %% [code]
# train
train['PredictionString'][0]

# %% [code]
img = imread(train_images_dir.format('ID_8a6e65317'))
img = preprocess_image(img)
mask, regr = get_mask_and_regr(img, train['PredictionString'][0])

print('img.shape', img.shape, 'std:', np.std(img))
print('mask.shape', mask.shape, 'std:', np.std(mask))
print('regr.shape', regr.shape, 'std:', np.std(regr))
# plt.figure(figsize=(16,16))
# plt.title('Processed image')
# plt.imshow(img)
# plt.show()

# plt.figure(figsize=(16,16))
# plt.title('Detection Mask')
# plt.imshow(mask)
# plt.show()

# plt.figure(figsize=(16,16))
# plt.title('Yaw values')
# plt.imshow(regr[:,:,-2])
# plt.show()

# %% [markdown]
# Define functions to convert back from 2d map to 3d coordinates and angles

# %% [code] {"_kg_hide-input":true}
DISTANCE_THRESH_CLEAR = 2

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x * MASK_SCALE ) + CUT_from_TOP
        y = (y * MASK_SCALE ) + CUT_from_RIGHT
        y = (y * MASK_SCALE ) + CUT_from_RIGHT
        return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

def extract_coords(prediction, flipped=False):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > 0)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
                optimize_xy(r, c,
                            coords[-1]['x'],
                            coords[-1]['y'],
                            coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

# %% [markdown]
# Ensure that all the forward and back transformations work consistently

# %% [code] {"_kg_hide-input":true}
# for idx in range(2):
#     fig, axes = plt.subplots(1, 2, figsize=(20,20))
    
#     for ax_i in range(2):
#         img0 = imread(PATH + 'train_images/' + train['ImageId'].iloc[idx] + '.jpg')
#         if ax_i == 1:
#             img0 = img0[:,::-1]
#         img = preprocess_image(img0, ax_i==1)
#         mask, regr = get_mask_and_regr(img0, train['PredictionString'][idx], ax_i==1)
#         regr = np.rollaxis(regr, 2, 0)
#         coords = extract_coords(np.concatenate([mask[None], regr], 0), ax_i==1)
        
#         axes[ax_i].set_title('Flip = {}'.format(ax_i==1))
#         axes[ax_i].imshow(visualize(img0, coords))
#     plt.show()

# %% [markdown]
# # PyTorch Dataset


# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
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
# %% [code]
#

# %% [code] {"_kg_hide-input":true}





df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
df_test = test

# Create dataset objects
train_dataset = CarDataset(df_train, train_images_dir, training=False)
dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
test_dataset = CarDataset(df_test, test_images_dir, training=False ,skip_masks = True)
test_dataset_all = CarDataset(train, train_images_dir, training=False , skip_masks = True )


# %% [markdown]
# Show some generated examples

# %% [code] {"_kg_hide-input":true}
# img, mask, regr = train_dataset[0]

# plt.figure(figsize=(16,16))
# plt.imshow(np.rollaxis(img, 0, 3))
# plt.show()

# plt.figure(figsize=(16,16))
# plt.imshow(mask)
# plt.show()

# plt.figure(figsize=(16,16))
# plt.imshow(regr[-2])
# plt.show()
# img.shape

# %% [code] {"_kg_hide-input":true}

# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_loader_All = DataLoader(dataset=test_dataset_all, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %% [markdown]
# # PyTorch Model

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
# !pip install efficientnet-pytorch

# %% [code]
# from efficientnet_pytorch import EfficientNet

# %% [code] {"_kg_hide-input":true}
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh

# %% [code] {"_kg_hide-input":true}
class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        #self.base_model = EfficientNet.from_pretrained('efficientnet-b7')
        model_name = 'fbresnet152' # could be fbresnet152 or inceptionresnetv2
        self.base_model = pretrainedmodels.__dict__['fbresnet152'](num_classes=1000, pretrained='imagenet')
#        self.base_model = list(self.base_model.children())[:-1] 
        
        self.conv0 = double_conv(5, 64)
 #       self.conv1 = double_conv(64, 256)
 #       self.conv1_2 = double_conv(128, 256)
        self.conv1 = double_conv(256, 256)
        self.conv2 = double_conv(1024, 512)
        self.conv3 = double_conv(512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)
        
        self.uupp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        
        
        self.mp = nn.MaxPool2d(2)
        
        self.up1 = up(1282 + 1024, 512)
        self.up2 = up(512 + 512, 256,bilinear=False)
        

    def forward(self, x):
        batch_size = x.shape[0]
        # print(self.base_model.layer3)
        x
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x = torch.cat([x, mesh1], 1)
        x = self.mp(self.conv0(x))
        # print('conv0  ', x.shape)
        x= self.base_model.layer1(x)
        # print('L1  ', x.shape)
        x = self.conv1(x)
        # print('conv1  ', x.shape)
        x= self.base_model.layer2(x)
        # print('L2  ', x.shape)
        x= self.base_model.layer3(x)
        # print('L3  ', x.shape)
        x = self.conv2(x)
        # print('conv2  ', x.shape)
        x = self.conv3(x)
        x = self.uupp(x)
        x = self.outc(x)
        
        #x = self.conv1(x)
        #print('conv1  ', x.shape)
        # 
        # mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        # x0 = torch.cat([x, mesh1], 1)
        # print('x0 ',x0.shape)
        # x1 = self.mp(self.conv0(x0))
        # print('x1 ',x1.shape)
        # x2 = self.mp(self.conv1(x1))
        # print('x2 ',x2.shape)
        # x2_ = (self.conv1_2(x2))
        # print('x2_ ',x2_.shape)
        # x3 = self.mp(self.conv2(x2_))
        # print('x3 ',x3.shape)
        # x4 = self.mp(self.conv3(x3))
        # print('x4 ',x4.shape)
        ##x4 = torch.cat([x4, x2], 1)
        ##print('x4Cat ',x4.shape)
        
        # x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        # feats = self.base_model.extract_features(x_center)
        # feats=self.convF(feats)
        # print('feats ',feats.shape)
        # bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        # feats = torch.cat([bg, feats, bg], 3)
        
        ##Add positional info
        # mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        # feats = torch.cat([feats, mesh2], 1)
        # print('feats ',feats.shape)
        
        # x = self.up1(feats, x4)
         
        # print('x ',x.shape)
        # x = self.up2(x, x3)
        
        # xs=self.convS(x3)
        # print('xs ',xs.shape)
        # x = torch.cat([xs, x], 1)
        

        # print('x ',x.shape)
        # x = self.outc(x)
        return x

# %% [code] {"_kg_hide-input":true}
# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



model = MyUNet(8).to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

# %% [code]
inp= torch.rand([1,3, 224, 840]).to(device)
print('input  ---->',inp.shape)
print('output ---->',model(inp).shape)
# cccc
# %% [code]
# gc.collect()
# gc.collect()
# gc.collect()

# %% [markdown]
# # Training

# %% [code] {"_kg_hide-input":true}
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

# %% [code] {"_kg_hide-input":true}
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

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
#%%time
import gc
import os
import pickle

save_dir= './kj20'
history = pd.DataFrame()


mode='eval'
# mode='train'
model_pretrained_name= os.path.join(save_dir , '_ep_10.model')

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

# %% [code]

# %% [code]


# %% [markdown]
# # Visualize predictions

# %% [code] {"_kg_hide-input":true}
#img, mask, regr = dev_dataset[0]

#plt.figure(figsize=(16,16))
#plt.title('Input image')
#plt.imshow(np.rollaxis(img, 0, 3))
#plt.show()

#plt.figure(figsize=(16,16))
#plt.title('Ground truth mask')
#plt.imshow(mask)
#plt.show()

# output = model(torch.tensor(img[None]).to(device))
# logits = output[0,0].data.cpu().numpy()

# plt.figure(figsize=(16,16))
# plt.title('Model predictions')
# plt.imshow(logits)
# plt.show()

# plt.figure(figsize=(16,16))
# plt.title('Model predictions thresholded')
# plt.imshow(logits > 0)
# plt.show()

# %% [code]
torch.cuda.empty_cache()
gc.collect()

# for idx in range(8):
    # img, mask, regr = dev_dataset[idx]
    
    # output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()
    # coords_pred = extract_coords(output[0])
    # coords_true = extract_coords(np.concatenate([mask[None], regr], 0))
    
    # img = imread(train_images_dir.format(df_dev['ImageId'].iloc[idx]))
    
    # fig, axes = plt.subplots(1, 2, figsize=(30,30))
    # axes[0].set_title('Ground truth')
    # axes[0].imshow(visualize(img, coords_true))
    # axes[1].set_title('Prediction')
    # axes[1].imshow(visualize(img, coords_pred))
    # plt.show()

# %% [markdown]
# # Make submission

print("predicion is started...")
predictions = []

# test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)

model.eval()


# evale_on_test_dataset_or_train='test'
evale_on_test_dataset_or_train='train'


if(evale_on_test_dataset_or_train =='train'):
    data_loader_ev=train_loader_All
    test = pd.read_csv(PATH + 'train.csv')
    cvs_file_name = 'predictions_train_{}.csv'.format(save_dir.split('/')[-1])
    pandas_file_name = 'predictions_train_{}.pkl.pandas'.format(save_dir.split('/')[-1])
    tensor_file_name= 'prediction_on_train_dataset_Tensors_{}_.pkl'.format(save_dir.split('/')[-1])
    
if(evale_on_test_dataset_or_train =='test'):
    data_loader_ev=test_loader
    test = pd.read_csv(PATH + 'sample_submission.csv')
    cvs_file_name = 'predictions_Test_{}.csv'.format(save_dir.split('/')[-1])
    pandas_file_name = 'predictions_Test_{}.pkl.pandas'.format(save_dir.split('/')[-1])
    tensor_file_name= 'prediction_on_Test_dataset_Tensors_{}_.pkl'.format(save_dir.split('/')[-1])

#for img, _, _ in tqdm(test_loader):
print('train_loader_All --> Dataset Len-->',len(data_loader_ev.dataset))

i__=0
predict_dict_res=OrderedDict()

for img, names in tqdm(data_loader_ev):
    with torch.no_grad():
        output = model(img.to(device))
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out)
        s = coords2str(coords)
        predictions.append(s)
        
    for i ,predict in enumerate(names):
        predict_dict_res.update({predict:output[i,...]})
    
    

# %% [code]





test['PredictionString'] = predictions
test.to_csv(os.path.join(save_dir, cvs_file_name), index=False)
test.head()
with open(os.path.join(save_dir, pandas_file_name), 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("predicion is done & csv file is saved.")

print('The length of predicted tensors:',len(predict_dict_res))

    #with open(os.path.join('./', 'train_prediction_{}.pkl'.format(model_name)), 'wb') as handle:
with open(os.path.join(save_dir, tensor_file_name), 'wb') as handle:
    pickle.dump(predict_dict_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("predicion Tensors are done & pkl file is saved.")