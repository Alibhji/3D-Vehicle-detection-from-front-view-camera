# %% [code]
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from  torch.utils.data import DataLoader ,Dataset
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import os
import time
import copy
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils
from math import sin ,cos ,floor

from collections import OrderedDict

# %% [code]
#! pip install pretrainedmodels

import pretrainedmodels
import torch
import pretrainedmodels.utils as utils


BATCH_SIZE = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 10

# %% [code]
PATH = '../input/pku-autonomous-driving/'
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inv = np.linalg.inv(camera_matrix)

model_name = 'vgg19' # could be fbresnet152 or inceptionresnetv2
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# model.eval()

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


# %% [code]
from tqdm import tqdm
# import magic
data=[]


Train_Masks= os.path.join(PATH,'train_masks')
Test_Masks= os.path.join(PATH,'test_masks')

train_images_dir = os.path.join(PATH , 'train_images','{}.jpg')
test_images_dir = os.path.join(PATH , 'test_images','{}.jpg')


Train_Masks=[item.split('.')[0] for item in os.listdir(Train_Masks)]
Test_Masks= [item.split('.')[0] for item in os.listdir(Test_Masks)]





img_size_flag=False


for i,_ in enumerate(tqdm(range(len(train)))):
    data_pd=pd.DataFrame()
    # k=k.append(pd.DataFrame(.update({'id':train'[ImageId]'[10]})),ignore_index = True)
    xs=get_img_coords(train['PredictionString'][i])[0]
    ys=get_img_coords(train['PredictionString'][i])[1]
    for j ,dic in enumerate(str2coords(train['PredictionString'][i])):
        D= OrderedDict(dic)
        img_id=train['ImageId'][i]
        D.update({'ImageId':img_id})
        D.update({'Mask': img_id in Train_Masks})
        D.update({'xs':  xs[j]})
        D.update({'ys':  ys[j]})

        
#         if(img_size_flag):
# #             input_img = cv2.imread(train_images_dir.format(img_id))
#             input_img =magic.from_file(train_images_dir.format(img_id))
#             D.update({'size': tuple(np.array(input_img.split(',')[-2].split('x')).astype(int))})
            
            
        
        data.append(D)
data_pd=data_pd.append(pd.DataFrame(data))


# %% [code]
train_dataset_info=pd.DataFrame()

for item in ['xs' , 'ys', 'x' , 'y' , 'z', 'pitch' ,'roll' , 'yaw']:
    train_dataset_info[item]=data_pd.groupby('ImageId')[item].apply(list)
train_dataset_info['mask']=data_pd.groupby('ImageId').first()['Mask']

num_objs=[]
for num , row in train_dataset_info.iterrows():
    num_objs.append(len(row['xs']))
train_dataset_info['num_objs']=num_objs

# %% [code]
# train_dataset_info.columns
# train_dataset_info

# %% [code]
# row=pd.DataFrame(train_dataset_info[train_dataset_info.index=='ID_001d6829a'])


# %% [code]
def image_read(path, custom_scale=3):
    image_scale= custom_scale
    image_resize = ((3384-24)//image_scale ,(2710-22)//image_scale)
    
    im=cv2.imread(path)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.normalize(im.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    im=im[:2688,:3360]
    im=cv2.resize(im,image_resize)
    
    return im

def get_xs_ys_from_pd(row , for_gen_mask=False):
    
    cut_from_x=22 # pixcels
    cut_from_y=24 # pixcels
    resize_factor=3
    resize_mask_factor=12
    xs=np.array((list(row['xs'])[0])).astype('float32')
    ys=np.array((list(row['ys'])[0])).astype('float32')
    
#     ys=ys-674 # the pandas is save with this bias wrongly in previous code!
    if(for_gen_mask):
        resize_factor=resize_mask_factor
        
    xs=(xs-cut_from_x)/resize_factor # x is cuted from the top
    ys=(ys)/resize_factor # y is cutted  from the left so it dosent need to subtracted
    
    xs = np.round(xs).astype(int)
    ys = np.round(ys).astype(int)
    
    return zip(xs,ys)




def plot_image(name,dataset_pd ,show_mask=False ,custom_scale=3, figure_szie=10):
    row=pd.DataFrame(dataset_pd[dataset_pd.index==name])
    xys=get_xs_ys_from_pd(row)
    im=image_read(train_images_dir.format(name) ,custom_scale)
    
    for obj in xys:
        im=cv2.circle(im,(int(obj[0]),int(obj[1])),4,(255,0,0),2)
        
    plt.figure(figsize=(figure_szie,figure_szie))
    plt.imshow(im)
    mask_flag=list(row['mask'])[0]
    
    if(mask_flag and show_mask):
        print('It has a mask')
        mask=image_read(train_masks_dir.format(name).format(name) ,custom_scale)
        plt.imshow( mask , alpha= 0.5)
        
    

# %% [code]
# name='ID_001d6829a'
# plot_image(name,train_dataset_info ,show_mask=True)

# %% [code]
def create_mask_regres(name ,mask_size= (224,280), regres_title=['yaw', 'pitch', 'roll', 'x', 'y', 'z'],dataset_pd=train_dataset_info):


    regres=np.zeros([len(regres_title),mask_size[0],mask_size[1]],dtype='float32')
    row=pd.DataFrame(dataset_pd[dataset_pd.index==name])

    xys = get_xs_ys_from_pd(row ,for_gen_mask=True)
    yaw=np.array((list(row['yaw'])[0])).astype('float32')
    pitch=np.array((list(row['pitch'])[0])).astype('float32')
    roll=np.array((list(row['roll'])[0])).astype('float32')
    x=np.array((list(row['x'])[0])).astype('float32')
    y=np.array((list(row['y'])[0])).astype('float32')
    z=np.array((list(row['z'])[0])).astype('float32')

    yaw = yaw /100
    pitch = pitch /100
    roll = roll /100

    x = x /100
    y = y /100
    z = z /100

    label_pd= pd.DataFrame({'yaw':list(yaw)} )
    label_pd['pitch']=list(pitch)
    label_pd['roll']=list(roll)
    label_pd['x']=list(x)
    label_pd['y']=list(y)
    label_pd['z']=list(z)


    xys=list(i for i in xys)
    # label_pd=pd.DataFrame([{'yaw':list(yaw)} , {'pitch':list(pitch)}, {'roll':list(roll)}, {'x':list(x)}, {'y':list(y)}, {'z':list(z)}])
    # label_pd
    for i in range(len(label_pd)):
        xs = xys[i][0]
        ys = xys[i][1]
        for ind, lbl in enumerate(regres_title):
            regres[ind,ys-1:ys+1,xs-1:xs+1] = label_pd[lbl][i]
            
        mask=regres[0,:,:]
#         regres=np.moveaxis(regres, 1, 2) 
        mask[mask>0]= 1
    
    return mask , regres

    
# mask , regres = create_mask_regres(name)
# plot_image(name,train_dataset_info,custom_scale=12,figure_szie=16)
# plt.imshow(mask,alpha=.6)

# %% [code]
class Baidu_Autonomous_dataset(Dataset):
    def __init__(self, dataframe , root_dir,custom_scale=3):
        self.df = dataframe
        self.root_dir = root_dir
        self.custom_scale=custom_scale

    
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        img_name, labels = self.df.values[idx] 
        path=self.root_dir.format(img_name)
        input_img = image_read(path , self.custom_scale)
        mask , regres= create_mask_regres (img_name)
        input_img = np.transpose(input_img, (2, 0, 1))
#         img_name=img_name.moveaxis( 0, -1)

        return [input_img,mask, regres, img_name]

# %% [code]
train_images_dir = os.path.join(PATH , 'train_images/{}.jpg')
test_images_dir = os.path.join(PATH , 'test_images/{}.jpg')

train_dataset = Baidu_Autonomous_dataset(train, train_images_dir )

# %% [code]
# print('Train dataset size:',len(train_dataset))

# %% [code]
# img, mask, regres,name=train_dataset[0]
# print(img.shape ,mask.shape ,regres.shape)
# img=img.transpose((1, 2, 0))
# img=cv2.rectangle(img, (0,0) , (img.shape[1] ,img.shape[0]), (255,0,0),2)
# plt.figure(figsize=(16,16))
# plt.imshow(img)

# # mImg=np.zeros_like(img[:,:,1])
# # stdImg=np.zeros_like(img[:,:,1])
# # dataset_info= OrderedDict()
# plt.figure(figsize=(16 ,16))
# plt.imshow(img)
# plt.imshow(mask , alpha= 0.5)
# plt.imshow(regres[5,...] , alpha= 0.5)

# for num ,data in enumerate(train_dataset):
#     temp=OrderedDict()
#     mImg += np.array(data[0]).mean(axis=2)
#     stdImg += np.array(data[0]).std(axis=2)
#     temp. update({'Mean' : np.array(data[0]).mean() })
#     temp. update({'STD' : np.array(data[0]).std() })
#     temp. update({'size' : np.array(data[0]).shape})
#     print('{}/{}  {}'.format(num , len(train_dataset),data[1]))
#     dataset_info.update({data[1]:temp})
# #     if num == 5:
# #         break


# %% [code]
# pd.DataFrame(dataset_info).T['Mean'].to
# pd.DataFrame(dataset_info).T.to_csv('train_data.csv')


# %% [code]
# plt.figure(figsize=(16,16))
# plt.imshow(mImg/4262)
# # # ! ls
# # from IPython.display import FileLink
# # FileLink(r'train_data.csv')

# %% [code]
# # torch.cuda.empty_cache()
# # del  tensor_o model
# # torch.cuda.reset_max_memory_cached()
# # print(torch.cuda.memory_cached())
# # print(torch.cuda.memory_allocated())

# scale=3

# # tensor_o = torch.rand([1,3,(2710-10)//scale,(3384-4)//scale]).to(device)
# print([(3360)/scale,(2688)/scale])
# tensor_o = torch.rand([1,3,(3360)//scale,(2688)//scale]).to(device)

# # tensor_o = torch.rand([1,3,32,32]).to(device)

# print('input size :',list(tensor_o.shape))

# # tensor_o = torch.rand([1,3,900,900]).to(device)
# output = model(tensor_o)

# print('output size:',list(output.shape))
# # (3360/32/3)*scale_mask, (2700/32/3)*scale_mask


# # del model , tensor_o ,output

# %% [code]
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh


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
    
class de_conv2x(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(de_conv2x, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,  2, stride=2),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
#         features = list(vgg16(pretrained = True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
#         self.features = nn.ModuleList(features).eval()
        base_model=pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        base_model=base_model._features
        base_model = list(base_model.children())[0:]
        self.base_model = nn.ModuleList(base_model).eval()
        
        self.conv0 = double_conv(69, 128)
        self.conv1 = double_conv(256, 256)
        self.conv2 = double_conv(512, 512)
        self.conv3 = double_conv(1536, 256)
        self.mp = nn.MaxPool2d(2)
        self.up2x= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
#         self.deconvF4 = de_conv2x(512,512)
        
#         self.deconv0 = de_conv2x(256,125)
#         self.deconv1 = de_conv2x(125,7)
        self.deconv1 = de_conv2x(256,7)
        
    def forward(self, x):
        results = []
        xx=x
        for ii,model in enumerate(self.base_model):
            xx = model(xx)
#             print(ii, xx.shape)
            if ii in {3,8,15,26,35}:
                results.append(xx)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3','relu5_5'])
        vgg_out=vgg_outputs(*results)
        fx0=vgg_out[0]
        fx1=vgg_out[1]
        fx2=vgg_out[2]
        fx3=vgg_out[3]
        fx4=vgg_out[4]
        
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        
        x0 = torch.cat([x, mesh1], 1)
        x = torch.cat([x0, fx0], 1)
        x = self.mp(self.conv0(x))
        
        x = torch.cat([x, fx1], 1)
        x = self.mp(self.conv1(x))   
        
        x = torch.cat([x, fx2], 1)
        x = self.mp(self.conv2(x)) 
        
        upfx4=self.up2x(fx4)
#         upfx4=self.deconvF4 (fx4)
        
        x = torch.cat([x, fx3 , upfx4], 1)
        x = self.mp(self.conv3(x))      
        
#         x = self.deconv0(x)
        x = self.up2x(x)
        x = self.deconv1(x)      
        
        
        return x
        

                
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                
model=Vgg16().to(device)
print(model)

# img = torch.rand([1,3,896,1120]).to(device)
# img=img.to(device)
# output = model1(img)
# print('output -->', output.shape)

# import gc 
# del model1
# del img 
# gc.collect()
# gc.collect()
# gc.collect()


# %% [code]

# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# img,mask,regrs,name=next(iter(train_loader))
# img.shape
# img=img.to(device)
# model=Vgg16().to(device)
# model.eval()
# output = model(img)
# print('output-size: ',output.shape)



# %% [code]
# !nvidia-smi -l 1

#  !pip install gpustat
# !gpustat

# !watch nvidia-smi

# !pip install  wandb


# !nvidia-settings -q all 



# %% [code]
# import gc 
# del model1
# del img 
# gc.collect()
# gc.collect()
# gc.collect()

# gc.collect()
# gc.collect()
# gc.collect()

# %% [code]


# %% [code]

# in_factor=3
# mask_factor = 12

# image_size_= np.array([2688,3360])
# input_size_=list(image_size_/in_factor)
# mask_size_=list(image_size_/mask_factor)


# print('Orginal image size: {} \nThe Input size : {} \nThe mask size: {}'.format(image_size_ ,input_size_ ,mask_size_))

# %% [code]
df_train, df_dev = train_test_split(train, test_size=0.1, random_state=63)
df_test = test
# Create dataset objects

train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

train_dataset = Baidu_Autonomous_dataset(df_train, train_images_dir)
dev_dataset = Baidu_Autonomous_dataset(df_dev, train_images_dir)
test_dataset = Baidu_Autonomous_dataset(df_test, test_images_dir)
# print(df_train.values[0])

print('Train dataset size:',len(train_dataset))
print('Validation dataset size:',len(dev_dataset))
print('Test dataset size:',len(test_dataset))
print()
print('Batch size: ',BATCH_SIZE)
print('Epoch number: ',n_epochs)
print('Device:',device)
print()

# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %% [code]
dataloaders_dict = {'train': train_loader ,'val': dev_loader}
print("Initializing Datasets and Dataloaders...")

# %% [code]
# import math
# # def lcm(a, b):
# #     return abs(a*b) // math.gcd(a, b)
# # lcm(24, 30)

# # k=4
# # for i in range(0 , 100): 
# #     gcdd=math.gcd(2710-i, (3384-k))
# #     if(gcdd> 1):
# #         print(i,'-->',gcdd, '.............',(2710-i)/gcdd, '  ' , (3384-k)/gcdd)
        
# # k=4
# a=80*32
# b=105*32

# print(np.array([a,b]))


# for i in range(0 , 100 ,32): 
# #     gcdd=math.gcd(2710-i, (3384-k))
#     gcdd=math.gcd(a+i, (b))
#     if(gcdd> 1):
# #         print(i,'-->',gcdd, '.............',(2710-i)/gcdd, '  ' , (3384-k)/gcdd)   
#         print(i,'  ',a+i,'-->',gcdd, '.............',(a+i)/gcdd, '  ' , (b)/gcdd) 
        


# %% [code]
# # aa=sym.solve([h2,w2],(p2,p4))
# # aa[p4]==48
# # # aa

# for i in range (110):
#     print(i,'-',i*32)

# %% [code]
# whos
# gggggggggg

# %% [code]
def criterion(prediction, mask, regr, size_average=True):

    # Binary mask loss
#     regr=regr.permute(0,3, 2, 1)
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



# %% [code]
# model.train()
# img_batch, mask_batch, regr_batch, _ =next(iter(train_loader))
# img_batch = img_batch.to(device)
# mask_batch = mask_batch.to(device)
# regr_batch = regr_batch.to(device)
# output = model(img_batch)
# print('out:' ,output.shape, '  In:',img_batch.shape)
# loss = criterion(output, mask_batch, regr_batch)

# %% [code]
# prediction=output
# mask=mask_batch
# regr=regr_batch
# pred_mask = torch.sigmoid(prediction[:, 0])
# # mask_loss = mask * torch.log(pred_mask + 1e-12) 
# mask.shape

# %% [code]
def train_model(epoch, history=None):
    ep_since = time.time()
    
    model.train()
    running_loss = 0.0

    for batch_idx, (img_batch, mask_batch, regr_batch, _) in enumerate(train_loader):
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
        
    ep_time_elapsed = time.time() - ep_since
    epoch_loss = running_loss / len(train_loader.dataset)
    print('Train Epoch: {} \tLearning Rate: {:.6f}\tLoss: {:.6f} \t Loss-Mean:{:.6f} \tTime:{:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data,epoch_loss,ep_time_elapsed))
    
#         ep_time_elapsed = time.time() - ep_since
#     epoch_loss = running_loss / len(train_loader[phase].dataset)
    
#     print('Train Epoch: {} \tLearning Rate: {:.6f}\tLoss: {:.6f} \t Loss-Mean: {} \tTime:{:.6f}'.format(
#         epoch,
#         optimizer.state_dict()['param_groups'][0]['lr'],
#         loss.data,ep_time_elapsed ,epoch_loss)


def evaluate_model(epoch, history=None):
    since = time.time()
    model.eval()
    loss = 0
    
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch, name in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss_batch = criterion(output, mask_batch, regr_batch, size_average=False).data
            time_elapsed = time.time() - since
            print('Evaluate Epoch: image_id{} \tLoss: {:.6f} \tTime:{:.6f}'.format(name,epoch_loss,ep_time_elapsed))
            loss += loss_batch
            
    
    loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    print('Dev loss: {:.4f}'.format(loss))

# %% [code]
# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

# %% [code]
history = pd.DataFrame()
# model.load_state_dict(torch.load('./vgg19_ep_18'))  
# model.load_state_dict(torch.load('./')) 

# with open('history_vgg19_ep_18', 'rb') as handle:
#     b = pickle.load(handle) 

import gc
save_dir= './vgg19_j9_12_21'

history = pd.DataFrame()

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_model(epoch, history)
    evaluate_model(epoch, history)
 ##   if(epoch % 5 ==0):
    torch.save(model.state_dict(), os.path.join(save_dir, '{}_ep_{}.model'.format(model_name,epoch)))
       
    with open(os.path.join(save_dir, 'history_{}.pkl'.format(model_name,epoch)), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)