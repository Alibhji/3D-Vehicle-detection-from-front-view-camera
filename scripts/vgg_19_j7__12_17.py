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



BATCH_SIZE = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 20
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print(os.environ["CUDA_VISIBLE_DEVICES"])

# %% [code]
PATH = '../data/'
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float)

camera_matrix_inv = np.linalg.inv(camera_matrix)

model_name = 'vgg19' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

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


def AddMargin(img,top =0 , bottom=0 , left=0 ,right =0):
#     borderType = cv2.BORDER_CONSTANT
    borderType = cv2.BORDER_REPLICATE
    value = [255, 255, 255]
#     top = int(0.05 * img.shape[0])  # shape[0] = rows
#     bottom = top
#     left = int(0.05 * img.shape[1])  # shape[1] = cols
#     right = left
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)
    return img

def get_pd_of_all_label(str_data, diff=0):
#         _3d_points= np.array(train['PredictionString'][img_ID].split(' ')).astype(np.float32).reshape([-1,7])
#         _3d_points= np.array(str_data.split(' ')).astype(np.float32).reshape([-1,7])
#        data=[float(i) for i in str_data.split(' ')]
        _3d_points=np.array([i for i in str_data.split(' ')]).astype(np.float).reshape([-1,7])
       
#        _3d_points = [data[i:i+7] for i in range(0, len(data),7)]
        
        _3d_points_pd =pd.DataFrame( {'id':_3d_points[:,0]} )
        _3d_points_pd['yaw']= _3d_points[:,1]
        _3d_points_pd['pitch']= _3d_points[:,2]
        _3d_points_pd['roll']= _3d_points[:,3]
        _3d_points_pd['x']= _3d_points[:,4]
        _3d_points_pd['y']= _3d_points[:,5]
        _3d_points_pd['z']= _3d_points[:,6]
        _3d_points_pd['xs']=get_img_coords(str_data)[0] 
        _3d_points_pd['ys']=get_img_coords(str_data)[1] + diff
        return _3d_points_pd
    
    
    

# %% [code]
# a=[i for i in range(16)]
# [a[i:i+4] for i in range(0, len(a),4)]

# %% [code]

# get_pd_of_all_label(df_train.values[0][1],100)
# dic_test=get_dict_of_all_label(df_train.values[0][1],10000)
# add_scaled_label(dic_test , 'xs' , 'xs_l',10)

# %% [code]
class CarDataset(Dataset):

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training
        self.x_pixel_cut=454
        self.top_padding=564*2
        self.image_resize=(900,900)
        self.mask_size=(224,224)
        self.diff =  self.top_padding - self.x_pixel_cut
        self.mark_objects=False
        self.pre_mdl_meanStd= (model.mean, model.std)
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name, labels = self.df.values[idx]
#         input_img = load_img(self.root_dir.format(img_name)).transpose(Image.ROTATE_90)
#         input_img = Image.open(self.root_dir.format(img_name))
#         print('image ID:',img_name )


        input_img = cv2.imread(self.root_dir.format(img_name))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#         input_img = cv2.normalize(input_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#         input_img = input_img.astype(np.uint8)
    
        input_img = input_img[self.x_pixel_cut:, 0:]
        input_img = AddMargin(input_img,top=self.top_padding)
        

        orginal_image_size=input_img.shape
        
        
        label_pd=get_pd_of_all_label(labels, self.diff)
        
#         #plot on the orginal image
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         for i in range(len(label_pd)):
#             xs=int(label_pd['xs'][i])
#             ys=int(label_pd['ys'][i])
#             if(xs < orginal_image_size[0]  and ys<orginal_image_size[1]):
#                 input_img=cv2.putText(input_img, str(i), (xs,ys-50), font, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
#                 input_img=cv2.circle(input_img, (xs,ys), 10, (0, 255, 0), 10)
                
                
        input_img = cv2.resize(input_img,self.image_resize)
        
                # calculate the scales
        
#         print('orginal_image_size', orginal_image_size)
        new_image_size=(input_img.shape[:2]) 
        
        
        image_scale= orginal_image_size[0] / new_image_size[0]
#         print('image_scale' ,image_scale)
        
#         print('new_image_size',new_image_size)
        label_pd['xs_re'] = label_pd.apply(lambda row : int(row['xs']/image_scale), axis = 1)
        label_pd['ys_re'] = label_pd.apply(lambda row : int(row['ys']/image_scale), axis = 1) 
        
        if(self.mark_objects):
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(label_pd)):
                xs=int(label_pd['xs_re'][i])
                ys=int(label_pd['ys_re'][i])
                if(xs < new_image_size[0]  and ys<new_image_size[1]):
                    input_img=cv2.putText(input_img, str(i), (xs,ys-5), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    input_img=cv2.circle(input_img, (xs,ys), 3, (255, 0, 0), 3)
                
        label_scale= orginal_image_size[0] / self.mask_size[0]
#         print('label_scale' ,label_scale)
        label_pd['xs_la'] = label_pd.apply(lambda row : int(row['xs']/label_scale), axis = 1)
        label_pd['ys_la'] = label_pd.apply(lambda row : int(row['ys']/label_scale), axis = 1)
        
        regres=np.zeros([7,self.mask_size[0],self.mask_size[1]],dtype='float32')
        
        for i in range(len(label_pd)):
            xs=int(label_pd['xs_la'][i])
            ys=int(label_pd['ys_la'][i])
            if(xs < self.mask_size[0]  and ys< self.mask_size[1]):
                for ind, lbl in enumerate(['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
                    regres[ind,ys,xs] = label_pd[lbl][i]
                if(self.mark_objects):
                    input_img=cv2.putText(input_img, str(i), (xs,ys-5), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    input_img=cv2.circle(input_img, (xs,ys), 3, (255, 0, 0), 3)

        mask=regres[0,:,:]
        mask[mask>0]= 1
        # change regres shape to 244,244,6
        regres=np.moveaxis(regres, 0, -1) 
    
        tf=[]
#         tf.append(transforms.ToPILImage())
#         tf.append(transforms.CenterCrop(400))
    
#         tf.append(transforms.ToPILImage())
#         tf.append(transforms.Resize((900,900)))
        tf.append(transforms.ToTensor())
        tf.append(transforms.Normalize(self.pre_mdl_meanStd[0],self.pre_mdl_meanStd[1]))
        
        transformations = transforms.Compose(tf)
        input_img = transformations(input_img)
        
        return [input_img , mask ,regres ]

# %% [code]
df_train, df_dev = train_test_split(train, test_size=0.1, random_state=63)
df_test = test
# Create dataset objects

train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

train_dataset = CarDataset(df_train, train_images_dir)
dev_dataset = CarDataset(df_dev, train_images_dir)
test_dataset = CarDataset(df_test, test_images_dir)
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
# it = iter(dataloaders_dict['train'])
# imgs , masks, regs  = next(it)
# masks

# # plt.imshow(masks)



# %% [code]
# for i,v in enumerate(dataloaders_dict['train']):
#     print(i)

# %% [code]
# train_images_dir = os.path.join(PATH + 'train_images/{}.jpg')
# train_dataset = CarDataset(train,train_images_dir)
# # type(train_dataset[0][0])
# img1=train_dataset[5][0].numpy()[...].swapaxes(0,2).swapaxes(0,1)
# # img1=train_dataset[0][0]
# print(img1.shape)
# plt.figure(figsize=(16,16))
# plt.imshow(img1)
# img1.shape
# # model.mean
# # img1
# print(train_dataset[0][2].shape)

# %% [code]
# img_num=2

# plt.figure(figsize=(10,10))
# mask=train_dataset[img_num][2][:,:,1]
# # mask=np.moveaxis(mask, 0, -1)[:,:,1]
# print(mask.shape)
# mask[mask>0]= 1

# # mask=train_dataset[img_num][1]

# plt.imshow(mask)
# img2=train_dataset[img_num][0].numpy().swapaxes(0,2).swapaxes(0,1)
# img2 = cv2.resize(img2,(224,224))
# plt.imshow(img2, alpha=0.2)
# # plt.imshow(Image2_mask, cmap='jet', alpha=0.5)



# %% [code]
# Define Model structure

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

class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
#         self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model=pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        
        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        
        self.mp = nn.MaxPool2d(2)
        
        self.up1 = up(514 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        
#         x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
#         feats = self.base_model.extract_features(x)
        feats = self.base_model._features(x)
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
    
#         bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
#         feats = torch.cat([bg, feats, bg], 3)
        
#         # Add positional info
        
        
        
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.up3(x)
        x = self.outc(x)
        return x

# %% [code]
def criterion(prediction, mask, regr, size_average=True):

    # Binary mask loss
    regr=regr.permute(0,3, 2, 1)
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
def train_model(epoch, history=None):
    ep_since = time.time()
    
    model.train()
    running_loss = 0.0

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(train_loader):
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

# %% [code]
model = MyUNet(8).to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

# %% [code]
history = pd.DataFrame()
# model.load_state_dict(torch.load('./vgg19_ep_18'))  
# model.load_state_dict(torch.load('./')) 

# with open('history_vgg19_ep_18', 'rb') as handle:
#     b = pickle.load(handle) 

import gc
save_dir= './vgg19_j7_12_17'

history = pd.DataFrame()

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_model(epoch, history)
    evaluate_model(epoch, history)
    
    torch.save(model.state_dict(), os.path.join(save_dir, '{}_ep_{}.model'.format(model_name,epoch)))
    
    with open(os.path.join(save_dir, 'history_{}.pkl'.format(model_name,epoch)), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)