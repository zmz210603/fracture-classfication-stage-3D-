from matplotlib import pyplot as plt
import torch

import numpy as np
import pandas as pd

from sklearn import preprocessing
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import cv2

import os


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


colors = [(0,0,0),(0,0,255),(0,0,255),(255,0,0)]

def preprocess_input(image):
    image /= 1.0
    return image


class UnetDataset(Dataset):
    def __init__(self, dataset_path, txt, input_shape):
        
        self.input_shape = input_shape
        self.filename_list = self.load_file_name_list(os.path.join(dataset_path, txt))

    def __getitem__(self, index):
        

        #-------------------------------------------------------#
        #         layer_stage_time                              #
        #-------------------------------------------------------#
        import re
        filename = self.filename_list[index][0]
        digits = re.findall(r'\d+', filename)[-3:]
        # print(digits)
        arrays = []
        for digit in digits:
            digit_array = np.ones(self.input_shape) * int(digit)
            arrays.append(digit_array)
        
        layer_stage_time = np.array(arrays)

        #-------------------------------------------------------#
        #          图片                                         #
        #-------------------------------------------------------#
        jpg = Image.open(self.filename_list[index][0])
        png = Image.open(self.filename_list[index][1])

        jpg, png = self.get_random_data(jpg, png, self.input_shape, random = False)
        # jpg      = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        jpg = np.array(jpg, np.float64)
        jpg[jpg <10] = 0
        jpg[jpg > 0] = 10

        jpg          = preprocess_input(jpg)[np.newaxis, ...]
        png          = np.array(png)
        png[png <= 13] = 0
        png[png >   0] = 1
        seg_labels   = np.eye(2 + 1)[png.reshape([-1])]
        seg_labels   = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), 2 + 1))
        # print(jpg.shape, png.shape,seg_labels.shape)
        
        #-------------------------------------------------------#
        #          res                                          #
        #-------------------------------------------------------#
        res = pd.read_excel(self.filename_list[index][2], header=0)
        res = res.values
        layer = int(digits[1])
        res_terson = torch.from_numpy(res[layer-1:layer])
        # scalar     = preprocessing.MinMaxScaler(feature_range=(0,1))
        # res_terson = scalar.fit_transform(res_terson)
        res_values = np.expand_dims(res_terson, -1)
        res_values = np.repeat(res_values, 20, axis=1)
        res_values = np.repeat(res_values, 200, axis=2)
        # print(res_values.shape)

        #-------------------------------------------------------#
        #          pump                                         #
        #-------------------------------------------------------#
        pum = pd.read_excel(self.filename_list[index][3], index_col= None, header=0)
        pump_value = pum.values
        df_terson = torch.from_numpy(pump_value)
        pad_size  = (16-df_terson.shape[0], 9-df_terson.shape[1])
        df_terson = F.pad(torch.Tensor(df_terson.float()), (0, pad_size[1], 0, pad_size[0]), mode='constant', value=0)
        pump      = df_terson[np.newaxis, ...]
        pump      = np.repeat(pump, 10, axis=1)
        pump      = np.repeat(pump, 20, axis=2)
        # print(pump.shape)
        
        return jpg, layer_stage_time, png, seg_labels, res_values, pump


    def __len__(self):
        
        return len(self.filename_list)
    

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list    

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, images, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=False):
        image   = cvtColor(images)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        h, w    = input_shape
        iw, ih  = image.size
        
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image       = image.resize((nw,nh), Image.Resampling.BICUBIC)
        new_image   = Image.new('L', [w, h], (0))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))

        label       = label.resize((nw,nh), Image.Resampling.BICUBIC)
        new_label   = Image.new('L', [w, h], (0))
        new_label.paste(label, ((w-nw)//2, (h-nh)//2))
        return new_image, new_label

 

#---------------------------#
#   读取数据集对应的txt
#---------------------------#
# Path  = 'fixed_lits'
# input_shape = [320, 320]

# val_dataset     = UnetDataset(Path, 'val_path_list copy.txt', input_shape)
# gen             = DataLoader(val_dataset, batch_size = 1, num_workers = 0, pin_memory=True,
#                                             drop_last = True)

# for iteration, batch in enumerate(gen):
#     imgs, stage_time, labels, f1_label, res, pumpdata = batch 
   
    # outputs = model_train(imgs, pumpdata, res)    
#     print(imgs.shape, labels.shape)

    # img = labels.permute(1,2,0)
    # print(img.shape)
    # ARRS_1 = []
    # f = open( f'A1.txt','w+')                   # 'w+'  ?????д
    # for i in range(len(img)):
    #     jointsFrame = img[i]                    # ???
    #     ARRS_1.append(jointsFrame)
    #     for Ji in range(len(img[i])):           #
    #         strNum = str(jointsFrame[Ji])
    #         f.write(strNum)       
    #         f.write(' ')
    #     f.write('\n')
    # f.close()  
        
#         print(imgs.shape)
#         print(pngs.shape)
#         print(labels.shape)