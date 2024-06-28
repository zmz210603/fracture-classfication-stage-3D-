#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
from PIL import Image

from unet import Unet

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list 


if __name__ == "__main__":

    #-------------------------------------------------------------------------#
    unet = Unet()
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #----------------------------------------------------------------------------------------------------------#
    count           = False
    name_classes    = ["_background_","frac"]
    #----------------------------------------------------------------------------------------------------------#
    
    Path  = 'fixed_lits'
    colors = [(0,0,0),(0,0,255),(0,0,255),(210,105,30),(255,0,0),(255,0,0)]
    
    if mode == "predict":

        while True:
            txt           = input('Input image filename:')
            # num           = input('num:')
            for num in range(192):
                filename_list = load_file_name_list(os.path.join(Path, txt))[int(num)]
                # print(filename_list)
                filename      = filename_list[0]
                
                # 获取bmp文件名
                file_name_with_extension = os.path.basename(filename)
                file_name = os.path.splitext(file_name_with_extension)[0]
                # print(file_name)
                # try:
                import re
                # 使用正则表达式提取文件名中的数字
                digits = re.findall(r'\d+', filename)[-3:]

                # 创建数字对应的数组
                arrays = []
                for digit in digits:
                    digit_array = np.ones((512, 512)) * int(digit)
                    arrays.append(digit_array)
                
                stage_time = np.array(arrays)[np.newaxis, ...]
                
                jpg = Image.open(filename_list[0])
                res = pd.read_excel(filename_list[2], header=0)
                res = res.values
                layer = int(digits[1])
                res_terson = torch.from_numpy(res[layer-1:layer])
                # scalar     = preprocessing.MinMaxScaler(feature_range=(0,1))
                # res_terson = scalar.fit_transform(res_terson)
                res_values = np.expand_dims(res_terson, -1)
                res_values = np.repeat(res_values, 20, axis=1)
                res_values = np.repeat(res_values, 200, axis=2)
                res_values = np.expand_dims(res_values, 0)

                pum = pd.read_excel(filename_list[3], index_col= None, header=0)
                pump_value = pum.values
                df_terson = torch.from_numpy(pump_value)
                pad_size  = (16-df_terson.shape[0], 9-df_terson.shape[1])
                df_terson = F.pad(torch.Tensor(df_terson.float()), (0, pad_size[1], 0, pad_size[0]), mode='constant', value=0)
                pump      = df_terson[np.newaxis, ...]
                pump      = np.repeat(pump, 10, axis=1)
                pump      = np.repeat(pump, 20, axis=2)
                pump      = pump[np.newaxis, ...]
                        
                r_image = unet.detect_image(jpg, stage_time, res_values, pump)                    
                r_image.save(f'result/{file_name}.png')