import numpy as np
import os
import SimpleITK as sitk
import random
import pandas as pd
from scipy import ndimage
from os.path import join
from sklearn import preprocessing
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

class LITS_preprocess:
    def __init__(self, fixed_dataset_path):
  
        self.fixed_path    = fixed_dataset_path
        self.valid_rate    = 0.0

    def write_train_val_name_list(self):
        os.makedirs(join(self.fixed_path, "data"), exist_ok=True)
        os.makedirs(join(self.fixed_path, "label"), exist_ok=True)
        
        data_name_list = os.listdir(join(self.fixed_path, "data"))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        # random.shuffle(data_name_list)

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num*(1-self.valid_rate))]
        val_name_list = data_name_list[int(data_num*(1-self.valid_rate)):int(data_num*((1-self.valid_rate) + self.valid_rate))]
        self.write_name_list(train_name_list, "train_path_list1.txt")
        self.write_name_list(val_name_list, "val_path_list.txt1")


    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            # print(name, name.split("_")[0])
            ct_path = os.path.join(self.fixed_path, 'data', name)
            seg_path = os.path.join(self.fixed_path, 'label', name).replace('_data', '')
            res_path = os.path.join(self.fixed_path, 'res', 'res6.xlsx')
            pump_path = os.path.join(self.fixed_path, 'pump', f'{name.split("_")[0]}_{name.split("_")[1]}.xlsx')
            f.write(ct_path + ' ' + seg_path + ' ' + res_path + ' ' + pump_path + "\n")
        f.close()


if __name__ == '__main__':

    fixed_dataset_path = 'fixed_lits'

    tool = LITS_preprocess(fixed_dataset_path) 
    tool.write_train_val_name_list()      # 创建索引txt文件