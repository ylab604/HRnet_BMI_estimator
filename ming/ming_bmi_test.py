import os
import numpy as np
import pandas as pd

# ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", self.data[i])
# BMI_ = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2

data = os.listdir('dataset/BMI/Image_test_remove')
error_list = []
new_bmi = []
for i in range(len(data)) :
    try:
        list_data_name=data[i].split('_')
        height = list_data_name[-2]
        weight = list_data_name[-1][:-8]
        bmi = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
    except:
        error_list.append(data[i])
        data[i] = data[i].replace(' (2)','_')
        data[i] = data[i].replace(']', '_')
        list_data_name = data[i].split('_')
        height = list_data_name[-3]
        weight = list_data_name[-2]
        bmi = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
        print(bmi)




for j in range(len(error_list)) :
    data[error_list[j]] = data[error_list[j]].replace(']','_')
    list_data_name = data[error_list[j]].split('_')
    height = list_data_name[-3]
    weight = list_data_name[-2]
    print ((int(weight) / 100000) / ((int(height) / 100000) ** 2))

name_list = data[0].split('_')
height = int(name_list[-2]) / 100000
weight = int(name_list[-1][:-8]) / 100000
bmi = weight / (height**2)


