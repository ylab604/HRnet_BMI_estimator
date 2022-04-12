from sklearn.preprocessing import MinMaxScaler
import os
import re
import pandas as pd
import numpy as np
from pickle import dump, load

mmscale = MinMaxScaler()

file_name = os.listdir('dataset/BMI/Image_train')
BMI_list = []
for i in range(len(file_name)) :
    ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", file_name[i])
    BMI_ = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
    BMI_list.append(BMI_)

BMI_list = np.array(BMI_list)
BMI_list = BMI_list.reshape(len(BMI_list), -1)
scaled_BMI=mmscale.fit_transform(BMI_list)

# back = mmscale.inverse_transform(scaled_BMI)

# dump(mmscale, open('dataset/BMI/minmaxscaler_bmi.pkl','wb'))

# load_minmax=load(open('dataset/BMI/minmaxscaler_bmi.pkl','rb'))
# load_m=load_minmax.transform(BMI_list)
