import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

csv_ = pd.read_csv('dataset/BMI/Image_test.csv', header = None)
sns.distplot(csv_[1])

file_name=os.listdir('dataset/BMI/Image_test/')

BMI_list = []

for i in range(len(file_name)) :
    ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", file_name[i])
    BMI_ = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
    BMI_list.append(float(BMI_))

sns.distplot(BMI_list)