import pandas as pd
import numpy as np
import os
import re

train_csv = pd.read_csv('dataset/BMI/Image_train.csv', header=None)
print(len(train_csv)) # 2064
print(len(os.listdir('dataset/BMI/Image_train')))  # 2349
print(len(os.listdir('dataset/BMI/Image_val'))) # 586

test_csv = pd.read_csv('dataset/BMI/Image_test.csv', header=None)
print(len(test_csv)) # 890
print(len(os.listdir('dataset/BMI/Image_test'))) #1254

#train_csv에 있는게, os.listdir에 없다. -> 이미지가 없다!, annoatation 있다 -> 398
train_ddiong = []
for i in range(len(train_csv)) :
    if train_csv[0][i] not in os.listdir('dataset/BMI/Image_train'):
        train_ddiong.append(train_csv[0][i])

# validation에 이미지가 있는 리스트 -> 398
# validation image랑 train_image 합쳐야 함
train_in_vali = []
for i in range(len(train_ddiong)) :
    if train_ddiong[i] in os.listdir('dataset/BMI/Image_val') :
        train_in_vali.append(train_ddiong[i])

#os.listdir에 있는데, train_csv에 없다. -> 이미지만 있다, annotation 없다 -> 683
train_1_ddiong = []
train_image = os.listdir('dataset/BMI/Image_train')
train_csv_name = list(train_csv[0])
for i in range(len(train_image)) :
    if train_image[i] not in train_csv_name:
        train_1_ddiong.append(train_image[i])

#test_csv에 있는거 다 이미지 있음
test_no_image = []
for i in range(len(test_csv)) :
    if test_csv[0][i] not in os.listdir('dataset/BMI/Image_test') :
        test_no_image.append(test_csv[0][i])

ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", '0_F_30_157480_7030682.jpg')
########################################################################
BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
##########################################################################

train_BMI = []

for i in range(len(train_image)) :
    try :
        ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", train_image[i])
        BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
        train_BMI.append(BMI)
    except :
        print(train_image[i])
#wow
#  '1_F_21_167640_7620352.jpg'
#  ????????????????????????????????
# 29.859475352559524