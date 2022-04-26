import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

data_root = "dataset/BMI/Image_train_remove_aug_again"
data_aug = os.listdir(data_root)

BMI_list= []

for i in range(len(data_aug)):
    try:
        name_list = data_aug[i].split('_')
        height = name_list[-2]
        weight = name_list[-1][:-8]
        BMI_ = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
        BMI_list.append(BMI_)
        # if BMI_ > 25 and BMI_<35 :
        #     if ('orgin' in name_list) or ('fliped' in name_list):
        #         print(name_list)

    except:
        data_aug[i]=data_aug[i].replace(']','_')
        name_list = data_aug[i].split('_')
        height = name_list[-3]
        weight = name_list[-2]
        BMI_ = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
        BMI_list.append(BMI_)
        # if BMI_ > 25 and BMI_<35 :
        #     if ('orgin' in name_list) or ('fliped' in name_list):
        #         print(name_list)

x = range(0,len(BMI_list))
plt.hist(BMI_list)

























#data_aug