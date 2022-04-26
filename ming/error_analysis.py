import numpy as np
import pandas as pd

df = pd.read_excel('/home/ylab3/HRnet_BMI_estimator/only_image/HRnet_48_aug_5O_Predict_onlyimage.xlsx')
df= df.drop(['Unnamed: 0'], axis=1)

df= df.sample(frac=1).reset_index(drop=True)

df['GT'] = df['GT'].astype(float)
df['Predict'] = df['Predict'].astype(float)

name_ = ['underweight','normal','overweight','obesity']
total = 0

for i in range(len(name_)) :
    l1_mean = np.mean(df[df['category']==name_[i]]['abs_l1'])
    GT_mean = np.mean(df[df['category'] == name_[i]]['GT'])
    n = len(df[df['category'] == name_[i]])
    MAPE=l1_mean/GT_mean * 100

    # total += MAPE * n
    print(name_[i] +' l1_mean : ' + str(l1_mean))
    print(name_[i] + ' GT_mean : ' + str(GT_mean))
    print(name_[i] + ' MAPE : ' + str(MAPE))

# df = df.loc[:int(len(df)*0.75)]
mape = np.sum(abs(df['Predict'] - df['GT'])/df['GT']) * 100 / len(df)
print()
print(mape)

# print(total / len(df))



# np.mean(df[df['category']==name_[i]]['abs_l1'])

