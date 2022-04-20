from PIL import Image
import os
import glob
import random
#플립하고, 로테이션(15도) 따라서, 원 데이터에서 4배
#글로브로 다부르기
#C:\Users\user\Downloads\datasets\datasets\Image_test_rmove
###############################################################################
data_root = "C:\\Users\\user\\Downloads\\Image_train_remove"
###############################################################################
#glob.glob(os.path.join(data_root,"*")
data = glob.glob(os.path.join(data_root,"*"))

def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

#isExist_flip = os.path.exists(os.path.join(data_root,"flip"))
#isExist_rotate = os.path.exists(os.path.join(data_root,"rotate"))

# if not isExist_flip:
#     os.mkdir(os.path.join(data_root,"flip"))

# if not isExist_rotate:
#     os.mkdir(os.path.join(data_root,"rotate"))

for i in range(len(data)):
    im = Image.open(data[i])
    # flip
    filename = data[i].split("\\")[-1]
    print(filename)
    hori_flippedImage = im.transpose(Image.FLIP_LEFT_RIGHT)
    # flip한거 저장 "\\flip\\"+
    hori_flippedImage.save(data_root+"\\fliped_"+filename)
    # 패딩 추가
    im_new_flip = add_margin(hori_flippedImage, 10, 100, 10, 100, (0, 0, 0))
    #flip 로테이트, 로테이트 한거 저장 "\\rotate\\"+
    angle = random.uniform(15,-15)
       
    out_flip = im_new_flip.rotate(angle)
    out_flip.save(data_root+"\\fliped_rotate_"+filename)
    #orgin 로테이트, 로테이트 한거 저장 "\\rotate\\"+
    im_new_orgin = add_margin(im,10, 100, 10, 100, (0, 0, 0))
    out_orgin = im_new_orgin.rotate(angle)
    out_orgin.save(data_root+"\\orgin_rotate_"+filename)
######
#again match imbalance
######
import os
import glob
data_root = "C:\\Users\\user\\Downloads\\Image_train_remove_aug"
data_aug = os.listdir(data_root)

for i in range(len(data_aug)):
    try:
        name_list = data_aug[i].split('_')
        height = name_list[-2]
        weight = name_list[-1][:-8]
        BMI_ = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
        if BMI_ > 25 and BMI_ < 35 :
            if "orgin" or "fliped" in name_list:
                if os.path.exists(data_root):
                    os.remove(os.path.join(data_root, data_aug[i]))

    except:
        data_aug[i]=data_aug[i].replace(']','_')
        name_list = data_aug[i].split('_')
        height = name_list[-3]
        weight = name_list[-2]
        BMI_ = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
        if BMI_ > 25 and BMI_ < 35 :
            if "orgin" or "fliped" in name_list:
                if os.path.exists(data_root):
                    os.remove(os.path.join(data_root, data_aug[i]))

#data_aug       
