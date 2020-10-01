import os
import numpy as np
from PIL import Image
import pandas as pd

img_size = 224

def data_arg_9(WW, HH, img, train_labels, path_new, index):
    crop_num = 1

    label = []
    train_name = []

    # cheng bi li
    p_center = [int(WW / 2), int(HH / 2)]
    p = [p_center]

    scale1_WW = int(4. / 5 * WW)  # scale 4/5, 3/5, 2/5, 1/5
    scale1_HH = int(4. / 5 * HH)
    scale2_WW = int(3. / 5 * WW)
    scale2_HH = int(3. / 5 * HH)
    scale3_WW = int(2. / 5 * WW)
    scale3_HH = int(2. / 5 * HH)
    scale4_WW = int(1. / 5 * WW)
    scale4_HH = int(1. / 5 * HH)
    scale_WW = [scale1_WW, scale2_WW, scale3_WW, scale4_WW]
    scale_HH = [scale1_HH, scale2_HH, scale3_HH, scale4_HH]
    for i in range(1):  # 1 point
        for j in range(4):  # 4 scale
            point = p[i]
            scale_j_WW = scale_WW[j]
            scale_j_HH = scale_HH[j]
            rectangle = (point[0] - scale_j_WW / 2, point[1] - scale_j_HH / 2, point[0] + scale_j_WW / 2, point[1] + scale_j_HH / 2)
            img_i_j = img.crop(rectangle)
            img_i_j_re = img_i_j.resize((img_size, img_size))
            label.append(train_labels)
            train_name.append(index[:-4] + '_9_' + str(crop_num) + '.png')
            img_i_j_re.save(path_new + index[:-4] + '_9_' + str(crop_num) + '.png')
            crop_num = crop_num + 1

    # NO cheng bi li
    WH = min(WW, HH)
    p_center = [int(WW / 2), int(HH / 2)]
    p = [p_center]

    scale1_WH = int(4. / 5 * WH)  # scale 4/5, 3/5, 2/5, 1/5
    scale2_WH = int(3. / 5 * WH)
    scale3_WH = int(2. / 5 * WH)
    scale4_WH = int(1. / 5 * WH)
    scale_WH = [scale1_WH, scale2_WH, scale3_WH, scale4_WH]
    for i in range(1):  # 1 point
        for j in range(4):  # 4 scale
            point = p[i]
            scale_j_WH = scale_WH[j]
            rectangle = (point[0] - scale_j_WH / 2, point[1] - scale_j_WH / 2, point[0] + scale_j_WH / 2,
                            point[1] + scale_j_WH / 2)
            img_i_j = img.crop(rectangle)
            img_i_j_re = img_i_j.resize((img_size, img_size))
            label.append(train_labels)
            train_name.append(index[:-4] + '_9_' + str(crop_num) + '.png')
            img_i_j_re.save(path_new + index[:-4] + '_9_' + str(crop_num) + '.png')
            crop_num = crop_num + 1

    return train_name, label


def data_arg_15(WW, HH, img, train_labels, path_new, index):
    crop_num = 1

    label = []
    train_name = []

    # cheng bi li
    p_center = [int(WW / 2), int(HH / 2)]
    p = [p_center]

    # scale 4/5, 3/5, 2/5, 1/5
    # scale 4/5, 3.5/5, 3.0/5, 2.5/5, 2.0/5, 1.5/5, 1.0/5
    scale1_WW = int(4. / 5 * WW)
    scale1_HH = int(4. / 5 * HH)
    scale2_WW = int(3.5 / 5 * WW)
    scale2_HH = int(3.5 / 5 * HH)
    scale3_WW = int(3. / 5 * WW)
    scale3_HH = int(3. / 5 * HH)
    scale4_WW = int(2.5 / 5 * WW)
    scale4_HH = int(2.5 / 5 * HH)
    scale5_WW = int(2. / 5 * WW)
    scale5_HH = int(2. / 5 * HH)
    scale6_WW = int(1.5 / 5 * WW)
    scale6_HH = int(1.5 / 5 * HH)
    scale7_WW = int(1. / 5 * WW)
    scale7_HH = int(1. / 5 * HH)
    scale_WW = [scale1_WW, scale2_WW, scale3_WW, scale4_WW, scale5_WW, scale6_WW, scale7_WW]
    scale_HH = [scale1_HH, scale2_HH, scale3_HH, scale4_HH, scale5_HH, scale6_HH, scale7_HH]
    for i in range(1):  # 1 point
        for j in range(7):  # 7 scale
            point = p[i]
            scale_j_WW = scale_WW[j]
            scale_j_HH = scale_HH[j]
            rectangle = (point[0] - scale_j_WW / 2, point[1] - scale_j_HH / 2, point[0] + scale_j_WW / 2,
                         point[1] + scale_j_HH / 2)
            img_i_j = img.crop(rectangle)
            img_i_j_re = img_i_j.resize((img_size, img_size))
            label.append(train_labels)
            train_name.append(index[:-4] + '_15_' + str(crop_num) + '.png')
            img_i_j_re.save(path_new + index[:-4] + '_15_' + str(crop_num) + '.png')
            crop_num = crop_num + 1

    # NO cheng bi li
    WH = min(WW, HH)
    p_center = [int(WW / 2), int(HH / 2)]
    p = [p_center]

    # scale 4/5, 3/5, 2/5, 1/5
    # scale 4/5, 3.5/5, 3.0/5, 2.5/5, 2.0/5, 1.5/5, 1.0/5
    scale1_WH = int(4.0 / 5 * WH)
    scale2_WH = int(3.5 / 5 * WH)
    scale3_WH = int(3.0 / 5 * WH)
    scale4_WH = int(2.5 / 5 * WH)
    scale5_WH = int(2.0 / 5 * WH)
    scale6_WH = int(1.5 / 5 * WH)
    scale7_WH = int(1.0 / 5 * WH)

    scale_WH = [scale1_WH, scale2_WH, scale3_WH, scale4_WH, scale5_WH, scale6_WH, scale7_WH]
    for i in range(1):  # 1 point
        for j in range(7):  # 6 scale
            point = p[i]
            scale_j_WH = scale_WH[j]
            rectangle = (point[0] - scale_j_WH / 2, point[1] - scale_j_WH / 2, point[0] + scale_j_WH / 2,
                         point[1] + scale_j_WH / 2)
            img_i_j = img.crop(rectangle)
            img_i_j_re = img_i_j.resize((img_size, img_size))
            label.append(train_labels)
            train_name.append(index[:-4] + '_15_' + str(crop_num) + '.png')
            img_i_j_re.save(path_new + index[:-4] + '_15_' + str(crop_num) + '.png')
            crop_num = crop_num + 1

    return train_name, label

class_p = 'Validation'  # 'Validation'
num_resize = 224
data_labels = pd.read_csv('data/ISIC-2017_'+class_p+'_Part3_GroundTruth.csv')
labels_ori = np.stack([data_labels.nodiease, data_labels.melanoma, data_labels.seborrheic_keratosis],axis=-1)
labels_ori = np.argmax(labels_ori,axis=-1)
imagePathDir = os.listdir('data/ISIC-2017_'+class_p+'_Data/Images/')
imagePathDir.sort()
aug_num = 9

path_new = 'cls_data/'+class_p+'_resize_crop'+str(aug_num)+'_cls/'
if not os.path.isdir(path_new):
    os.makedirs(path_new)

num = 0
labels = []
train_names = []
for index in imagePathDir:
    print(num)
    # read img
    img = Image.open('data/ISIC-2017_'+class_p+'_Data/Images/'+index)
    img_re = img.resize((num_resize, num_resize))
    img_re = img_re.resize((num_resize, num_resize))
    labels.append(labels_ori[num])
    train_names.append(index[:-4] + '.png')
    img_re.save(path_new + index[:-4] + '.png')

    [WW, HH] = img.size

    train_name_crop, labels_crop = data_arg_9(WW, HH, img, labels_ori[num], path_new, index)

    train_names.extend(train_name_crop)
    labels.extend(labels_crop)

    num = num + 1


dataframe = pd.DataFrame({'ID':train_names,'Labels':labels})
dataframe.to_csv('cls_data/ISIC-2017_'+class_p+'_Part3_GroundTruth_crop'+str(aug_num)+'_cls.csv',index=False)
