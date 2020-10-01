import pandas as pd
import os

path = 'ISIC/'
if not os.path.isdir(path):
    os.mkdir(path)

# train classification
class_p = 'Training_Add'

txtName = 'ISIC/'+class_p+'_cls.txt'
f = open(txtName, 'a+')

path = 'cls_data/'+class_p+'_resize_crop_cls/'
path_gt = pd.read_csv('cls_data/ISIC-2017_'+class_p+'_Part3_GroundTruth_crop_cls.csv')
labels = path_gt.Labels
names = path_gt.ID

for i in range(len(names)):
    trainIMG = names[i]
    trainGT = str(labels[i])
    result = trainIMG + ' ' + trainGT +'\n'
    f.write(result)

f.close()



aug_num = 9
# val classification
class_p = 'Validation'

txtName = 'ISIC/'+class_p+'_crop'+str(aug_num)+'_cls.txt'
f = open(txtName, 'a+')

path = 'cls_data/'+class_p+'_resize_crop'+str(aug_num)+'_cls/'
path_gt = pd.read_csv('cls_data/ISIC-2017_'+class_p+'_Part3_GroundTruth_crop'+str(aug_num)+'_cls.csv')
labels = path_gt.Labels
names = path_gt.ID

for i in range(len(names)):
    trainIMG = names[i]
    trainGT = str(labels[i])
    result = trainIMG + ' ' + trainGT +'\n'
    f.write(result)

f.close()


# test classification
class_p = 'Testing'

txtName = 'ISIC/'+class_p+'_crop'+str(aug_num)+'_cls.txt'
f = open(txtName, 'a+')

path = 'cls_data/'+class_p+'_resize_crop'+str(aug_num)+'_cls/'
path_gt = pd.read_csv('cls_data/ISIC-2017_'+class_p+'_Part3_GroundTruth_crop'+str(aug_num)+'_cls.csv')
labels = path_gt.Labels
names = path_gt.ID

for i in range(len(names)):
    trainIMG = names[i]
    trainGT = str(labels[i])
    result = trainIMG + ' ' + trainGT +'\n'
    f.write(result)

f.close()
