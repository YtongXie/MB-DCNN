import os
from PIL import Image


def resize_bic(image, label):
    w, h = label.size
    x = 1600
    if (w > x) | (h > x):
        if w > h:
            y = int(h * x / w)
            out_im = image.resize((x, y), Image.BICUBIC)
            out_la = label.resize((x, y), Image.NEAREST)
        else:
            y = int(w * x / h)
            out_im = image.resize((y, x), Image.BICUBIC)
            out_la = label.resize((y, x), Image.NEAREST)

    else:
        out_im = image
        out_la = label

    return out_im, out_la


##### Training
class_p = 'Training'

imagePathDir = os.listdir('data/ISIC-2017_'+class_p+'_Data/Images/')
imagePathDir.sort()
maskPathDir = os.listdir('data/ISIC-2017_'+class_p+'_Data/Annotation/')
maskPathDir.sort()

path_new = 'seg_data/'+class_p+'_resize_seg/Images/'
if not os.path.isdir(path_new):
    os.makedirs(path_new)

path_gt_new = 'seg_data/'+class_p+'_resize_seg/Annotation/'
if not os.path.isdir(path_gt_new):
    os.makedirs(path_gt_new)

num = 0
label = []
train = []
for index in imagePathDir:
    print(num)
    # read img
    img = Image.open('data/ISIC-2017_'+class_p+'_Data/Images/'+index)
    mask = Image.open('data/ISIC-2017_'+class_p+'_Data/Annotation/'+index[:-4]+'_segmentation.png')
    img_re, mask_re = resize_bic(img, mask)

    img_re.save(path_new + index[:-4] + '.png')
    mask_re.save(path_gt_new + index[:-4] + '.png')

    num = num + 1


