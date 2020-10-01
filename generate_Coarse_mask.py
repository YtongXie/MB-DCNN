import torch
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from skimage import io
from net.models import deeplabv3plus
from dataset.my_datasets import MyGenDataSet
from torch.utils import data


def generate_mode_seg0(dataloader, model, path):

    for index, batch in tqdm(enumerate(dataloader)):
        image, name = batch
        image = image.cuda()
        # print(name)

        rot_90 = torch.rot90(image, 1, [2, 3])
        rot_180 = torch.rot90(image, 2, [2, 3])
        rot_270 = torch.rot90(image, 3, [2, 3])
        hor_flip = torch.flip(image, [-1])
        ver_flip = torch.flip(image, [-2])
        image = torch.cat([image, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)

        model.eval()
        with torch.no_grad():
            pred = model(image)

        pred = pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.int16(np.argmax(pred[0], axis=0))

        io.imsave(os.path.join(path, name[0]), np.int64(pred_arg) * 255)

    return True

def generate_mode_seg1(dataloader, model, path):

    for index, batch in tqdm(enumerate(dataloader)):
        image_ori, image, name = batch
        image = image.cuda()
        # print(name)

        rot_90 = torch.rot90(image, 1, [2, 3])
        rot_180 = torch.rot90(image, 2, [2, 3])
        rot_270 = torch.rot90(image, 3, [2, 3])
        hor_flip = torch.flip(image, [-1])
        ver_flip = torch.flip(image, [-2])
        image = torch.cat([image, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)

        model.eval()
        with torch.no_grad():
            pred = model(image)

        pred = pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.int16(np.argmax(pred[0], axis=0))
        pred_arg = cv2.resize(pred_arg, (image_ori.shape[2], image_ori.shape[1]), interpolation=cv2.INTER_NEAREST)

        io.imsave(os.path.join(path, name[0]), np.int64(pred_arg) * 255)

    return True


########################### Load coarse segmentation network.
cudnn.enabled = True
model = deeplabv3plus(num_classes=2)
model.cuda()
model = torch.nn.DataParallel(model)
pretrained_dict = torch.load('models/DR_CoarseSN/CoarseSN.pth')
model.load_state_dict(pretrained_dict)
model.eval()
model.float()


########################### Coarse_masks for MaskCN

#### Training
class_p = 'Training'
data_root = 'dataset/cls_data/'+class_p+'_Add_resize_crop_cls/'
data_list = 'dataset/ISIC/'+class_p+'_Add_cls.txt'
dataloader = data.DataLoader(MyGenDataSet(data_root, data_list, mode=0), batch_size=1, shuffle=False, num_workers=8,
                            pin_memory=True)

path = 'Coarse_masks/'+class_p+'_MaskCN/'
if not os.path.isdir(path):
    os.makedirs(path)

generate_mode_seg0(dataloader, model, path)


#### Validation
class_p = 'Validation' ### 'Testing'
data_root = 'dataset/cls_data/'+class_p+'_resize_crop9_cls/'
data_list = 'dataset/ISIC/'+class_p+'_crop9_cls.txt'
dataloader = data.DataLoader(MyGenDataSet(data_root, data_list, mode=0), batch_size=1, shuffle=False, num_workers=8,
                            pin_memory=True)

path = 'Coarse_masks/'+class_p+'_MaskCN/'
if not os.path.isdir(path):
    os.makedirs(path)

generate_mode_seg0(dataloader, model, path)



########################### Coarse_masks for EnhancedSN

#### Training
class_p = 'Training'
data_root = 'dataset/seg_data/'+class_p+'_resize_seg/'
data_list = 'dataset/ISIC/'+class_p+'_seg.txt'
dataloader = data.DataLoader(MyGenDataSet(data_root, data_list, mode=1), batch_size=1, shuffle=False, num_workers=8,
                            pin_memory=True)

path = 'Coarse_masks/'+class_p+'_EnhancedSN/'
if not os.path.isdir(path):
    os.makedirs(path)

generate_mode_seg1(dataloader, model, path)


#### Validation
class_p = 'Validation' ### 'Testing'
data_root = 'dataset/seg_data/ISIC-2017_'+class_p+'_Data/'
data_list = 'dataset/ISIC/'+class_p+'_seg.txt'
dataloader = data.DataLoader(MyGenDataSet(data_root, data_list, mode=1), batch_size=1, shuffle=False, num_workers=8,
                            pin_memory=True)

path = 'Coarse_masks/'+class_p+'_EnhancedSN/'
if not os.path.isdir(path):
    os.makedirs(path)

generate_mode_seg1(dataloader, model, path)


