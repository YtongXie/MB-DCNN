import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from net.models import Xception_dilation, deeplabv3plus_en
from sklearn.metrics import accuracy_score
from apex import amp
from dataset.my_datasets import MyTestDataSet_seg
from torch.utils import data
import torch.nn.functional as F


def val_pred(MaskCN, EnhanceSN, image, coarsemask):

    rot_90 = torch.rot90(image, 1, [2, 3])
    rot_180 = torch.rot90(image, 2, [2, 3])
    rot_270 = torch.rot90(image, 3, [2, 3])
    hor_flip = torch.flip(image, [-1])
    ver_flip = torch.flip(image, [-2])
    image = torch.cat([image, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)

    rot_90_cm = torch.rot90(coarsemask, 1, [2, 3])
    rot_180_cm = torch.rot90(coarsemask, 2, [2, 3])
    rot_270_cm = torch.rot90(coarsemask, 3, [2, 3])
    hor_flip_cm = torch.flip(coarsemask, [-1])
    ver_flip_cm = torch.flip(coarsemask, [-2])
    coarsemask = torch.cat([coarsemask, rot_90_cm, rot_180_cm, rot_270_cm, hor_flip_cm, ver_flip_cm], dim=0)

    EnhanceSN.eval()
    with torch.no_grad():
        data_cla = torch.cat((image, coarsemask), dim=1)
        cla_cam = cam(MaskCN, data_cla)
        cla_cam = torch.from_numpy(np.stack(cla_cam)).unsqueeze(1).cuda()
        pred = EnhanceSN(image, cla_cam)

    pred = pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])

    return pred


def val_mode_seg(valloader, MaskCN, EnhanceSN):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    for index, batch in tqdm(enumerate(valloader)):

        image0, image1, image2, coarsemask0, coarsemask1, coarsemask2, mask, name = batch
        image0 = image0.cuda()
        image1 = image1.cuda()
        image2 = image2.cuda()
        coarsemask0 = coarsemask0.unsqueeze(1).cuda()
        coarsemask1 = coarsemask1.unsqueeze(1).cuda()
        coarsemask2 = coarsemask2.unsqueeze(1).cuda()

        mask = mask[0].data.numpy()
        test_mask = np.int64(mask > 0)
        # print(name)

        pred0 = val_pred(MaskCN, EnhanceSN, image0, coarsemask0)
        pred1 = val_pred(MaskCN, EnhanceSN, image1, coarsemask1)
        pred2 = val_pred(MaskCN, EnhanceSN, image2, coarsemask2)
        pred0 = F.interpolate(pred0, size=(mask.shape[0], mask.shape[1]), mode='bicubic')
        pred1 = F.interpolate(pred1, size=(mask.shape[0], mask.shape[1]), mode='bicubic')
        pred2 = F.interpolate(pred2, size=(mask.shape[0], mask.shape[1]), mode='bicubic')
        pred = pred0 + pred1 + pred2

        pred = torch.softmax(pred[0], dim=0).cpu().data.numpy()
        pred_arg = np.int16(np.argmax(pred, axis=0))

        # y_pred
        y_true_f = test_mask.reshape(test_mask.shape[0] * test_mask.shape[1], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1], order='F')

        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)


def cam(model, inputs):
    with torch.no_grad():
        preds = model(inputs)
        class_idx = preds.argmax(dim=1)
        model_layers = model.get_layers()

    params = list(model.parameters())
    weights = np.squeeze(params[-2].data.cpu().numpy())
    bz, nc, h, w = model_layers[0].shape
    output_cam = []
    for idx in range(bz):
        cam = np.zeros((h, w), dtype=np.float32)
        for i, weight in enumerate(weights[class_idx[idx]]):
            cam += weight * model_layers[0][idx][i].data.cpu().numpy()

        cam_img = np.maximum(cam, 0)
        cam_img = cam / np.max(cam_img)
        output_cam.append(cam_img)

    return output_cam

model_urls = {'MaskCN': 'models/MaskCN/MaskCN.pth', 'EnhancedSN': 'models/DR_EnhanceSN/CoarseSN.pth'}

INPUT_CHANNEL = 4
NUM_CLASSES_SEG = 2
NUM_CLASSES_CLS = 3

cudnn.enabled = True

############# Load mask-guided classification network and pretrained weights
MaskCN = Xception_dilation(num_classes=NUM_CLASSES_CLS, input_channel=INPUT_CHANNEL)
MaskCN.cuda()
pretrained_dict = torch.load(model_urls['MaskCN'])
MaskCN.load_state_dict(pretrained_dict)
MaskCN.eval()


############# Load enhanced segmentation network and pretrained weights
EnhanceSN = deeplabv3plus_en(num_classes=NUM_CLASSES_SEG)
EnhanceSN.cuda()
# EnhanceSN = amp.initialize(EnhanceSN, opt_level="O1")
EnhanceSN = torch.nn.DataParallel(EnhanceSN)
pretrained_dict = torch.load(model_urls['EnhancedSN'])
EnhanceSN.load_state_dict(pretrained_dict)
EnhanceSN.eval()


############# Load testing data
data_test_root = 'dataset/seg_data/ISIC-2017_Testing_Data/'
data_test_root_mask = 'Coarse_masks/Testing_EnhancedSN/'
data_test_list = 'dataset/ISIC/Testing_seg.txt'
testloader = data.DataLoader(MyTestDataSet_seg(data_test_root, data_test_list, root_path_coarsemask=data_test_root_mask), batch_size=1, shuffle=False,
                             num_workers=8,
                             pin_memory=True)


############# Start the testing
[tacc, tdice, tsen, tspe, tjac_score] = val_mode_seg(testloader, MaskCN, EnhanceSN)
line_test = "test: tacc=%f, tdice=%f, tsensitivity=%f, tspecifity=%f, tjac=%f \n" % \
            (np.nanmean(tacc), np.nanmean(tdice), np.nanmean(tsen), np.nanmean(tspe),
             np.nanmean(tjac_score))
print(line_test)
