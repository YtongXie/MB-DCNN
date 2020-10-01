import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from net.models import Xception_dilation, deeplabv3plus_en
from sklearn.metrics import accuracy_score
from net import loss
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from apex import amp
from tensorboardX import SummaryWriter
from dataset.my_datasets import MyDataSet_seg, MyValDataSet_seg
from torch.utils import data


INPUT_SIZE = '224, 224'
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 5e-5
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
INPUT_CHANNEL = 4
NUM_CLASSES_SEG = 2
NUM_CLASSES_CLS = 3
TRAIN_NUM = 2000
BATCH_SIZE = 16
EPOCH = 500
STEPS = (TRAIN_NUM / BATCH_SIZE) * EPOCH
FP16 = True
NAME = 'DR_EnhanceSN/'


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def val_mode_seg(valloader, val_cams, EnhanceSN, path, epoch):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    for index, batch in tqdm(enumerate(valloader)):

        data, coarsemask, mask, name = batch
        data = data.cuda()
        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)
        # print(name)

        EnhanceSN.eval()
        with torch.no_grad():
            cla_cam = val_cams[index]
            cla_cam = torch.from_numpy(cla_cam).unsqueeze(0).unsqueeze(0)
            pred = EnhanceSN(data, cla_cam.cuda())

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)

        # y_pred
        y_true_f = val_mask.reshape(val_mask.shape[0] * val_mask.shape[1], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1], order='F')

        intersection = np.float(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

        if index in [100]:
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax.imshow(data[0].cpu().data.numpy().transpose(1, 2, 0))
            ax.axis('off')
            ax = fig.add_subplot(132)
            ax.imshow(mask)
            ax.axis('off')
            ax = fig.add_subplot(133)
            ax.imshow(pred_arg)
            ax.axis('off')
            fig.suptitle('RGB image,ground truth mask, predicted mask', fontsize=6)
            fig.savefig(path + name[0][:-4] + '_e' + str(epoch) + '.png', dpi=200, bbox_inches='tight')
            ax.cla()
            fig.clf()
            plt.close()

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)


def val_mode_cam(valloader, MaskCN):

    val_cam = []
    for index, batch in tqdm(enumerate(valloader)):

        data, coarsemask, mask, name = batch
        data = data.cuda()
        coarsemask = coarsemask.unsqueeze(1).cuda()
        # print(name)

        with torch.no_grad():
            data_cla = torch.cat((data, coarsemask), dim=1)
            cla_cam = cam(MaskCN, data_cla)

        val_cam.append(cla_cam[0])

    return val_cam


def Jaccard(pred_arg, mask):
    pred_arg = np.argmax(pred_arg.cpu().data.numpy(), axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], order='F')
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2], order='F')

    intersection = np.float(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


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


def main():
    """Create the network and start the training."""
    model_urls = {'CoarseSN': 'models/DR_CoarseSN/CoarseSN.pth', 'MaskCN': 'models/MaskCN/MaskCN.pth'}

    writer = SummaryWriter('models/' + NAME)

    cudnn.enabled = True

    ############# Create mask-guided classification network.
    MaskCN = Xception_dilation(num_classes=NUM_CLASSES_CLS, input_channel=INPUT_CHANNEL)
    MaskCN.cuda()
    if FP16 is True:
        MaskCN = amp.initialize(MaskCN, opt_level="O1")

    ############# Load pretrained weights
    pretrained_dict = torch.load(model_urls['MaskCN'])
    MaskCN.load_state_dict(pretrained_dict)
    MaskCN.eval()

    ############# Create enhanced segmentation network.
    EnhanceSN = deeplabv3plus_en(num_classes=NUM_CLASSES_SEG)
    optimizer = torch.optim.Adam(EnhanceSN.parameters(), lr=LEARNING_RATE)
    EnhanceSN.cuda()
    if FP16 is True:
        EnhanceSN, optimizer = amp.initialize(EnhanceSN, optimizer, opt_level="O1")
    EnhanceSN = torch.nn.DataParallel(EnhanceSN)

    ############# Load pretrained weights
    pretrained_dict = torch.load(model_urls['CoarseSN'])
    net_dict = EnhanceSN.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    EnhanceSN.load_state_dict(net_dict)
    EnhanceSN.train()
    EnhanceSN.float()

    print(len(net_dict))
    print(len(pretrained_dict))

    DR_loss = loss.Fusin_Dice_rank()

    cudnn.benchmark = True

    ############# Load training and validation data
    data_train_root = 'dataset/seg_data/Training_resize_seg/'
    data_train_root_mask = 'Coarse_masks/Training_EnhancedSN/'
    data_train_list = 'dataset/ISIC/Training_seg.txt'
    trainloader = data.DataLoader(MyDataSet_seg(data_train_root, data_train_list, root_path_coarsemask=data_train_root_mask, crop_size=(w, h)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    data_val_root = 'dataset/seg_data/ISIC-2017_Validation_Data/'
    data_val_root_mask = 'Coarse_masks/Validation_EnhancedSN/'
    data_val_list = 'dataset/ISIC/Validation_seg.txt'
    valloader = data.DataLoader(MyValDataSet_seg(data_val_root, data_val_list, root_path_coarsemask=data_val_root_mask), batch_size=1, shuffle=False,
                                num_workers=8,
                                pin_memory=True)

    ############# Generate CAM for validation data
    val_cams = val_mode_cam(valloader, MaskCN)

    path = 'models/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'outputxx.txt'

    val_jac = []

    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_D = []
        train_loss_R = []
        train_loss_total = []
        train_jac = []

        for i_iter, batch in tqdm(enumerate(trainloader)):

            # if i_iter > 50:
            #     continue

            step = (TRAIN_NUM / BATCH_SIZE) * epoch + i_iter

            images, coarsemask, labels, name = batch
            images = images.cuda()
            coarsemask = coarsemask.unsqueeze(1).cuda()
            labels = labels.cuda().squeeze(1)

            with torch.no_grad():
                input_cla = torch.cat((images, coarsemask), dim=1)
                cla_cam = cam(MaskCN, input_cla)

            cla_cam = torch.from_numpy(np.stack(cla_cam)).unsqueeze(1).cuda()

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, step)

            EnhanceSN.train()
            preds = EnhanceSN(images, cla_cam)

            loss_D, loss_R = DR_loss(preds, labels)
            term = loss_D + 0.05 * loss_R

            if FP16 is True:
                with amp.scale_loss(term, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                term.backward()
            optimizer.step()

            writer.add_scalar('learning_rate', lr, step)
            writer.add_scalar('loss', term.cpu().data.numpy(), step)

            train_loss_D.append(loss_D.cpu().data.numpy())
            train_loss_R.append(loss_R.cpu().data.numpy())
            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds, labels))


        print("train_epoch%d: lossTotal=%f, lossDice=%f, lossRank=%f, Jaccard=%f \n" % (
        epoch, np.nanmean(train_loss_total), np.nanmean(train_loss_D), np.nanmean(train_loss_R), np.nanmean(train_jac)))


        ############# Start the validation
        [vacc, vdice, vsen, vspe, vjac_score] = val_mode_seg(valloader, val_cams, EnhanceSN, path, epoch)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f \n" % \
                   (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score))

        print(line_val)
        f = open(f_path, "a")
        f.write(line_val)

        val_jac.append(np.nanmean(vjac_score))

        ############# Plot val curve
        plt.figure()
        plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')

        writer.add_scalar('val_Jaccard', np.nanmean(vjac_score), epoch)

        ############# Save network
        torch.save(EnhanceSN.state_dict(), path + 'CoarseSN_e' + str(epoch) + '.pth')


if __name__ == '__main__':
    main()

