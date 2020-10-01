import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from sklearn import metrics
from net.models import Xception_dilation
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from dataset.my_datasets import MyDataSet_cls, MyValDataSet_cls
from torch.utils import data
from apex import amp

model_urls = {'Xception_dilation': 'models/xception-43020ad28.pth'}

INPUT_SIZE = '224, 224'
h, w = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
INPUT_CHANNEL = 4
NUM_CLASSES_SEG = 2
NUM_CLASSES_CLS = 3
BATCH_SIZE = 32
STEPS = 50001
FP16 = False
NAME = 'MaskCN/'


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    CM = metrics.confusion_matrix(label, binary_score)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, auc, AP, sens, spec


def val_mode_Scls(valloader, model, num):
    # valiadation
    pro_score_crop = []
    label_val_crop = []
    for index, batch in enumerate(valloader):
        data, coarsemask, label, name = batch
        data = data.cuda()
        coarsemask = coarsemask.unsqueeze(1).cuda()

        model.eval()
        with torch.no_grad():
            data_cla = torch.cat((data, coarsemask), dim=1)
            pred = model(data_cla)

        pro_score_crop.append(torch.softmax(pred[0], dim=0).cpu().data.numpy())
        label_val_crop.append(label[0].data.numpy())

    pro_score_crop = np.array(pro_score_crop)
    label_val_crop = np.array(label_val_crop)

    pro_score = []
    label_val = []

    for i in range(int(len(label_val_crop) / num)):
        score_sum = 0
        label_sum = 0
        for j in range(num):
            score_sum += pro_score_crop[i * num + j]
            label_sum += label_val_crop[i * num + j]
        pro_score.append(score_sum / num)
        label_val.append(label_sum / num)

    pro_score = np.array(pro_score)
    binary_score = np.eye(3)[np.argmax(np.array(pro_score), axis=-1)]
    label_val = np.eye(3)[np.int64(np.array(label_val))]
    # m
    label_val_a = label_val[:, 1]
    pro_score_a = pro_score[:, 1]
    binary_score_a = binary_score[:, 1]
    val_acc_m, val_auc_m, val_AP_m, sens_m, spec_m = cla_evaluate(label_val_a, binary_score_a, pro_score_a)
    # sk
    label_val_a = label_val[:, 2]
    pro_score_a = pro_score[:, 2]
    binary_score_a = binary_score[:, 2]
    val_acc_sk, val_auc_sk, val_AP_sk, sens_sk, spec_sk = cla_evaluate(label_val_a, binary_score_a, pro_score_a)

    return val_acc_m, val_auc_m, val_AP_m, sens_m, spec_m, val_acc_sk, val_auc_sk, val_AP_sk, sens_sk, spec_sk


def main():
    """Create the network and start the training."""
    writer = SummaryWriter('models/' + NAME)

    cudnn.enabled = True

    ############# Create mask-guided classification network.
    model = Xception_dilation(num_classes=NUM_CLASSES_CLS, input_channel=INPUT_CHANNEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.cuda()
    if FP16 is True:
        model_cls, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    ############# Load pretrained weights
    pretrained_dict = torch.load(model_urls['Xception_dilation'])
    if INPUT_CHANNEL == 4:
        conv1_weights_update = torch.cat(
            (pretrained_dict['conv1.weight'], pretrained_dict['conv1.weight'].mean(1).unsqueeze(1)), dim=1)
        pretrained_dict['conv1.weight'] = conv1_weights_update
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)

    print(len(net_dict))
    print(len(pretrained_dict))

    model.train()
    model.float()

    ce_loss = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    ############# Load training and validation data
    data_train_root = 'dataset/cls_data/Training_Add_resize_crop_cls/'
    data_train_root_mask = 'Coarse_masks/Training_MaskCN/'
    data_train_list = 'dataset/ISIC/Training_Add_cls.txt'
    trainloader = data.DataLoader(MyDataSet_cls(data_train_root, data_train_root_mask, data_train_list, max_iters=STEPS * BATCH_SIZE),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    data_val_root = 'dataset/cls_data/Validation_resize_crop9_cls/'
    data_val_root_mask = 'Coarse_masks/Validation_MaskCN/'
    data_val_list = 'dataset/ISIC/Validation_crop9_cls.txt'
    valloader = data.DataLoader(MyValDataSet_cls(data_val_root, data_val_root_mask, data_val_list), batch_size=1, shuffle=False,
                                num_workers=8,
                                pin_memory=True)

    path = 'models/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'outputxx.txt'

    val_m = []
    val_sk = []
    val_mean = []

    train_loss = []

    ############# Start the training
    for i_iter, batch in tqdm(enumerate(trainloader)):

        lr = adjust_learning_rate(optimizer, i_iter)
        writer.add_scalar('learning_rate', lr, i_iter)

        images, coarsemask, labels, name = batch
        images = images.cuda()
        coarsemask = coarsemask.unsqueeze(1).cuda()
        labels = labels.cuda().long()
        input_cla = torch.cat((images, coarsemask), dim=1)

        optimizer.zero_grad()
        model.train()
        preds = model(input_cla)

        term = ce_loss(preds, labels.long())
        if FP16 is True:
            with amp.scale_loss(term, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            term.backward()
        optimizer.step()

        train_loss.append(term.cpu().data.numpy())
        writer.add_scalar('loss', term.cpu().data.numpy(), i_iter)

        if (i_iter > 500) & (i_iter % 100 == 0):
            epoch = int(i_iter / 100)

            print("train_epoch%d: loss=%f\n" % (epoch, np.nanmean(train_loss)))

            ############# Start the validation
            [val_acc_m, val_auc_m, val_AP_m, val_sens_m, val_spec_m, val_acc_sk, val_auc_sk, val_AP_sk, val_sens_sk,
             val_spec_sk] = val_mode_Scls(valloader, model, 9)
            line_val_m = "val%d:vacc_m=%f,vauc_m=%f,vAP_m=%f,vsens_m=%f,spec_m=%f \n" % (
            epoch, val_acc_m, val_auc_m, val_AP_m, val_sens_m, val_spec_m)
            line_val_sk = "val%d:vacc_sk=%f,vauc_sk=%f,vAP_sk=%f,vsens_sk=%f,vspec_sk=%f \n" % (
            epoch, val_acc_sk, val_auc_sk, val_AP_sk, val_sens_sk, val_spec_sk)
            print(line_val_m)
            print(line_val_sk)
            f = open(f_path, "a")
            f.write(line_val_m)
            f.write(line_val_sk)

            val_m.append(np.nanmean(val_auc_m))
            val_sk.append(np.nanmean(val_auc_sk))
            val_mean.append((np.nanmean(val_auc_m) + np.nanmean(val_auc_sk)) / 2.)

            ############# Plot val curves
            plt.figure()
            plt.plot(val_m, label='val_m', color='red')
            plt.plot(val_sk, label='val_sk', color='green')
            plt.plot(val_mean, label='val_mean', color='blue')
            plt.legend(loc='best')

            plt.savefig(os.path.join(path, 'loss.png'))
            plt.clf()
            plt.close()
            plt.show()

            plt.close('all')

            writer.add_scalar('val_auc_m', np.nanmean(val_auc_m), i_iter)
            writer.add_scalar('val_auc_sk', np.nanmean(val_auc_sk), i_iter)
            writer.add_scalar('val_auc_mean', (np.nanmean(val_auc_m) + np.nanmean(val_auc_sk)) / 2., i_iter)

            ############# Save network
            torch.save(model.state_dict(), path + 'MaskCN_e' + str(epoch) + '.pth')


if __name__ == '__main__':
    main()

