import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn import metrics
from net.models import Xception_dilation
from dataset.my_datasets import MyValDataSet_cls
from torch.utils import data
from apex import amp

INPUT_SIZE = '224, 224'
h, w = map(int, INPUT_SIZE.split(','))
INPUT_CHANNEL = 4
NUM_CLASSES_CLS = 3

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
    for index, batch in tqdm(enumerate(valloader)):
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


cudnn.enabled = True

############# Load mask-guided classification network and pretrained weights
model = Xception_dilation(num_classes=NUM_CLASSES_CLS, input_channel=INPUT_CHANNEL)
model.cuda()
# model = amp.initialize(model, opt_level="O1")
pretrained_dict = torch.load('models/MaskCN/MaskCN.pth')
model.load_state_dict(pretrained_dict)


############# Load testing data
data_test_root = 'dataset/cls_data/Testing_resize_crop9_cls/'
data_test_root_mask = 'Coarse_masks/Testing_MaskCN/'
data_test_list = 'dataset/ISIC/Testing_crop9_cls.txt'
testloader = data.DataLoader(MyValDataSet_cls(data_test_root, data_test_root_mask, data_test_list), batch_size=1, shuffle=False,
                             num_workers=8,
                             pin_memory=True)

############# Start the testing
[test_acc_m, test_auc_m, test_AP_m, test_sens_m, test_spec_m, test_acc_sk, test_auc_sk, test_AP_sk,
 test_sens_sk, test_spec_sk] = val_mode_Scls(testloader, model, 9)
line_test_m = "test:tacc_m=%f,tauc_m=%f,tAP_m=%f,tsens_m=%f,tspec_m=%f \n" % (test_acc_m, test_auc_m, test_AP_m, test_sens_m, test_spec_m)
line_test_sk = "test:tacc_sk=%f,tauc_sk=%f,tAP_sk=%f,tsens_sk=%f,tspec_sk=%f \n" % (test_acc_sk, test_auc_sk, test_AP_sk, test_sens_sk, test_spec_sk)
print(line_test_m)
print(line_test_sk)

