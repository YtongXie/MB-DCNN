import torch
import torch.nn as nn

def dice_loss(predict, target):

	smooth = 1e-5

	y_true_f = target.contiguous().view(target.shape[0], -1)
	y_pred_f = predict.contiguous().view(predict.shape[0], -1)
	intersection = torch.sum(torch.mul(y_pred_f, y_true_f), dim=1)
	union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1) + smooth
	dice_score = (2.0 * intersection / union)

	dice_loss = 1 - dice_score

	return dice_loss

def rank_loss(predict, target):
	top_k = 30

	y_true_f = target.contiguous().view(target.shape[0], -1)
	y_pred_f = predict.contiguous().view(predict.shape[0], -1)

	N_topvalue, N_indice = (y_pred_f * (1 - y_true_f)).topk(top_k, dim=-1, largest=True, sorted=True)

	P_values, P_indice = ((1.0 - y_pred_f) * y_true_f).topk(top_k, dim=-1, largest=True, sorted=True)
	P_downvalue = 1 - P_values

	beta = 1
	rank_loss = 0
	for i in range(top_k):
		for j in range(top_k):
			th_value = N_topvalue[:,i] - beta * P_downvalue[:,j] + 0.3
			rank_loss = rank_loss + (th_value * (th_value>0).float()).mean()

	return rank_loss/(top_k * top_k)


class Fusin_Dice_rank(nn.Module):
	def __init__(self):
		super(Fusin_Dice_rank, self).__init__()

	def forward(self, predicts, target):

		preds = torch.softmax(predicts, dim=1)
		dice_loss0 = dice_loss(preds[:, 0, :, :], 1 - target)
		dice_loss1 = dice_loss(preds[:, 1, :, :], target)
		loss_D = (dice_loss0.mean() + dice_loss1.mean())/2.0

		loss_R = rank_loss(preds[:, 1, :, :], target)

		return loss_D, loss_R
