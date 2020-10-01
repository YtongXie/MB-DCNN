import torch
import torch.nn as nn
import torch.nn.functional as F
import net.xception as xception

class SeparableConv2d(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
		super(SeparableConv2d,self).__init__()

		self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
		self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

	def forward(self,x):
		x = self.conv1(x)
		x = self.pointwise(x)
		return x


class Block(nn.Module):
	def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
		super(Block, self).__init__()

		if out_filters != in_filters or strides!=1:
			self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
			self.skipbn = nn.BatchNorm2d(out_filters)
		else:
			self.skip=None

		rep=[]

		filters=in_filters
		if grow_first:
			rep.append(nn.ReLU(inplace=True))
			rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(out_filters))
			filters = out_filters

		for i in range(reps-1):
			rep.append(nn.ReLU(inplace=True))
			rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(filters))

		if not grow_first:
			rep.append(nn.ReLU(inplace=True))
			rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
			rep.append(nn.BatchNorm2d(out_filters))

		if not start_with_relu:
			rep = rep[1:]
		else:
			rep[0] = nn.ReLU(inplace=True)

		if strides != 1:
			rep.append(nn.MaxPool2d(3,strides,1))
		self.rep = nn.Sequential(*rep)

	def forward(self,inp):
		x = self.rep(inp)

		if self.skip is not None:
			skip = self.skip(inp)
			skip = self.skipbn(skip)
		else:
			skip = inp

		x+=skip
		return x

class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch3 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch4 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out)
		self.branch5_relu = nn.ReLU(inplace=True)
		self.conv_cat = nn.Sequential(
			nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)


	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x, 2, True)
		global_feature = torch.mean(global_feature, 3, True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result


class deeplabv3plus(nn.Module):
	def __init__(self, num_classes=None):
		super(deeplabv3plus, self).__init__()
		self.MODEL_NUM_CLASSES = num_classes
		self.backbone = None
		self.backbone_layers = None
		self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//16, bn_mom = 0.99)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

		self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)
		self.cat_conv = nn.Sequential(
				nn.Conv2d(256+48, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, self.MODEL_NUM_CLASSES, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.backbone = xception.Xception(os=16)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat)
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result


class deeplabv3plus_en(nn.Module):
	def __init__(self, num_classes=None):
		super(deeplabv3plus_en, self).__init__()
		self.MODEL_NUM_CLASSES = num_classes
		self.backbone = None
		self.backbone_layers = None
		self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//16, bn_mom = 0.99)
		self.dropout1 = nn.Dropout(0.5)

		self.cam_conv = nn.Sequential(nn.Conv2d(256+1, 256, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
		)

		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

		self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)

		self.cat_conv = nn.Sequential(
				nn.Conv2d(256+48, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, self.MODEL_NUM_CLASSES, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.backbone = xception.Xception(os=16)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x, cla_cam):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)

		feature_cat0 = torch.cat([feature_aspp, cla_cam], 1)
		feature_cam = self.cam_conv(feature_cat0)
		feature_cam = self.upsample_sub(feature_cam)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat1 = torch.cat([feature_cam, feature_shallow], 1)
		result = self.cat_conv(feature_cat1)
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result


class Xception_dilation(nn.Module):
	def __init__(self, input_channel=None, num_classes=None):
		super(Xception_dilation, self).__init__()
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(input_channel, 32, 3, 2, 0, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu1 = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
		self.bn2 = nn.BatchNorm2d(64)
		self.relu2 = nn.ReLU(inplace=True)
		# do relu here

		self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
		self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
		self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

		self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
		self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
		self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
		self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

		self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
		self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
		self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
		self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

		self.residual = nn.Sequential(
			nn.Conv2d(728, 1024, 1, 1, dilation=2, bias=False),
			nn.BatchNorm2d(1024),
		)

		self.SepConv1 = nn.Sequential(
			nn.ReLU(inplace=False),
			SeparableConv2d(728, 728, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(728)
		)

		self.SepConv2 = nn.Sequential(
			nn.ReLU(inplace=False),
			SeparableConv2d(728, 1024, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(1024)
		)

		self.SepConv3 = nn.Sequential(
			SeparableConv2d(1024, 1536, 3, dilation=2, stride=1, padding=2, bias=False),
			nn.BatchNorm2d(1536),
			nn.ReLU(inplace=False)
		)

		self.SepConv4 = nn.Sequential(
			SeparableConv2d(1536, 2048, 3, dilation=2, stride=1, padding=2, bias=False),
			nn.BatchNorm2d(2048),
			nn.ReLU(inplace=False)
		)

		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.cls = nn.Linear(2048, num_classes)


	def get_layers(self):
		return self.layers

	def forward(self, x):
		self.layers = []
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)

		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		x = self.block6(x)
		x = self.block7(x)
		x = self.block8(x)
		x = self.block9(x)
		x = self.block10(x)
		x = self.block11(x)

		res = self.residual(x)
		x = self.SepConv1(x)
		x = self.SepConv2(x)
		x += res
		x = self.SepConv3(x)
		x = self.SepConv4(x)
		self.layers.append(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.cls(x)
		self.layers.append(x)

		return x


