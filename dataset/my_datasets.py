import numpy as np
import torchvision.transforms.functional as tf
import random
from torch.utils import data
from torchvision import transforms
from PIL import Image

################# Dataset for Seg
class MyDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224), max_iters=None):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_w, self.crop_h = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        self.train_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(224)
             ])

        self.train_coarsemask_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(224)
             ])

        self.train_gt_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(224)
             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        if self.root_path_coarsemask is None:

            image = Image.open(self.root_path + datafiles["img"])
            label = Image.open(self.root_path + datafiles["label"])

            is_crop = [0,1]
            random.shuffle(is_crop)

            if is_crop[0] == 0:
                [WW, HH] = image.size
                p_center = [int(WW / 2), int(HH / 2)]
                crop_num = np.array(range(30, int(np.mean(p_center) / 2), 30))

                random.shuffle(crop_num)
                crop_p = crop_num[0]
                rectangle = (crop_p, crop_p, WW - crop_p, HH - crop_p)
                image = image.crop(rectangle)
                label = label.crop(rectangle)

                image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
                label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

            else:
                image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
                label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

            seed = np.random.randint(2147483647)
            random.seed(seed)
            image = self.train_augmentation(image)

            random.seed(seed)
            label = self.train_gt_augmentation(label)

            image = np.array(image) / 255.
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32)

            label = np.array(label)
            label = np.float32(label > 0)

            name = datafiles["img"][7:23]

            return image.copy(), label.copy(), name

        else:
            image = Image.open(self.root_path + datafiles["img"])
            coarsemask = Image.open(self.root_path_coarsemask + datafiles["img"][7::])
            label = Image.open(self.root_path + datafiles["label"])
            assert coarsemask.size == label.size

            is_crop = [0,1]
            random.shuffle(is_crop)

            if is_crop[0] == 0:
                [WW, HH] = image.size
                p_center = [int(WW / 2), int(HH / 2)]
                crop_num = np.array(range(30, int(np.mean(p_center) / 2), 30))

                random.shuffle(crop_num)
                crop_p = crop_num[0]
                rectangle = (crop_p, crop_p, WW - crop_p, HH - crop_p)
                image = image.crop(rectangle)
                coarsemask = coarsemask.crop(rectangle)
                label = label.crop(rectangle)

                image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
                coarsemask = coarsemask.resize((self.crop_w, self.crop_h), Image.NEAREST)
                label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

            else:
                image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
                coarsemask = coarsemask.resize((self.crop_w, self.crop_h), Image.NEAREST)
                label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)

            seed = np.random.randint(2147483647)
            random.seed(seed)
            image = self.train_augmentation(image)

            random.seed(seed)
            coarsemask = self.train_coarsemask_augmentation(coarsemask)

            random.seed(seed)
            label = self.train_gt_augmentation(label)

            image = np.array(image) / 255.
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32)

            coarsemask = np.array(coarsemask)
            coarsemask = np.float32(coarsemask > 0)

            label = np.array(label)
            label = np.float32(label > 0)

            name = datafiles["img"][7:23]

            return image.copy(), coarsemask.copy(), label.copy(), name


class MyValDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224)):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        if self.root_path_coarsemask is None:
            image = Image.open(self.root_path + datafiles["img"])
            label = Image.open(self.root_path + datafiles["label"])

            image = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
            label = label.resize((self.crop_h, self.crop_w), Image.NEAREST)

            image = np.array(image) / 255.
            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32)

            label = np.array(label)

            name = datafiles["img"][7:23]

            return image.copy(), label.copy(), name
        else:
            image = Image.open(self.root_path + datafiles["img"])
            coarsemask = Image.open(self.root_path_coarsemask + datafiles["img"][7::])
            label = Image.open(self.root_path + datafiles["label"])
            assert coarsemask.size == label.size

            image = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
            coarsemask = coarsemask.resize((self.crop_h, self.crop_w), Image.NEAREST)
            label = label.resize((self.crop_h, self.crop_w), Image.NEAREST)

            image = np.array(image) / 255.
            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32)

            coarsemask = np.array(coarsemask)
            coarsemask = np.float32(coarsemask > 0)

            label = np.array(label)

            name = datafiles["img"][7:23]

            return image.copy(), coarsemask.copy(), label.copy(), name


class MyTestDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224)):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        if self.root_path_coarsemask is None:
            image = Image.open(self.root_path + datafiles["img"])
            label = Image.open(self.root_path + datafiles["label"])

            image0 = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
            image0 = np.array(image0) / 255.
            image0 = image0.transpose(2, 0, 1).astype(np.float32)

            image1 = image.resize((self.crop_h + 32, self.crop_w + 32), Image.BICUBIC)
            image1 = np.array(image1) / 255.
            image1 = image1.transpose(2, 0, 1).astype(np.float32)

            image2 = image.resize((self.crop_h + 64, self.crop_w + 64), Image.BICUBIC)
            image2 = np.array(image2) / 255.
            image2 = image2.transpose(2, 0, 1).astype(np.float32)

            label = np.array(label)

            name = datafiles["img"][7:23]

            return image0.copy(), image1.copy(), image2.copy(), label.copy(), name

        else:
            image = Image.open(self.root_path + datafiles["img"])
            coarsemask = Image.open(self.root_path_coarsemask + datafiles["img"][7::])
            label = Image.open(self.root_path + datafiles["label"])
            assert coarsemask.size == label.size

            image0 = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
            image0 = np.array(image0) / 255.
            image0 = image0.transpose(2, 0, 1).astype(np.float32)
            coarsemask0 = coarsemask.resize((self.crop_h, self.crop_w), Image.NEAREST)
            coarsemask0 = np.float32(np.array(coarsemask0) > 0)

            image1 = image.resize((self.crop_h + 32, self.crop_w + 32), Image.BICUBIC)
            image1 = np.array(image1) / 255.
            image1 = image1.transpose(2, 0, 1).astype(np.float32)
            coarsemask1 = coarsemask.resize((self.crop_h + 32, self.crop_w + 32), Image.NEAREST)
            coarsemask1 = np.float32(np.array(coarsemask1) > 0)

            image2 = image.resize((self.crop_h + 64, self.crop_w + 64), Image.BICUBIC)
            image2 = np.array(image2) / 255.
            image2 = image2.transpose(2, 0, 1).astype(np.float32)
            coarsemask2 = coarsemask.resize((self.crop_h + 64, self.crop_w + 64), Image.NEAREST)
            coarsemask2 = np.float32(np.array(coarsemask2) > 0)

            label = np.array(label)

            name = datafiles["img"][7:23]

            return image0.copy(), image1.copy(), image2.copy(), coarsemask0.copy(), coarsemask1.copy(), coarsemask2.copy(), label.copy(), name


################# Dataset for generating Coarsemask
class MyGenDataSet(data.Dataset):
    def __init__(self, root_path, list_path, mode=0, crop_size=(224, 224)):
        self.root_path = root_path
        self.list_path = list_path
        self.mode = mode
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_ori = Image.open(self.root_path + datafiles["img"])
        if self.mode == 0:
            image = np.array(image_ori) / 255.
            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32)
            name = datafiles["img"]
            return image.copy(), name
        else:
            image = image_ori.resize((self.crop_h, self.crop_w), Image.BICUBIC)
            image = np.array(image) / 255.
            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32)
            image_ori = np.array(image_ori)
            name = datafiles["img"][7:23]
            return image_ori.copy(), image.copy(), name


################# Dataset for MaskCN
class MyDataSet_cls(data.Dataset):
    def __init__(self, root_path, root_path_coarsemask, list_path, max_iters=None):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        self.train_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(224)
             ])

        self.train_coarsemask_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(224)
             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(self.root_path + datafiles["img"])
        coarsemask = Image.open(self.root_path_coarsemask + datafiles["img"])
        label = np.array(np.int(datafiles["label"]))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        image = self.train_augmentation(image)

        random.seed(seed)
        coarsemask = self.train_coarsemask_augmentation(coarsemask)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        coarsemask = np.array(coarsemask)
        coarsemask = np.float32(coarsemask > 0)

        name = datafiles["img"]

        return image.copy(), coarsemask.copy(), label, name


class MyValDataSet_cls(data.Dataset):
    def __init__(self, root_path, root_path_coarsemask, list_path, crop_size=(224, 224)):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ') + 1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(self.root_path + datafiles["img"])
        coarsemask = Image.open(self.root_path_coarsemask + datafiles["img"])
        label = np.array(np.int(datafiles["label"]))

        image = np.array(image) / 255.
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)

        coarsemask = np.array(coarsemask)
        coarsemask = np.float32(coarsemask > 0)

        name = datafiles["img"]

        return image.copy(), coarsemask.copy(), label, name


