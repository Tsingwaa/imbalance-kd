"""
Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    """This Class Object is built for generate an imbalanced CIFAR10 for the official CIFAR10 dataset."""
    cls_num = 10

    def __init__(self, phase, imbalance_ratio, root=None, imb_type='exp'):
        np.random.seed(230)

        self.num_per_cls_dict = dict()
        isTrain = True if phase == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, isTrain, transform=None, target_transform=None, download=True)
        self.isTrain = isTrain
        if self.isTrain:
            # generate the number of every class as a list
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            # According to the generated list, generate imbalanced data
            self.gen_imbalanced_data(img_num_list)

            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            # if not isTrain phase, there is not generate imbalanced data process but straightly transform.
            self.transform = transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.labels = self.targets

        print("{} Mode: {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, annotation in enumerate(self.get_annotations()):
            cat_id = annotation["category_id"]
            if cat_id not in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        """Get number of each class"""
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        """Get imbalanced data by number list of every class"""
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        """Get the number of images"""
        return len(self.labels)

    def get_num_classes(self):
        """get the whole class number"""
        return self.cls_num

    def get_annotations(self):
        """???"""
        annotations = []
        for label in self.labels:
            annotations.append({'category_id': int(label)})
        return annotations

    def get_cls_num_list(self):
        """Get a list containing image numbers of every class"""
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    This Class Object is built for generate an imbalanced CIFAR10 for the official CIFAR10 dataset.
    """
    cls_num = 100
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
