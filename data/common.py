import torch
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True, class_label_map=None, class_aug_dict=None, class_global_transform_map=None):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.class_label_map = class_label_map or {}
        self.class_aug_dict = class_aug_dict or {}
        self.class_global_transform_map = class_global_transform_map or {}
        

        self.pid_dict = {}
        if self.relabel:
            pids = list()
            for i, item in enumerate(img_items):
                if item[1] in pids: continue
                pids.append(item[1])
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        if len(self.img_items[index]) > 3:
            img_path, pid, camid, others = self.img_items[index]
        else:
            img_path, pid, camid = self.img_items[index]
            others = ''
        img = read_image(img_path)
        cls = self.class_label_map.get(img_path)
        global_t = self.class_global_transform_map.get(cls, self.transform) if self.class_global_transform_map else self.transform
        if global_t is not None:
            img = global_t(img)

        if self.class_aug_dict:
            aug = self.class_aug_dict.get(cls) if cls else None
            if aug is not None:
                img = aug(img)

        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "others": others
        }

    @property
    def num_classes(self):
        return len(self.pids)
