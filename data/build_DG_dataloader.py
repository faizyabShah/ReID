import os
import logging
import torch
import sys
import collections.abc as container_abcs

# from torch._six import container_abcs, string_classes, int_classes
int_classes = int
string_classes = str
from torch.utils.data import DataLoader
from utils import comm
import random

from utils.class_split import attach_class_info, normalize_class_name, read_class_map, split_records_by_class

from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms, build_class_specific_transforms

_root = os.getenv("REID_DATASETS", "../../data")


def _class_mode_to_filter(class_mode):
    if class_mode is None:
        return None
    mode = normalize_class_name(class_mode)
    if mode in {"all", "any", "none", ""}:
        return None
    if mode in {"traffic", "nontraffic", "other", "non_traffic"}:
        return mode
    return mode


def _matches_class_mode(class_name, class_mode):
    if class_mode is None:
        return True
    normalized = normalize_class_name(class_name)
    if class_mode == "traffic":
        return normalized == "traffic"
    if class_mode in {"nontraffic", "other", "non_traffic"}:
        return normalized != "traffic"
    return True


def build_reid_train_loader(cfg, class_mode=None):
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('*'*100)
        print('Hmm, Big Debugger is watching me')
        print('*'*100)
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS
    include_flip = not cfg.INPUT.CLASS_SPECIFIC_AUG.ENABLED
    train_transforms = build_transforms(cfg, is_train=True, is_fake=False, include_flip=include_flip)
    
    class_global_transform_map = {}
    if cfg.INPUT.CLASS_SPECIFIC_AUG.ENABLED:
        train_transforms_traffic_sign = build_transforms(
            cfg, is_train=True, is_fake=False,
            include_flip=False, skip_cj=True, skip_rea=True
        )
        # raw CSV label → override global transform
        class_global_transform_map['trafficsignal'] = train_transforms_traffic_sign

    # Faiz's class-specific augmentation (faizyab/class-specific-augmentations)
    if cfg.INPUT.ENABLE_CLASS_SPECIFIC_AUG:
        train_transforms = build_class_specific_transforms(cfg, is_train=True, is_fake=False, enable_class_specific=True)

    train_items = list()
    domain_idx = 0
    camera_all = list()
    class_label_map = {}

    _root = cfg.DATASETS.ROOT_DIR
    classes_path = os.path.join(_root, 'train_classes.csv')
    train_classes_map = read_class_map(classes_path)

    # load datasets
    for d in cfg.DATASETS.TRAIN:
        if d == 'CUHK03_NP':
            dataset = DATASET_REGISTRY.get('CUHK03')(root=_root, cuhk03_labeled=False)
        else:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        # Ensure every train item has an add_info dict at index 3 with 'domains' and optional 'class'
        for i, item in enumerate(dataset.train):
            # reuse the add_info dict a dataset may already provide (e.g. 'class_id' from UrbanElementsReID)
            add_info = dict(item[3]) if len(item) >= 4 and isinstance(item[3], dict) else {}
            add_info.setdefault('class_id', -1)
            camera_all.append(dataset.train[i][2])
            if cfg.DATALOADER.CAMERA_TO_DOMAIN:
                add_info['domains'] = dataset.train[i][2]
            else:
                add_info['domains'] = int(domain_idx)

            # attempt to attach class id (if available in train_classes.csv)
            img_path = dataset.train[i][0]
            cls_name = train_classes_map.get(img_path, train_classes_map.get(os.path.basename(img_path)))
            if cls_name is not None:
                add_info['class_name'] = cls_name

            dataset.train[i] = tuple(list(dataset.train[i][:3]) + [add_info])
        if class_mode is not None:
            train_filtered = []
            for item in dataset.train:
                class_name = item[3].get('class_name') if len(item) > 3 else None
                if _matches_class_mode(class_name, class_mode):
                    train_filtered.append(item)
            dataset.train = train_filtered
            dataset.num_train_pids = dataset.get_num_pids(dataset.train)
        if comm.is_main_process():
            dataset.show_train()
        domain_idx += 1
        train_items.extend(dataset.train)
        if hasattr(dataset, 'train_class_map'):
            class_label_map.update(dataset.train_class_map)
        
    class_aug_dict = {}
    if cfg.INPUT.CLASS_SPECIFIC_AUG.ENABLED:
        from .transforms.build import build_class_transforms
        size_train = cfg.INPUT.SIZE_TRAIN
        for raw_label in set(class_label_map.values()):
            aug = build_class_transforms(raw_label, size_train)
            if aug is not None:
                class_aug_dict[raw_label] = aug

        train_set = CommDataset(train_items, train_transforms, relabel=True,
                            class_label_map=class_label_map,
                            class_aug_dict=class_aug_dict,
                            class_global_transform_map=class_global_transform_map)
    else:
        train_set = CommDataset(train_items, train_transforms, relabel=True)
    train_set.num_cams = len(set(camera_all)) if camera_all else 0

    train_loader = make_sampler(
        train_set=train_set,
        num_batch=cfg.SOLVER.IMS_PER_BATCH,
        num_instance=cfg.DATALOADER.NUM_INSTANCE,
        num_workers=num_workers,
        mini_batch_size=cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
        drop_last=cfg.DATALOADER.DROP_LAST,
        flag1=cfg.DATALOADER.NAIVE_WAY,
        flag2=cfg.DATALOADER.DELETE_REM,
        cfg = cfg)

    return train_loader


def build_reid_test_loader(cfg, dataset_name, opt=None, flag_test=True, shuffle=False, only_gallery=False, only_query=False, eval_time=False, class_mode=None):
    test_transforms = build_transforms(cfg, is_train=False)
    _root = cfg.DATASETS.ROOT_DIR
    if opt is None:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            if flag_test:
                dataset.show_test()
            else:
                dataset.show_train()
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=[_root, opt])
    query_class_map = read_class_map(os.path.join(_root, 'query_classes.csv'))
    gallery_class_map = read_class_map(os.path.join(_root, 'test_classes.csv'))

    if flag_test:
        if only_gallery:
            test_items = dataset.gallery
        elif only_query:
            test_set = CommDataset([random.choice(dataset.query)], test_transforms, relabel=False)
            return test_set
        else:
            test_items = dataset.query + dataset.gallery
        if shuffle: # only for visualization
            random.shuffle(test_items)
    else:
        test_items = dataset.train

    # Class annotation / filtering of test items is only used by the class-split experiment
    # (faizyab/traffic-class-things); the default path keeps the plain (img, pid, camid) items.
    if flag_test and _class_mode_to_filter(class_mode) is not None:
        annotated_items = []
        for item in test_items:
            image_path = item[0]
            image_name = os.path.basename(image_path)
            class_name = query_class_map.get(image_name)
            if class_name is None:
                class_name = query_class_map.get(image_path)
            if class_name is None:
                class_name = gallery_class_map.get(image_name)
            if class_name is None:
                class_name = gallery_class_map.get(image_path)
            annotated_items.append(attach_class_info(item, class_name))
        test_items = annotated_items

        filter_mode = _class_mode_to_filter(class_mode)
        if filter_mode is not None:
            if filter_mode == 'traffic':
                test_items = split_records_by_class(test_items, 'traffic')
            elif filter_mode in {'nontraffic', 'non_traffic', 'other'}:
                test_items = [item for item in test_items if item[3].get('class_name') != 'traffic']
            else:
                test_items = [item for item in test_items if _matches_class_mode(item[3].get('class_name'), filter_mode)]
        if comm.is_main_process():
            num_query_items = len(dataset.query)
            num_gallery_items = len(dataset.gallery)
            if only_gallery:
                num_query_items = 0
                num_gallery_items = len(test_items)
            elif only_query:
                num_query_items = len(test_items)
                num_gallery_items = 0
            else:
                if class_mode is not None:
                    num_query_items = len([item for item in test_items if item[3].get('q_or_g') == 'query'])
                    num_gallery_items = len([item for item in test_items if item[3].get('q_or_g') == 'gallery'])
            logger = logging.getLogger('PAT')
            logger.info('=> Loaded {}'.format(dataset.__class__.__name__))
            logger.info('  ----------------------------------------')
            logger.info('  subset   | # ids | # images | # cameras')
            logger.info('  ----------------------------------------')
            logger.info('  query    | {:5d} | {:8d} | {:9d}'.format(len(set([item[1] for item in test_items[:num_query_items]])), num_query_items, len(set([item[2] for item in test_items[:num_query_items]])) if num_query_items else 0))
            logger.info('  gallery  | {:5d} | {:8d} | {:9d}'.format(len(set([item[1] for item in test_items[num_query_items:num_query_items+num_gallery_items]])), num_gallery_items, len(set([item[2] for item in test_items[num_query_items:num_query_items+num_gallery_items]])) if num_gallery_items else 0))
            logger.info('  ----------------------------------------')

    test_set = CommDataset(test_items, test_transforms, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
    elif isinstance(elem, list):
        out_g = []
        out_pt1 = []
        out_pt2 = []
        out_pt3 = []
        # out = torch.stack(elem, dim=0)
        for i, tensor_list in enumerate(batched_inputs):
            out_g.append(tensor_list[0])
            out_pt1.append(tensor_list[1])
            out_pt2.append(tensor_list[2])
            out_pt3.append(tensor_list[3])
        out = torch.stack(out_g, dim=0)
        out_pt1 = torch.stack(out_pt1, dim=0)
        out_pt2 = torch.stack(out_pt2, dim=0)
        out_pt3 = torch.stack(out_pt3, dim=0)
        return out, out_pt1, out_pt2, out_pt3


def make_sampler(train_set, num_batch, num_instance, num_workers,
                 mini_batch_size, drop_last=True, flag1=True, flag2=True, seed=None, cfg=None):
    # Optionally use class-balanced sampling when requested in config.
    if cfg is not None and getattr(cfg.DATALOADER, 'CLASS_BALANCE', False):
        try:
            data_sampler = samplers.ClassBalancedIdentitySampler(train_set.img_items,
                                                                  mini_batch_size, num_instance)
        except Exception:
            # Fallback to random identity sampler if class-balanced cannot be constructed
            data_sampler = samplers.RandomIdentitySampler(train_set.img_items,
                                                          mini_batch_size, num_instance)
    else:
        if flag1:
            data_sampler = samplers.RandomIdentitySampler(train_set.img_items,
                                                          mini_batch_size, num_instance)
        else:
            data_sampler = samplers.DomainSuffleSampler(train_set.img_items,
                                                         num_batch, num_instance, flag2, seed, cfg)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader