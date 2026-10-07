# encoding: utf-8
"""
Modified UrbanElementsReID dataset to load class labels from train_classes.csv.
Class labels are passed through add_info dict for class-conditional part attention.
"""

import glob
import csv
import re
import torch
import xml.dom.minidom as XD
import os.path as osp
import xml.etree.ElementTree as ET

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

# Class name -> integer ID mapping
CLASS_NAME_TO_ID = {
    'Crosswalk': 0,
    'Container': 1,
    'RubbishBins': 2,
    'TrafficSign': 3,
}

# maps object category name -> integer class id used for query conditioning (AttQueryCond)
_CLASS_MAP = {'Crosswalk': 0, 'Container': 1, 'Trashbin': 2, 'Trafficsign': 3}

@DATASET_REGISTRY.register()
class UrbanElementsReID(ImageDataset):

    def __init__(self, root='/home/jgf/Desktop/rhome/jgf/baselineChallenge/UrbanElementsReID',
                 verbose=True, **kwargs):
        self.dataset_dir = root

        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir, 'image_train/')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_train/')

        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True, include_class=True)
        query = self._process_dir(self.query_dir, relabel=False, include_class=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False, include_class=False)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.train_class_map = self._build_class_map()

        super(UrbanElementsReID, self).__init__(self.train, self.query, self.gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _readCSV_(self, csv_dir):
        camids = []
        imageNames = []
        pids = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
                pids.append(int(row[2]))
        return list(zip(camids, imageNames, pids))

    def _readCSV_with_class_(self, csv_dir):
        """Read CSV with class labels (train_classes.csv format):
        cameraID, imageName, objectID, Class
        """
        camids = []
        imageNames = []
        pids = []
        class_names = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # skip header
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
                pids.append(int(row[2]))
                class_names.append(row[3].strip())
        return list(zip(camids, imageNames, pids, class_names))

    def _readCSV_eval_(self, csv_dir):
        camids = []
        imageNames = []
        pids = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
                pids.append(-1)
        return list(zip(camids, imageNames, pids))
    
    def _build_class_map(self):
        csv_dir = osp.join(self.dataset_dir, 'train_classes.csv')
        class_map = {}
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                img_path = osp.join(self.train_dir, str(row[1]))
                class_map[img_path] = row[3]  # 'Class' column
        return class_map


    def _process_dir(self, dir_path, relabel=False, include_class=False):
        """Process directory.
        
        Args:
            dir_path: path to images
            relabel: whether to relabel pids
            include_class: if True, read train_classes.csv and return 4-tuples 
                          with add_info dict. If False, return standard 3-tuples.
        """
        if include_class:
            classes_csv = osp.join(self.dataset_dir, 'train_classes.csv')
            if osp.exists(classes_csv):
                print(f"[UrbanElementsReID] Loading with class labels from {classes_csv}")
                xml_file = self._readCSV_with_class_(classes_csv)
                has_classes = True
            else:
                print(f"[UrbanElementsReID] No train_classes.csv found, falling back to train.csv")
                xml_file = self._readCSV_(osp.join(self.dataset_dir, 'train.csv'))
                has_classes = False
        else:
            xml_file = self._readCSV_(osp.join(self.dataset_dir, 'train.csv'))
            has_classes = False

        pid_container = set()
        for item in xml_file:
            pid = item[2]
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for item in xml_file:
            if has_classes:
                camid_str, imageName, pid, class_name = item
            else:
                camid_str, imageName, pid = item
                class_name = None

            camid = int(camid_str[1:]) - 1
            if pid == -1:
                continue
            if relabel:
                pid = pid2label[pid]

            img_path = osp.join(dir_path, imageName)

            if include_class and has_classes:
                # Training: return 4-tuple with class info
                class_id = CLASS_NAME_TO_ID.get(class_name, -1)
                add_info = {'class_id': class_id,
                            'cond_class_id': _CLASS_MAP.get(class_name, 0) if class_name is not None else 0}
                dataset.append((img_path, pid, camid, add_info))
            else:
                # Query/Gallery: return standard 3-tuple
                dataset.append((img_path, pid, camid))

        return dataset

    def _process_dir_test(self, dir_path, relabel=False, query=True):
        dataset = []
        if query:
            xml_dir = osp.join(self.dataset_dir_test, 'query.csv')
        else:
            xml_dir = osp.join(self.dataset_dir_test, 'test.csv')

        xml_file = self._readCSV_eval_(xml_dir)
        for camid, imageName, pid in xml_file:
            camid = int(camid[1:]) - 1
            dataset.append((osp.join(dir_path, imageName), pid, camid))
        return dataset

    def _process_track(self, path):
        file = open(path)
        tracklet = dict()
        frame2trackID = dict()
        nums = []
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            tracklet[track_id] = curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
        return tracklet, nums, frame2trackID

    def _process_dir_testVeri(self, dir_path, relabel=False):
        dataset = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d\d\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset