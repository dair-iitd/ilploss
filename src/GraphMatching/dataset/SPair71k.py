import glob
import json
import os
import pickle
import random
from collections import defaultdict
import numpy as np
from PIL import Image

from src.GraphMatching.utils.config import cfg

from IPython.core.debugger import Pdb

cache_path = cfg.CACHE_PATH
pair_ann_path = cfg.SPair.ROOT_DIR + "/PairAnnotation"
layout_path = cfg.SPair.ROOT_DIR + "/Layout"
image_path = cfg.SPair.ROOT_DIR + "/JPEGImages"
dataset_size = cfg.SPair.size

sets_translation_dict = dict(train="trn", test="test",val="val")
difficulty_params_dict = dict(
    trn=cfg.TRAIN.difficulty_params, val=cfg.EVAL.difficulty_params, test=cfg.EVAL.difficulty_params
)


class SPair71k:
    def __init__(self, sets, obj_resize, num_keypoints = 0):
        """
        :param sets: 'train' or 'test'? or 'val'??
        :param obj_resize: resized object size
        """
        self.sets = sets_translation_dict[sets]
        layout_file = os.path.join(layout_path, dataset_size, self.sets + ".txt")
        print("Load from: ", layout_file)
        self.ann_files = open(layout_file, "r").read().split("\n")
        self.ann_files = self.ann_files[: len(self.ann_files) - 1]
        self.difficulty_params = difficulty_params_dict[self.sets]
        self.num_keypoints = num_keypoints
        #if 'num_keypoints' in cfg and cfg.num_keypoints > 0:
        #    self.difficulty_params['num_keypoints'] = cfg.num_keypoints
            
        self.pair_ann_path = pair_ann_path
        self.image_path = image_path
        self.classes = list(map(lambda x: os.path.basename(x), glob.glob("%s/*" % image_path)))
        self.classes.sort()
        self.obj_resize = obj_resize
        self.combine_classes = cfg.combine_classes
        filter_results = self.filter_annotations(self.ann_files, self.difficulty_params)

        self.ann_files_filtered = filter_results[0]
        self.ann_files_filtered_cls_dict = filter_results[1]
        self.ann_files_filtered_cls_dict_idx = filter_results[2]
        self.classes = filter_results[3]

        self.total_size = len(self.ann_files_filtered)
        self.size_by_cls = {cls: len(ann_list) for cls, ann_list in self.ann_files_filtered_cls_dict.items()}


    def load_json(self,ann_file_handle):
        annotation = json.load(ann_file_handle)
        annotation['num_keypoints'] = len(annotation["src_kps"])
        return annotation


    def filter_annotations(self, ann_files, difficulty_params):
        if len(difficulty_params) > 0:
            basepath = os.path.join(self.pair_ann_path, "pickled", self.sets)
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            difficulty_paramas_str = self.diff_dict_to_str(difficulty_params)
            try:
                filepath = os.path.join(basepath, difficulty_paramas_str + ".pickle")
                ann_files_filtered = pickle.load(open(filepath, "rb"))
                print(
                    f"Found filtered annotations for difficulty parameters {difficulty_params} and {self.sets}-set at {filepath}"
                )
            except (OSError, IOError) as e:
                print(
                    f"No pickled annotations found for difficulty parameters {difficulty_params} and {self.sets}-set. Filtering..."
                )
                ann_files_filtered_dict = {}

                for ann_file in ann_files:
                    with open(os.path.join(self.pair_ann_path, self.sets, ann_file + ".json")) as f:
                        annotation = self.load_json(f)
                        #annotation = json.load(f)
                        #annotation['num_keypoints'] = len(annotation["src_kps"])
                    #
                    diff = {key: annotation[key] for key in self.difficulty_params.keys()}
                    diff_str = self.diff_dict_to_str(diff)
                    if diff_str in ann_files_filtered_dict:
                        ann_files_filtered_dict[diff_str].append(ann_file)
                    else:
                        ann_files_filtered_dict[diff_str] = [ann_file]
                total_l = 0
                for diff_str, file_list in ann_files_filtered_dict.items():
                    total_l += len(file_list)
                    filepath = os.path.join(basepath, diff_str + ".pickle")
                    pickle.dump(file_list, open(filepath, "wb"))
                assert total_l == len(ann_files)
                print(f"Done filtering. Saved filtered annotations to {basepath}.")
                ann_files_filtered = ann_files_filtered_dict[difficulty_paramas_str]
        else:
            print(f"No difficulty parameters for {self.sets}-set. Using all available data.")
            ann_files_filtered = ann_files

        ann_files_filtered_cls_dict = defaultdict(list)
        ann_files_filtered_cls_dict_idx = defaultdict(list)
        for _i,_x in enumerate(ann_files_filtered):
            this_class = _x.split(':')[-1]
            if this_class in self.classes:
                ann_files_filtered_cls_dict[this_class].append(_x)
                ann_files_filtered_cls_dict_idx[this_class].append(_i)
            else:
                print(this_class, 'missing. why?')
        #ann_files_filtered_cls_dict = {
        #    cls: list(filter(lambda x: cls in x, ann_files_filtered)) for cls in self.classes
        #}
        class_len = {cls: len(ann_list) for cls, ann_list in ann_files_filtered_cls_dict.items()}
        print(f"Number of annotation pairs matching the difficulty params in {self.sets}-set: {class_len}")
        if self.combine_classes:
            cls_name = "combined"
            ann_files_filtered_cls_dict = {cls_name: ann_files_filtered}
            ann_files_filtered_cls_dict_idx = {cls_name: list(range(len(ann_files_filtered)))}
            filtered_classes = [cls_name]
            print(f"Combining {self.sets}-set classes. Total of {len(ann_files_filtered)} image pairs used.")
        else:
            filtered_classes = []
            for cls, ann_f in ann_files_filtered_cls_dict.items():
                if len(ann_f) > 0:
                    filtered_classes.append(cls)
                else:
                    print(f"Excluding class {cls} from {self.sets}-set.")
        return ann_files_filtered, ann_files_filtered_cls_dict, ann_files_filtered_cls_dict_idx, filtered_classes

    def diff_dict_to_str(self, diff):
        diff_str = ""
        keys = ["mirror", "viewpoint_variation", "scale_variation", "truncation", "occlusion",'num_keypoints']
        for key in keys:
            if key in diff.keys():
                diff_str += key
                diff_str += str(diff[key])
        return diff_str

    def get_k_samples(self, idx, k, mode, cls=None, shuffle=True):
        """
        Randomly get a sample of k objects from VOC-Berkeley keypoints dataset
        :param idx: Index of datapoint to sample, None for random sampling
        :param k: number of datapoints in sample
        :param mode: sampling strategy
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (k samples of data, k \choose 2 groundtruth permutation matrices)
        """
        if k != 2:
            raise NotImplementedError(
                f"No strategy implemented to sample {k} graphs from SPair dataset. So far only k=2 is possible."
            )
        
        if cls is None:
            if idx is None:
                cls = self.classes[random.randrange(0, len(self.classes))]
                ann_files = self.ann_files_filtered_cls_dict[cls]
            else:
                ann_files = self.ann_files_filtered
        elif type(cls) == int:
            cls = self.classes[cls]
            ann_files = self.ann_files_filtered_cls_dict[cls]
        else:
            assert type(cls) == str
            ann_files = self.ann_files_filtered_cls_dict[cls]

        # get pre-processed images

        assert len(ann_files) > 0
        if idx is None:
            idx = random.randrange(len(ann_files))
            ann_file = ann_files[idx] + ".json"
            return_idx = self.ann_files_filtered_cls_dict_idx[cls][idx]
        else:
            ann_file = ann_files[idx] + ".json"
            if cls is None:
                return_idx = idx
            else:
                return_idx = self.ann_files_filtered_cls_dict_idx[cls][idx]
        with open(os.path.join(self.pair_ann_path, self.sets, ann_file)) as f:
            annotation = self.load_json(f)
            #annotation = json.load(f)

        category = annotation["category"]
        if cls is not None and not self.combine_classes:
            assert cls == category
        assert all(annotation[key] == value for key, value in self.difficulty_params.items())

        if mode == "intersection":
            assert len(annotation["src_kps"]) == len(annotation["trg_kps"])
            num_kps = len(annotation["src_kps"])
            selected_kps = list(range(num_kps))
            if self.num_keypoints > 0 and num_kps > self.num_keypoints:
                #randomly select self.num_keypoints from num_kps
                selected_kps = random.sample(range(num_kps), self.num_keypoints)
                for key in ['src_kps','trg_kps','kps_ids']:
                    annotation[key] = [annotation[key][i_] for i_ in selected_kps]
                annotation['num_keypoints'] = len(selected_kps)
                num_kps = len(annotation["src_kps"])
            
            perm_mat_init = np.eye(num_kps)
            anno_list = []
            perm_list = []

            for st in ("src", "trg"):
                if shuffle:
                    perm = np.random.permutation(np.arange(num_kps))
                else:
                    perm = np.arange(num_kps)
                kps = annotation[f"{st}_kps"]
                img_path = os.path.join(self.image_path, category, annotation[f"{st}_imname"])
                img, kps = self.rescale_im_and_kps(img_path, kps)
                kps_permuted = [kps[i] for i in perm]
                anno_dict = dict(image=img, keypoints=kps_permuted)
                anno_list.append(anno_dict)
                perm_list.append(perm)

            perm_mat = perm_mat_init[perm_list[0]][:, perm_list[1]]
        else:
            raise NotImplementedError(f"Unknown sampling strategy {mode}")

        return return_idx, anno_list, [perm_mat], perm_list[0], perm_list[1], selected_kps

    def rescale_im_and_kps(self, img_path, kps):

        with Image.open(str(img_path)) as img:
            w, h = img.size
            img = img.resize(self.obj_resize, resample=Image.BICUBIC)

        keypoint_list = []
        for kp in kps:
            x = kp[0] * self.obj_resize[0] / w
            y = kp[1] * self.obj_resize[1] / h
            keypoint_list.append(dict(x=x, y=y))

        return img, keypoint_list


if __name__ == "__main__":
    trn_dataset = SPair71k("train", (256, 256))
