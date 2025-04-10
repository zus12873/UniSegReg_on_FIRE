import os
import cv2
import glob
import shutil
import torch
import numpy as np
from scipy import ndimage
from PIL import Image
import tqdm

from ..utils import eval_functions as EF
from ..utils.process_methods import OneHot

ranking = []
dice_with_gt = []

class QualityAssessment:
    def __init__(self, model_seg, model_reg):
        self.model_seg = model_seg
        self.model_reg = model_reg


    def assessment(self, dataset, 
                target_path, 
                img_suffix, 
                lab_suffix, 
                weight_suffix,
                n_classes, 
                threshold=0.6, 
                threshold_min = 0.2,
                min_good=4,
                combine_method=1,
                target_key=None):
        ndata = len(dataset)
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        # m = ndata // batch_size
        # n = ndata % batch_size
        dices = []
        fpaths = []
        if not os.path.exists(target_path + '/tmp'):
            os.makedirs(target_path + '/tmp')
        if not os.path.exists(target_path + '/good'):
            os.makedirs(target_path + '/good')
        if not os.path.exists(target_path + '/bad'):
            os.makedirs(target_path + '/bad')
        for i in tqdm.tqdm(range(ndata)):
            data_dict = dataset[i]
            results = self.compare(data_dict, n_classes, combine_method, target_key)
            for mi, r in enumerate(results):
                fpath = dataset._data_target[i]
                fname = os.path.basename(fpath)
                tmp_lab_path = target_path+'/tmp/'+fname.replace(img_suffix, lab_suffix)
                tmp_weight_path = target_path+'/tmp/'+fname.replace(img_suffix, weight_suffix)
                # Image.fromarray((r[1] * 255).astype(np.uint8)).save(tmp_lab_path)
                Image.fromarray((r[2] * 255).astype(np.uint8)).save(tmp_weight_path)
                np.save(tmp_lab_path, r[1])
                # np.save(tmp_weight_path, r[2])

                dices.append(r[0])
                fpaths.append([fpath, tmp_lab_path, tmp_weight_path])
        sort_idx = np.argsort(dices)[::-1]
        count_good = 0
        for si in sort_idx:
            fpath = fpaths[si]
            if dices[si] > threshold_min and (count_good < min_good or dices[si] > threshold):
                count_good += 1
                temp_target_path = target_path+'/good'
            else:
                temp_target_path = target_path+'/bad'
            for p in fpath:
                if os.path.exists(p):
                    shutil.copy2(p, temp_target_path)
        shutil.rmtree(target_path + '/tmp')
        return count_good





    def compare(self, data_dict, n_classes, combine_method, target_key):
        unseen_dict = data_dict['data_dict']
        source_dicts = data_dict['assessment_dicts']
        one_hot = OneHot(n_classes)

        # images from models (batch size 1)
        prob_segs = [self.model_seg.predict(unseen_dict)]
        org = unseen_dict[target_key+self.img_suffix].numpy()
        for i in range(len(source_dicts)-1):
            random_aug = RandomAug2D(org.shape[2:])
            tmp_org = random_aug.aug(org)
            tmp_dict = {target_key+self.img_suffix: torch.Tensor(tmp_org.copy())}
            prob_seg_ = self.model_seg.predict(tmp_dict)
            prob_seg = random_aug.reverse(prob_seg_)
            prob_segs.append(prob_seg)

        prob_regs = []
        for s_dict in source_dicts:
            tmp_dict = {**unseen_dict, **s_dict}
            prob_reg = self.model_reg.predict(tmp_dict)
            prob_reg = np.array([one_hot(np.argmax(x, 0)) for x in prob_reg])
            prob_regs.append(prob_reg)

        
        dice = [1]
        combined = np.clip(np.mean(prob_segs + prob_regs, 0), 0, 1)
        weight_map = combined[:, 1, ...]

        results = np.array([[dice[i], combined[i], weight_map[i]] for i in range(len(dice))], dtype=object)
        return results



    def combine_union(self, pred_reg, pred_seg):
        return (np.argmax(pred_reg, 1) + np.argmax(pred_seg, 1)) > 0
    def combine_intersection(self, pred_reg, pred_seg):
        x = np.argmax(pred_reg * pred_seg, 1)
        return x
    def combine_weighted_map(self, pred_reg, pred_seg):
        # weight_map = np.sum(pred_reg * pred_seg, 1)
        weight_map = (np.argmax(pred_reg, 1) + np.argmax(pred_seg, 1)) / 2
        return weight_map


    def img_blend(self, img, lab, colormap=cv2.COLORMAP_JET, weight=[0.7,0.3]):
        # if img.ndim == 2:
        #     img = np.stack((img,)*3, axis=-1)
        clab = np.float32(lab)
        if lab.ndim == 1:
            lab[0,0] = 1
            lab = lab / np.max(lab)
            lab = np.array(lab * 255, np.uint8)
            clab = (cv2.applyColorMap(lab, colormap) / 255).astype(np.float32)
            clab[lab==0] = 0
        comb = cv2.addWeighted(img, weight[0], clab, weight[1], 0)
        comb -= np.min(comb)
        comb /= np.max(comb)
        return comb


class RandomAug2D: 
    def __init__(self, shape):
        methods = [Rotate(np.random.randint(-20, 20)), 
                #    Flip(np.random.choice([2,3])), 
                   Shift(np.random.randint(-shape[0]/10, shape[0]/10), np.random.randint(-shape[1]/10, shape[1]/10))]
        self.random_method = np.random.choice(methods)
    
    def aug(self, img):
        img_ = self.random_method.aug(img)
        return img_
    
    def reverse(self, img):
        img_ = self.random_method.reverse(img)
        return img_

class Rotate:
    def __init__(self, angle, ax=[2,3]):
        self.angle = angle
        self.ax = ax
    def aug(self, img):
        img_ = ndimage.rotate(img, self.angle, self.ax, reshape=False)
        return img_
    def reverse(self, img):
        img_ = ndimage.rotate(img, -self.angle, self.ax, reshape=False)
        return img_

class Flip:
    def __init__(self, ax=2):
        self.ax = ax
    def aug(self, img):
        img_ = np.flip(img, self.ax)
        return img_
    def reverse(self, img):
        img_ = np.flip(img, self.ax)
        return img_
    
class Shift:
    def __init__(self, shift_x, shift_y):
        self.shift_x = shift_x
        self.shift_y = shift_y
    def aug(self, img):
        img_ = np.zeros_like(img)
        img_ = ndimage.shift(img, (0, 0, self.shift_x, self.shift_y), mode='constant', cval=0)
        return img_
    def reverse(self, img):
        img_ = np.zeros_like(img)
        img_[:,0,...] = ndimage.shift(img[:,0,...], (0, -self.shift_x, -self.shift_y), mode='constant', cval=1)
        img_[:,1:,...] = ndimage.shift(img[:,1:,...], (0, 0, -self.shift_x, -self.shift_y), mode='constant', cval=0)
        return img_
    