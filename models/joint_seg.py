import numpy as np

import torch
import torch.nn.functional as F

from snmi.utils import eval_functions as EF, loss_fuctions as LF, utils as U
from snmi.models import BaseModel


class JointSegModel(BaseModel):

    def __init__(self, net, optimizer, img_suffix, lab_suffix, target_key, dropout_rate, 
                 weight_suffix=None,
                loss_functions={LF.CrossEntropy(): 0.5, LF.SoftDice(): 0.5}, 
                device=None):
        super().__init__(net, optimizer, device)
        self.img_suffix = target_key + img_suffix
        self.lab_suffix = target_key + lab_suffix
        self.weight_suffix = target_key + weight_suffix if weight_suffix is not None else None
        self.dropout_rate = dropout_rate
        self.loss_function = loss_functions

    def predict(self, batch):
        img = batch[self.img_suffix]
        
        # 使用修复函数处理张量格式
        from fix_tensor_format import fix_tensor_format
        img = fix_tensor_format(img)
        
        img = img.to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(img)
        pred = F.softmax(logits, dim=1)
        pred = pred.cpu().numpy()
        return pred

    def train_step(self, batch, epoch):
        img = batch[self.img_suffix]
        
        # 使用修复函数处理张量格式
        from fix_tensor_format import fix_tensor_format
        img = fix_tensor_format(img)
        
        img = img.to(device=self.device)

        self.net.train()
        self.optimizer.zero_grad()

        logits = self.net(img, self.dropout_rate)
        loss = self.get_loss(batch, logits)   # calculate loss
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().numpy() # return loss for logging

    def get_loss(self, batch, logits):
        lab = batch[self.lab_suffix].to(device=self.device)
        data_dict = {'logits': logits, 'target': lab}
        if self.weight_suffix is not None:
            weight_map = batch[self.weight_suffix].to(device=self.device)
            data_dict.update({'weight': weight_map})
        

        loss = 0
        for lf in self.loss_function:
            loss += lf(data_dict) * self.loss_function[lf]

        return loss

    def eval_step(self, batch, **kwargs):
        self.net.eval() # switch to evaluation model

        img = batch[self.img_suffix]
        lab = batch[self.lab_suffix]
        
        # 使用修复函数处理张量格式
        from fix_tensor_format import fix_tensor_format
        img = fix_tensor_format(img)
        
        img_g = img.to(device=self.device)
        lab_g = lab.to(device=self.device)

        with torch.no_grad():
            logits = self.net(img_g)
            loss = self.get_loss(batch, logits).cpu().numpy()

        pred = F.softmax(logits, dim=1)
        pred = pred.argmax(dim=1)
        pred = torch.zeros_like(lab_g).scatter_(1, pred.unsqueeze(1), 1.)
        dice = EF.dice_coefficient(pred, lab_g, ignore_background=True).cpu().numpy()
        hausdorff = EF.hausdorff(pred, lab_g, ignore_background=True).cpu().numpy()
        eval_results = {'loss': loss,
                        'dice': dice,
                        'hausdorff': hausdorff}
        
        log_image = kwargs.get('log_image', False)
        if log_image:
            # tensor to numpy
            
            img = img.numpy()
            lab = np.argmax(lab.numpy(), 1, keepdims=True)
            pred = np.argmax(pred.cpu().numpy(), 1, keepdims=True)
            img_in = [img, lab, pred]
            if self.weight_suffix is not None:
                weight_map = batch[self.weight_suffix].numpy()
                img_in = img_in + [weight_map]
            img = self.get_imgs_eval(img_in) # create drawable image, see more details inside the function
            eval_results.update({'image': img})


        return eval_results
        

    def get_imgs_eval(self, imgs):
        # check the images are 2d or 3d
        if len(np.shape(imgs[0])) == 4:
            return self.get_imgs_eval_2d(imgs)
        elif len(np.shape(imgs[0])) == 5:
            return self.get_imgs_eval_3d(imgs)
        else:
            raise('Error: image dimension should be 4 (2d) or 5 (3d)!')

    def get_imgs_eval_2d(self, imgs):
        img, lab, pred = imgs[:3]
        combine_lab = U.gray_img_blend(lab, pred, channel_first=True)
        img_in = [img, lab, pred, combine_lab]
        if len(imgs) == 4:
            weight = imgs[3]
            weight[:,:,0,0] = 1
            img_in = img_in + [weight]

        img = U.combine_2d_imgs_from_tensor(img_in, channel_first=True)
        img_dict = {'img-lab-pred-combined': img}
        return img_dict

    def get_imgs_eval_3d(self, imgs):
        img, lab, pred = imgs
        img = U.combine_3d_imgs_from_tensor([img, lab, pred], vis_slices=6, channel_first=True)
        img_dict = {'img-lab-pred': img}
        return img_dict

