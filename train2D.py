import glob
import scipy.io as sio
import numpy as np
import shutil
import os

import torch
from torch.utils.data.dataloader import DataLoader

from snmi.core import Trainer, BaseDataset
from snmi._custom import SegRegDataset
from snmi.utils import augmentation_methods as A, loss_fuctions as LF, process_methods as P, utils as U

from snmi.nets.net_reg import MrReg
from snmi.nets.unet2D import Unet2D
from snmi._custom.quality_assessment_float import QualityAssessment
from models import JointSegModel, JointRegModel

import sys
from fix_tensor_format import fix_tensor_format  # 导入修复函数

# Set up paths for dataset
PATH_TO_2D_DATASET = "FIRE"  # Replace with actual path to 2D dataset


class cfg:
    data_path = PATH_TO_2D_DATASET  # Brain MRI 2D dataset
    output_path = './results_2D'

    test_only = False
    test_pre_seg = True
    test_pre_reg = False
    test_seg = True
    test_reg = False
    # pre_training parameters
    pre_train_seg = True
    pre_train_reg = False
    pre_seg_ep = 50
    pre_seg_lr = 1e-4
    pre_seg_eval_frequency = 5
    pre_seg_loss = {LF.SoftDice(): 0.5, LF.CrossEntropy(): 0.5}

    pre_reg_ep = 200
    pre_reg_lr = 1e-3
    pre_reg_eval_frequency = 10
    pre_reg_loss = {LF.GNCC(): 1} 
    pre_disp_loss = {LF.Grad('l2'): [128, 64, 32, 16, 8]}

    # training parameters
    train_batch_size = 5
    eval_batch_size = 5
    eval_frequency = 10

    # iterative
    iters_start = 0
    iters = 10
    min_good = 37
    threshold_dice = 0.9
    threshold_dice_min = 0.2
    num_source = 150
    num_assessment = 5


    # segmentation setting
    seg_ep = 100
    seg_lr = 1e-5
    seg_loss = {LF.SoftDice(): 0.5, LF.CrossEntropy(): 0.5}
    seg_layers = 5
    seg_dropout_rate = 0.2
    
    # registration setting
    reg_ep = 50
    reg_lr = 1e-4
    freeze_reg = False
    
    reg_loss = {LF.GNCC(): [1, 1, 1, 1, 1]} 
    disp_loss = {LF.Grad('l2'): [128, 64, 32, 16, 8]}
    lab_loss = {LF.RegDice(weight=False): 1}
    # lab_loss = {LF.RegCrossEntropy(): 1}
    reg_layers = 5
    int_steps = 0
    
    # training setttings
    log_train_image = False
    seg_log_validation_image = False
    reg_log_validation_image = True
    save_frequency=None
    val_loss_key='loss' # key of evaluation result to control early stop and best ckpt save
    early_stop_patience=None
    learning_rate_schedule=None

    # set the suffix for data loading
    img_suffix = 'org.tif'
    lab_suffix = 'lab_b.npy'
    weight_suffix = 'weight.tif'
    source_key = 'source'
    target_key = 'target'

    pre = {img_suffix: [P.min_max, P.ToTensor(0)],
        lab_suffix: [P.OneHot([0,255])],
        weight_suffix: [lambda img: np.array(img > 0.5, np.float32), P.ToTensor(0)]}

# Data
train = glob.glob(cfg.data_path + '/train/*org.tif')
valid = glob.glob(cfg.data_path + '/valid/*org.tif')
test = glob.glob(cfg.data_path + '/test/*org.tif')
source_test = glob.glob(cfg.data_path + '/source_test/*org.tif')


# sys.exit()

tmp_random = np.random.RandomState(123)
tmp_random.shuffle(train)
tmp_random.shuffle(valid)
tmp_random.shuffle(test)

source = train[:cfg.num_source]
# source check
source_check_path = cfg.output_path + '/source_start'
if not os.path.exists(source_check_path):
    os.makedirs(source_check_path)
[shutil.copy2(x, source_check_path) for x in source]
[shutil.copy2(x.replace(cfg.img_suffix, cfg.lab_suffix), source_check_path) for x in source]

target_train = train[cfg.num_source:]
target_valid = valid
target_test = test

# Network
net_seg = Unet2D(n_classes=2, in_channels=1, n_layers=cfg.seg_layers)
net_reg = MrReg(n_layers=cfg.reg_layers, int_steps=cfg.int_steps)

# Optimizer
pre_opt_seg = torch.optim.Adam(net_seg.parameters(), lr=cfg.pre_seg_lr)
pre_opt_reg = torch.optim.Adam(net_reg.parameters(), lr=cfg.pre_reg_lr)
opt_seg = torch.optim.Adam(net_seg.parameters(), lr=cfg.seg_lr)
opt_reg = torch.optim.Adam(net_reg.parameters(), lr=cfg.reg_lr)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pre train
# aug = A.RandomAugmentation(cfg.target_key+cfg.img_suffix, cfg.target_key+cfg.lab_suffix)
aug = None
seg_train_set = SegRegDataset([], source, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre, aug)
seg_train_loader = DataLoader(seg_train_set, batch_size=cfg.train_batch_size, shuffle=True)

reg_target_list = train.copy()
np.random.shuffle(reg_target_list)
reg_train_set = SegRegDataset(train, reg_target_list, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
reg_train_loader = DataLoader(reg_train_set, batch_size=cfg.train_batch_size, shuffle=True)

valid_set = SegRegDataset(source, target_valid, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
valid_loader = DataLoader(valid_set, batch_size=cfg.eval_batch_size)

pre_model_seg = JointSegModel(net_seg, pre_opt_seg, cfg.img_suffix, cfg.lab_suffix, cfg.target_key, weight_suffix=cfg.weight_suffix,
                     dropout_rate=cfg.seg_dropout_rate, loss_functions=cfg.pre_seg_loss, device=device)
pre_model_reg = JointRegModel(net_reg, pre_opt_reg, cfg.img_suffix, cfg.source_key, cfg.target_key, lab_suffix=cfg.lab_suffix, weight_suffix=cfg.weight_suffix,
                    loss_functions=cfg.reg_loss, disp_loss_functions=cfg.disp_loss, lab_loss_functions=None, device=device)

pre_trainer_seg = Trainer(pre_model_seg)
pre_trainer_reg = Trainer(pre_model_reg)

if not cfg.test_only :
    if cfg.pre_train_seg:
        pre_trainer_seg.train(seg_train_loader, valid_loader, 
            epochs=cfg.pre_seg_ep, 
            output_path=cfg.output_path + '/pre_seg', 
            log_train_image=True, 
            log_validation_image=cfg.seg_log_validation_image,
            eval_frequency=cfg.pre_seg_eval_frequency,
            print_tag=f'pre Seg - ')
    # sys.exit()
    if cfg.pre_train_reg:
        pre_trainer_reg.cur_epoch = 0
        pre_trainer_reg.train(reg_train_loader, valid_loader, 
            epochs=cfg.pre_reg_ep, 
            output_path=cfg.output_path + '/pre_reg', 
            log_train_image=cfg.log_train_image, 
            log_validation_image=cfg.reg_log_validation_image,
            eval_frequency=cfg.pre_reg_eval_frequency,
            print_tag=f'pre Reg - ')

# Joint Model and Trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_seg = JointSegModel(net_seg, opt_seg, cfg.img_suffix, cfg.lab_suffix, cfg.target_key, weight_suffix=cfg.weight_suffix,
                     dropout_rate=cfg.seg_dropout_rate, loss_functions=cfg.seg_loss, device=device)
model_reg = JointRegModel(net_reg, opt_reg, cfg.img_suffix, cfg.source_key, cfg.target_key, lab_suffix=cfg.lab_suffix, weight_suffix=cfg.weight_suffix,
                    loss_functions=cfg.reg_loss, disp_loss_functions=cfg.disp_loss, lab_loss_functions=cfg.lab_loss, device=device)

trainer_seg = Trainer(model_seg)
trainer_reg = Trainer(model_reg)

# Assessment initialization
count_good = 0
assessment = QualityAssessment(model_seg, model_reg)
ass_pre = {cfg.img_suffix: [P.min_max, P.ToTensor(0)],
            cfg.lab_suffix: [P.ToTensor(0)]}
target_set = SegRegDataset(source, target_train, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix], ass_pre, is_assessment=True, num_assessment=cfg.num_assessment)
if cfg.iters_start == 0:
    # Restore pre-trained model
    if cfg.pre_train_seg:
        ckpt_seg = torch.load(cfg.output_path + '/pre_seg/ckpt/model_best.pt')
    else:
        ckpt_seg = torch.load(cfg.output_path + '/../pre_seg/ckpt/model_final.pt')

    if cfg.pre_train_reg:
        ckpt_reg = torch.load(cfg.output_path + '/pre_reg/ckpt/model_final.pt')     
    else:
        ckpt_reg = torch.load(cfg.output_path + '/../pre_reg/ckpt/model_final.pt')
    net_seg.load_state_dict(ckpt_seg['net0'])
    net_reg.load_state_dict(ckpt_reg['net0'])
    # freeze encoder and first 3 decoder of registration model
    if cfg.freeze_reg:
        net_reg.enc.requires_grad_(False)
        # for fi in range(3):
        #     net_reg.dec.dec_list[fi].requires_grad_(False)
        for name, param in net_reg.named_parameters():
            print(name,param.requires_grad)

    # pre test
    print('pre test')
    with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'Baseline: \n')
    if cfg.pre_train_seg or cfg.test_pre_seg:
        seg_test_set = SegRegDataset([], target_test, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        seg_test_loader = DataLoader(seg_test_set, batch_size=cfg.eval_batch_size)
        seg_results = trainer_seg.test(seg_test_loader, cfg.output_path + '/pre_seg', desc=f'Pre Test Seg: ', log_image=cfg.seg_log_validation_image)
        sio.savemat(f'{cfg.output_path}/pre_seg/test_results.mat', seg_results)
        with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'  Seg - {U.dict_to_str(seg_results)}\n')
    if cfg.pre_train_reg or cfg.test_pre_reg:
        reg_test_set = SegRegDataset(sorted(source_test), sorted(target_test*len(source_test)), cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        reg_test_loader = DataLoader(reg_test_set, batch_size=len(source_test))
        reg_results = trainer_reg.test(reg_test_loader, cfg.output_path + '/pre_reg', desc=f'Pre Test Reg: ', log_image=cfg.reg_log_validation_image)
        sio.savemat(f'{cfg.output_path}/pre_reg/test_results.mat', reg_results)
        with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'  Reg - {U.dict_to_str(reg_results)}\n')
    seg_results = None 
    reg_results = None

    # generate assessed data
    if not cfg.test_only:
        count_good = assessment.assessment(target_set, cfg.output_path+'/assessed_data', 
                                cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix,
                                [0,1], cfg.threshold_dice, cfg.threshold_dice_min, cfg.min_good, target_key=cfg.target_key)
        with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'  count good - {count_good}/{len(target_train)}\n\n')
        count_good = 0

# reg_test_set = SegRegDataset(sorted(source_test), sorted(target_test*len(source_test)), cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
# reg_test_loader = DataLoader(reg_test_set, batch_size=len(source_test))
# reg_results = trainer_reg.test(reg_test_loader, cfg.output_path + '/../pre_reg', desc=f'Pre Test Reg: ', log_image=cfg.reg_log_validation_image)
# sio.savemat(f'{cfg.output_path}/pre_reg/test_results.mat', reg_results)
# with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
#     f.write(f'  Reg - {U.dict_to_str(reg_results)}\n')

# return 0
# Iterative training
for it_idx in range(cfg.iters_start, cfg.iters):
    # Path
    it_output_path = f'{cfg.output_path}/iter-{it_idx}'
    it_path_seg = f'{it_output_path}/seg'
    it_path_reg = f'{it_output_path}/reg'

    # iter restore
    if it_idx == 0:
        it_target_list = glob.glob(f'{cfg.output_path}/assessed_data/good/*{cfg.img_suffix}')
    else:
        it_target_list = glob.glob(f'{cfg.output_path}/iter-{it_idx-1}/assessed_data/good/*{cfg.img_suffix}')
        trainer_seg.restore(f'{cfg.output_path}/iter-{it_idx-1}/seg/ckpt/model_best.pt')
        trainer_reg.restore(f'{cfg.output_path}/iter-{it_idx-1}/reg/ckpt/model_final.pt')
        # trainer_seg.restore(f'{cfg.output_path}/iter-{it_idx-1}/seg/ckpt/model_best.pt')
        # trainer_reg.restore(f'{cfg.output_path}/iter-{it_idx-1}/reg/ckpt/model_best.pt')

    # Train data loader
    seg_train_list = source + it_target_list
    seg_train_set = SegRegDataset([], seg_train_list, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
    seg_train_loader = DataLoader(seg_train_set, batch_size=cfg.train_batch_size, shuffle=True)
    
    reg_train_list = source + it_target_list
    reg_target_list = reg_train_list.copy()
    np.random.shuffle(reg_target_list)
    reg_train_set = SegRegDataset(reg_train_list, reg_target_list, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
    reg_train_loader = DataLoader(reg_train_set, batch_size=cfg.train_batch_size, shuffle=True)

    valid_set = SegRegDataset(source, target_valid, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
    valid_loader = DataLoader(valid_set, batch_size=cfg.eval_batch_size)



    if not cfg.test_only:
        trainer_seg.cur_epoch = 0
        # trainer_seg.model.optimizer = torch.optim.Adam(net_se`g.parameters(), lr=cfg.seg_lr)
        trainer_seg.train(seg_train_loader, valid_loader, 
            epochs=cfg.seg_ep, 
            output_path=it_path_seg, 
            log_train_image=cfg.log_train_image, 
            log_validation_image=cfg.seg_log_validation_image,
            eval_frequency=cfg.eval_frequency,
            print_tag=f'iter {it_idx} Seg - ')
        
        trainer_reg.cur_epoch = 0
        # trainer_reg.model.optimizer = torch.optim.Adam(net_seg.parameters(), lr=cfg.reg_lr)
        trainer_reg.train(reg_train_loader, valid_loader, 
            epochs=cfg.reg_ep, 
            output_path=it_path_reg, 
            log_train_image=cfg.log_train_image, 
            # log_validation_image=False,
            log_validation_image=cfg.reg_log_validation_image,
            eval_frequency=cfg.eval_frequency,
            print_tag=f'iter {it_idx} Reg - ')

        # assessment
        count_good = assessment.assessment(target_set, it_output_path+'/assessed_data', 
                          cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix,
                          [0,1], cfg.threshold_dice, cfg.threshold_dice_min, cfg.min_good * (it_idx+2), target_key=cfg.target_key)
            

    # test seg
    print()
    if cfg.test_seg:
        trainer_seg.restore(f'{it_path_seg}/ckpt/model_best.pt')
        seg_test_set = SegRegDataset([], target_test, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        seg_test_loader = DataLoader(seg_test_set, batch_size=cfg.eval_batch_size)
        it_seg_results = trainer_seg.test(seg_test_loader, it_path_seg, desc=f'iter {it_idx} Seg - Test: ', log_image=cfg.seg_log_validation_image)
        sio.savemat(f'{it_path_seg}/test_results.mat', it_seg_results)
        with open(f'{it_path_seg}/test_results.txt', 'a+') as f:
            f.write(U.dict_to_str(it_seg_results) + '\n')
    
    
    # test reg
    if cfg.test_reg:
        trainer_reg.restore(f'{it_path_reg}/ckpt/model_best.pt')
        reg_test_set = SegRegDataset(sorted(source_test), sorted(target_test*len(source_test)), cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        reg_test_loader = DataLoader(reg_test_set, batch_size=len(source_test))
        it_reg_results = trainer_reg.test(reg_test_loader, it_path_reg, desc=f'iter {it_idx} Reg - Test: ', log_image=cfg.reg_log_validation_image)
        sio.savemat(f'{it_path_reg}/test_results.mat', it_reg_results)
        with open(f'{it_path_reg}/test_results.txt', 'a+') as f:
            f.write(U.dict_to_str(it_reg_results) + '\n')
    
    # if 'image' in it_reg_results:
    #     imgs = it_reg_results['image']
    #     from snmi.utils.log_writers import LogWriterFile
    #     writer = LogWriterFile(it_path_reg + '/test_images')
    #     for i, img in enumerate(imgs):
    #         for key in img:
    #             writer.write_image({key:img[key]}, epoch=i)
    
    print()

    with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'iter {it_idx}:\n')
            if cfg.test_seg:
                f.write(f'  Seg - {U.dict_to_str(it_seg_results)}\n')
            if cfg.test_reg:
                f.write(f'  Reg - {U.dict_to_str(it_reg_results)}\n')
            if count_good > 0:
                f.write(f'  count good - {count_good}/{len(target_train)}\n')
                count_good = 0
            f.write('\n')
    it_seg_results = None
    it_reg_results = None
