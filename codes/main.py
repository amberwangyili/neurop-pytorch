from utils import *
from data import build_train_loader,build_val_loader
import argparse
import logging
import os
import random
import numpy as np
import torch
from models import build_model
from tqdm import tqdm
from collections import defaultdict
import imageio



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='./configs/test_neurop_mit5k_dark.yaml')
    args = parser.parse_args()
    opt = parse(args.config)
    opt = dict_to_nonedict(opt)

    for fp in opt['path']:
        path = (opt['path'][fp])
        if not os.path.exists(path):
            os.makedirs(path)
    logger_name = f"{opt['name']}{get_timestamp()}"
    logger = logging.getLogger('base')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)        
    log_file = f"{opt['path']['log']}/{logger_name}.log"    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info(dict2str(opt))


    dataset_opt = opt['datasets']
    val_loader = build_val_loader(dataset_opt)
    model = build_model(opt)



    #### validation
    pbar = ProgressBar(len(val_loader))
    avg_psnr = 0.
    avg_ssim = 0.
    idx = 0
    for val_data in val_loader:
        
        img_name = get_file_name(val_data['LQ_path'][0])
        img_dir = opt['path']['results_root']
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        idx += 1
        model.feed_data(val_data)
        model.test()
        visuals = model.get_current_visuals()

        sr_img = visuals['rlt']
        gt_img = visuals['GT']  

        # save_img_path = os.path.join(img_dir,'{:s}.png'.format(img_name))
        # imageio.imwrite(save_img_path, (255.0 * sr_img).astype('uint8'))

        psnr = calculate_psnr(sr_img, gt_img)
        ssim = calculate_ssim((255.0 * sr_img).astype('uint8'), (255.0 * gt_img).astype('uint8'))
        avg_psnr += psnr
        avg_ssim += ssim
        pbar.update('Test {}'.format(img_name))
        logger.info('# Test {}, PSNR: {:.4e}, SSIM: {:.4e}'.format(img_name,psnr,ssim))

    logger.info('# Validation # Average PSNR: {:.4e}, Average SSIM: {:.4e}'.format(avg_psnr/idx,avg_ssim/idx))
