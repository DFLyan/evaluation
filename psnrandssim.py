#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
from skimage.measure import compare_ssim as ssim_c
import tensorflow as tf
import tensorlayer as tl
import math as ma
from model import *
from utils import *
from config import config, log_config
from skimage.measure import compare_psnr,compare_mse


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_img16s.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs





def evaluate():
    ###====================== PRE-LOAD DATA ==========================###
    path = 'samples/DTCS/train91_gray_64/83/test/0.25_g/'
    test_xhat_img_list = sorted(tl.files.load_file_list(path=path, regx='.*._gen.png', printable=False))
    test_xhat_imgs = read_all_imgs(test_xhat_img_list[0:11], path=path, n_threads=1)

    test_hr_img_list = sorted(tl.files.load_file_list(path=path, regx='.*._hr.png', printable=False))
    test_hr_imgs = read_all_imgs(test_hr_img_list[0:11], path=path, n_threads=1)


    ###======================= EVALUATION =============================###
    global sum,sum_s,sum_t
    sum = 0
    sum_s = 0
    sum_t = 0
    sum_m = 0

    for imid in range(0, len(test_hr_imgs)):
        xhat_imgs_ = tl.prepro.threading_data(test_xhat_imgs[imid:imid + 1], fn=norm)
        b_imgs_ = tl.prepro.threading_data(test_hr_imgs[imid:imid + 1], fn=norm)
        psnr = compare_psnr(b_imgs_, xhat_imgs_)
        print("PSNR:%.8f" % psnr)
        # mse = compare_mse(b_imgs_, xhat_imgs_)
        # b_imgs_ = np.reshape(b_imgs_, [xhat_imgs_.shape[1], xhat_imgs_.shape[1], xhat_imgs_.shape[0]])
        # xhat_imgs_ = np.reshape(xhat_imgs_, [xhat_imgs_.shape[1], xhat_imgs_.shape[1], xhat_imgs_.shape[0]])
        # ssim = ssim_c(X=b_imgs_, Y=xhat_imgs_, multichannel=False)
        # print("SSIM:%.8f" % ssim)
        sum += psnr
        # sum_m += mse
        # sum_s += ssim


        # if imid % 100 == 0:
        #     print("[*] save images")
            # out = np.reshape(out, [block_size, block_size, 3])
            # tl.vis.save_image(out,  save_dir+'/test_gen%d.jpg' % imid)
            # b_imgs_hr = np.reshape(b_imgs_, [block_size, block_size, 3])
            # tl.vis.save_image(b_imgs_hr, save_dir+'/test_hr%d.jpg' % imid)

    # time_sum = time.time() - start_time
    print("TIME_SUM:%.8f" % sum_t)
    print("Num of image:%d" % len(test_hr_imgs))
    psnr_a = sum / len(test_hr_imgs)
    print("PSNR_AVERAGE:%.8f" % psnr_a)
    mse_a = sum_m/ len(test_hr_imgs)
    print("MSE_AVERAGE:%.8f" % mse_a)
    ssim_a = sum_s / len(test_hr_imgs)
    print("SSIM_AVERAGE:%.8f" % ssim_a)
    # time_a = sum_t / len(test_hr_imgs)
    # print("TIME_AVERAGE:%.8f" % time_a)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate_RGB', help='evaluate_RGB')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'evaluate_RGB':
        evaluate()
    else:
        raise Exception("Unknow --mode")