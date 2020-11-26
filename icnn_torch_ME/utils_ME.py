'''Utility functions for activation_maximization.
This codes in this repository are based on CNN preferred image (cnnpref) https://github.com/KamitaniLab/cnnpref, which is written for 'Caffe'. These scripts are released under the MIT license.

Copyright (c) 2018 Kamitani Lab (<http://kamitani-lab.ist.i.kyoto-u.ac.jp/>)

Author: Ken SHIRAKAWA <shirakawa.ken.38w@st.kyoto-u.ac.jp.jp>
'''

import os
import numpy as np
import scipy.ndimage as nd
# if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
import cv2
import copy
import torch
from PIL import Image



def vid_preprocess(vid, img_mean=np.float32([104, 117, 123]), img_std=np.float32([1., 1., 1.]), norm=255):
    '''convert to Pytorch's input video layout'''
    vid = vid / norm
    video = np.float32(np.transpose(vid, (3, 0, 1, 2)) - np.reshape(img_mean, (3, 1, 1, 1))) / np.reshape(img_std,
                                                                                                          (3, 1, 1, 1))
    return video



def vid_deprocess(vid, img_mean=np.float32([104, 117, 123]), img_std=np.float32([1, 1, 1]), norm=255):
    '''convert from pytorch's input video layout'''
    video = vid * np.reshape(img_std, (3, 1, 1, 1))
    video = video + np.reshape(img_mean, (3, 1, 1, 1))
    return video.transpose(1, 2, 3, 0) * norm

### if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
def load_video(file_path,ret_type, bgr=False, height=224,width=224):
    """
    ret_type select return as float32 float54, or uint8
    ret_type allows ['float', 'float64']
    """
    cap = cv2.VideoCapture(file_path)
    vid = []
  
    while True:
        
        ret, img = cap.read()
        
        if not ret:
            break
        
        # torchvision model allow RGB image , range of [0,1] and normalised 
        if bgr:
            img = cv2.resize(img, (height, width))
        else:
            img = cv2.resize(cv2.cvtColor(img , cv2.COLOR_BGR2RGB), (height, width))
       

        if ret_type == 'float32':
            img = img.astype(np.float32)
        elif ret_type== 'float':
            img = img.astype(np.float)
            
        vid.append(img)

    return np.array(vid)

def save_video(vid, save_name, save_intermidiate_path, bgr='True', fr_rate = 30):
    fr, height, width, ch = vid.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(os.path.join(save_intermidiate_path, save_name), fourcc, fr_rate, (width, height))
    for j in range(fr):
        frame = vid[j]
        if bgr == False:
            frame = frame[..., [2, 1, 0]]

        writer.write(frame.astype(np.uint8))
    writer.release()

def save_video_from_gray(vid_gray, save_name, save_intermidiate_path, bgr='True', fr_rate = 30):
    #input video is preferable to norm 0-255
    vid = np.array([np.array([vid_gray[...,i]] * 3).transpose(1,2,0) for i in range(vid_gray.shape[2])])
    fr, height, width, ch = vid.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(os.path.join(save_intermidiate_path, save_name), fourcc, fr_rate, (width, height))
    for j in range(fr):
        frame = vid[j]
        if bgr == False:
            frame = frame[..., [2, 1, 0]]
        
        writer.write(frame.astype(np.uint8))
    writer.release()

def save_gif(vid, save_name, save_intermidiate_path, bgr='False', fr_rate= 30):
    gif_list = []
    for frame in vid:
        if bgr == True:
            gif_list.append(Image.fromarray(frame[...,[2,1,0]].astype(np.uint8)))
        else:
            gif_list.append(Image.fromarray(frame.astype(np.uint8)))
    gif_list[0].save(os.path.join(save_intermidiate_path, save_name), save_all=True,
                     append_images=gif_list[1:], loop=0, duration=fr_rate)



def normalise_vid(vid):
    '''Normalize the image.
    Map the minimum pixel to 0; map the maximum pixel to 255.
    Convert the pixels to be int
    '''
    for j, img in enumerate(vid):
        img = img - img.min()
        if img.max() > 0:
            img = img * (255.0 / img.max())
        img = np.uint8(img)
        vid[j] = img
    return vid

def rgb2gray(video):
    
    return video[0] * 0.2989 + video[1] * 0.5870 + video[2] * 0.1140




def p_norm(x, p=2):
    '''p-norm loss and gradient'''
    loss = np.sum(np.abs(x) ** p)
    grad = p * (np.abs(x) ** (p - 1)) * np.sign(x)
    return loss, grad


def TV_norm(x, TVbeta=1):
    '''TV_norm loss and gradient'''
    TVbeta = float(TVbeta)
    d1 = np.roll(x, -1, 1)
    d1[:, -1, :] = x[:, -1, :]
    d1 = d1 - x
    d2 = np.roll(x, -1, 2)
    d2[:, :, -1] = x[:, :, -1]
    d2 = d2 - x
    v = (np.sqrt(d1 * d1 + d2 * d2)) ** TVbeta
    loss = v.sum()
    v[v < 1e-5] = 1e-5
    d1_ = (v ** (2 * (TVbeta / 2 - 1) / TVbeta)) * d1
    d2_ = (v ** (2 * (TVbeta / 2 - 1) / TVbeta)) * d2
    d11 = np.roll(d1_, 1, 1) - d1_
    d22 = np.roll(d2_, 1, 2) - d2_
    d11[:, 0, :] = -d1_[:, 0, :]
    d22[:, :, 0] = -d2_[:, :, 0]
    grad = TVbeta * (d11 + d22)
    return loss, grad


def TV_norm_vid(x, TVbeta_s=1, TVbeta_t=1):
    '''TV_norm loss and gradient'''
    TVbeta1 = float(TVbeta_s)
    TVbeta2 = float(TVbeta_t)
    d1 = np.roll(x, -1, 1)
    # maybe add new axis
    d1[:, :, -1, :] = x[:, :, -1, :]
    d1 = d1 - x
    d2 = np.roll(x, -1, 2)
    d2[:, :, :, -1] = x[:, :, :, -1]
    d2 = d2 - x
    d3 = np.roll(x, -1, 3)
    d3[:, -1, :, :] = x[:, -1, :, :]
    d3 = d3 - x

    v = (np.sqrt(d1 * d1 + d2 * d2) ** TVbeta1 + np.sqrt(d3 * d3) ** TVbeta2)
    loss = v.sum()
    v[v < 1e-5] = 1e-5
    d1_ = (v ** (2 * (TVbeta1 / 2 - 1) / TVbeta1)) * d1
    d2_ = (v ** (2 * (TVbeta1 / 2 - 1) / TVbeta1)) * d2
    d3_ = (v ** (2 * (TVbeta2 / 2 - 1) / TVbeta2)) * d3
    d11 = np.roll(d1_, 1, 1) - d1_
    d22 = np.roll(d2_, 1, 2) - d2_
    d33 = np.roll(d3_, 1, 2) - d3
    d11[:, :, 0, :] = -d1_[:, :, 0, :]
    d22[:, :, :, 0] = -d2_[:, :, :, 0]
    d33[:, 0, :, :] = -d3_[:, 0, :, :]
    grad = TVbeta1 * (d11 + d22) + TVbeta2 * (d33)
    return loss, grad


def image_norm(img):
    '''calculate the norm of the RGB for each pixel'''
    img_norm = np.sqrt(img[0] ** 2 + img[1] ** 2 + img[2] ** 2)
    return img_norm


def gaussian_blur(img, sigma):
    '''smooth the image with gaussian filter'''
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img
def gaussian_blur_ME(img, sigma):
    '''smooth the image with gaussian filter'''
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def clip_extreme_pixel(img, pct=1):
    '''clip the pixels with extreme values'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img = np.clip(img, np.percentile(img, pct / 2.), np.percentile(img, 100 - pct / 2.))
    return img


def clip_small_norm_pixel(img, pct=1):
    '''clip pixels with small RGB norm'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img_norm = image_norm(img)
    small_pixel = img_norm < np.percentile(img_norm, pct)

    img[0][small_pixel] = 0
    img[1][small_pixel] = 0
    img[2][small_pixel] = 0
    return img


def clip_small_contribution_pixel(img, grad, pct=1):
    '''clip pixels with small contribution'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img_contribution = image_norm(img * grad)
    small_pixel = img_contribution < np.percentile(img_contribution, pct)

    img[0][small_pixel] = 0
    img[1][small_pixel] = 0
    img[2][small_pixel] = 0
    return img

def create_feature_mask_ME(net, input_shape = (56, 56, 16), channel=None, t_index = None):
    '''
    create feature mask for all layers;
    select CNN units using masks or channels
    input:
        net: pytorch model to visualize intermediate layer
        input_shape: tuple that assigns input image to net
        channel:     int number to set the channel to visalize
        x_index, y_index, t_index: [optional] int number to set the position of chanel to maximize
    output:
        feature_masks: a numpy ndarray consists of masks for CNN features, mask has the same shape as the CNN features of the corresponding layer;
    '''

    tmp_input = np.random.randint(0,256, input_shape).astype(np.float32)
    # to convert pytorch input
    #axis_len = np.arange(len(tmp_input.shape))
    tmp_input = torch.Tensor(tmp_input)
    feat_shape = net.predict(tmp_input).shape#get_target_feature_shape(net, tmp_input, extract_feat_list)
    feature_mask = np.zeros(feat_shape)

    feature_mask[channel, t_index] = 1
    return feature_mask