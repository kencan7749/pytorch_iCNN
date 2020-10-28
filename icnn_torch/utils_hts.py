import numpy as np


import torch.nn as nn
import torch

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,20,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook


def vid2flow(vid):
    """
    input vid .. shape (fr, h,w , ch)

    return flow for input to DNN (h,w, fr * ch )
    """

    flow = []
    for fr in range(len(vid)):

        for j in range(vid.shape[-1]):
            flow.append(vid[fr][..., j])

    return np.array(flow).transpose(1, 2, 0)


def flow2vid(flow, axis=2, input_ch_num = 2):
    """
        input vid .. shape (h,w, fr * ch )

        return flow for input to DNN shape (fr, h,w , ch)
        """
    fr = flow.shape[-1] // input_ch_num
    flow_rgb = []
    for f in range(fr):
        frame = flow[:, :, input_ch_num * f:input_ch_num * f + input_ch_num]
        if input_ch_num == 2:
            fshape = frame.shape[:2]
            flow_rgb.append(np.append(frame, np.ones(fshape + (1,)) * 128, axis=axis))
        else:
            flow_rgb.append(frame, axis=axis)
    return np.array(flow_rgb)

def flow_preprocess(img, img_mean=np.array([0.5,0.5] * 10), img_std = np.array([0.226,0.226]*10), norm=255):
    img  = img/norm
    ch = img.shape[2]
    for t,m,s in zip(range(ch), img_mean, img_std):
        img[...,t] = (img[...,t] -m)/s
    return img.transpose(2,0,1)

def flow_deprocess(img,  img_mean=np.array([0.5,0.5] * 10), img_std = np.array([0.226,0.226]*10), norm=255):
    img = img.transpose(1,2,0)
    ch = img.shape[2]
    for t,m,s in zip(range(ch), img_mean, img_std):
        img[...,t] = s * img[...,t] + m
    return img * norm


def flow_norm(flow, lower_bound=-20, upper_bound=20):
    fr = flow.shape[0] // 2
    intensity_range = upper_bound - lower_bound
    flow_list = []
    for flow_num in range(fr):
        flow_x = flow[:, :, 0 + flow_num * 2]
        flow_y = flow[:, :, 1 + flow_num * 2]

        np.clip(flow_x, lower_bound, upper_bound)
        flow_x = (flow_x - lower_bound) * 255 / intensity_range

        np.clip(flow_y, lower_bound, upper_bound)
        flow_y = (flow_y - lower_bound) * 255 / intensity_range

        flow_list.append([flow_x, flow_y])
    return flow_list



