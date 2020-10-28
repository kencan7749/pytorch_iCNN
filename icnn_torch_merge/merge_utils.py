import numpy as np
import torch


def torch_img_preprocess(img, img_mean=torch.tensor([0.485, 0.456, 0.406]),
                   img_std=torch.tensor([0.229, 0.224, 0.225]), norm=255):
    """convert to Pytorch's input image layout"""
    img = img.div(norm)
    image = (img.permute(2,0,1) -img_mean.reshape(3, 1, 1) )/ img_std.reshape(3, 1, 1)
    return image



def torch_vid_preprocess(vid, img_mean=torch.tensor([104, 117, 123]), img_std=torch.tensor([1., 1., 1.]), norm=255):
    """convert to Pytorch's input video layout"""
    vid = vid.div(norm)
    video = ( vid.permute(3,0,1,2) - img_mean.reshape(3, 1, 1, 1)) / img_std.reshape(3, 1, 1, 1)
    return video


def img_deprocess(img, img_mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
                  img_std=np.array([0.229, 0.224, 0.225], dtype=np.float32), norm=255):
    """convert from Pytorch's input image layout"""
    image = img * np.array([[img_std]]).T
    return np.dstack((image + np.reshape(img_mean, (3, 1, 1)))) * norm  # [:,:,::-1]


def vid_deprocess(vid, img_mean=np.float32([104, 117, 123]), img_std=np.float32([1, 1, 1]), norm=255):
    """convert from pytorch's input video layout"""
    video = vid * np.reshape(img_std, (3, 1, 1, 1))
    video = video + np.reshape(img_mean, (3, 1, 1, 1))
    return video.transpose(1, 2, 3, 0) * norm



def create_layer_weight(target_layer_dict):
    weight = np.ones(len(target_layer_dict)).astype(np.float32)
    weight = weight / weight.sum()
    
    layer_weight = {}
    for j, layer in enumerate(target_layer_dict.keys()):
        layer_weight[layer] = weight[j]
    return layer_weight

def calc_loss_one_model(net,fw, features, feature_masks, feature_weight, loss_fun, w_model=0.5):
    layer_weight = list(features.keys())
    loss = 0
    net.zero_grad()
    num_of_layer = len(features.keys())
    layer_list = list(features.keys())
    for j in range(num_of_layer):
        #optim.zero_grad()
        target_layer_id = num_of_layer-1-j
        target_layer = layer_list[target_layer_id]
        
        #extract activation or mask at input true video and mask
        act_j = fw[target_layer_id].clone()
        feat_j = features[target_layer].clone()
        mask_j = feature_masks[target_layer]
        
        layer_weight_j = feature_weight[target_layer]
        
        masked_act_j = torch.masked_select(act_j, torch.FloatTensor(mask_j).bool())
        masked_feat_j = torch.masked_select(feat_j, torch.FloatTensor(mask_j).bool())
        
        
        #calulate loss 
        loss_j = loss_fun(masked_act_j, masked_feat_j) * layer_weight_j * w_model

        loss_j.backward(retain_graph=True)
        
        loss += loss_j.detach().numpy()
        
    #optim.step()
    return loss

def rgb2gray(video):
    return video[0] * 0.2989 + video[1] * 0.5870 + video[2] * 0.1140


#flow 

def torch_vid2flow(vid):
    """
    input vid .. shape (fr, h,w , ch)

    return flow for input to DNN (h,w, fr * ch )

    """
    
    fr, h, w, ch = vid.shape
    return vid.permute(1,2,0,3).reshape(h,w, -1)


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

def torch_flow_preprocess(img, img_mean=np.array([0.5,0.5] * 10), img_std = np.array([0.226,0.226]*10), norm=255):
    img  = img/norm
    ch = img.shape[2]
    for t,m,s in zip(range(ch), img_mean, img_std):
        img[...,t] = (img[...,t] -m)/s
    return img.permute(2,0,1)

def torch_flow_deprocess(img,  img_mean=np.array([0.5,0.5] * 10), img_std = np.array([0.226,0.226]*10), norm=255):
    img = img.transpose(1,2,0)
    ch = img.shape[2]
    for t,m,s in zip(range(ch), img_mean, img_std):
        img[...,t] = s * img[...,t] + m
    return img * norm
