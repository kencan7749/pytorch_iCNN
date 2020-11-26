'''Invert CNN feature to reconstruct image: Reconstruct image from CNN features using gradient descent with momentum.

Author: Ken SHIRAKAWA <shirakawa.ken.38w@st.kyoto-u.ac.jp.>
'''
import os
from datetime import datetime

import numpy as np
import PIL.Image
import torch.optim as optim
import torch
from loss import *
from utils import img_deprocess, img_preprocess, normalise_img, \
    vid_deprocess, normalise_vid, vid_preprocess, clip_extreme_value, get_cnn_features, create_feature_masks,\
    save_video, save_gif, gaussian_blur, clip_small_norm_value


def reconstruct_stim(features, net,
                     img_mean=np.array((0, 0, 0)).astype(np.float32),
                     img_std=np.array((1, 1, 1)).astype(np.float32),
                     norm=255,
                     bgr=False,
                     initial_input=None,
                     input_size=(224, 224, 3),
                     feature_masks=None,
                     layer_weight=None, channel=None, mask=None,
                     opt_name='SGD',
                     prehook_dict = {},
                     lr_start=2., lr_end=1e-10,
                      momentum_start=0.9, momentum_end=0.9,
                      pix_decay = True,decay_start=0.2, decay_end=1e-10,
                      grad_normalize = True,
                      image_jitter=False, jitter_size=4,
                      image_blur=True, sigma_start=2, sigma_end=0.5,
                      p=3, lamda=0.5,
                      TVlambda = [0,0],
                     
                      clip_extreme=False, clip_extreme_every=4, e_pct_start=1, e_pct_end=1,
                      clip_small_norm=False, clip_small_norm_every=4, n_pct_start=5., n_pct_end=5.,

                     loss_type='l2', iter_n=200, loss_weight = None,
                     save_intermediate=False,
                     save_intermediate_every=1, save_intermediate_path=None,
                     disp_every=1,
                     ):
    if loss_type == "l2":
        loss_fun = torch.nn.MSELoss(reduction='sum')
    elif loss_type == "L2_with_reg":
        loss_fun = MSE_with_regulariztion(L_lambda=lamda, alpha=p, TV_lambda=TVlambda)
    elif loss_type == "CorrLoss":
        loss_fun = CorrLoss()
    elif loss_type == "FeatCorrLoss":
        loss_fun = FeatCorrLoss()
    elif loss_type == "MSE_Corr_FeatCorr":
        loss_fun = MergeLoss([torch.nn.MSELoss(reduction='sum'),CorrLoss(), FeatCorrLoss() ], weight_list = loss_weight)
    elif loss_type == "MSE_FeatCorr":
        loss_fun = MergeLoss([torch.nn.MSELoss(reduction='sum'), FeatCorrLoss() ], weight_list = loss_weight)
    else:
        assert loss_type + ' is not correct'
    # make save dir
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('..', 'recon_img_by_icnn' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)

    # image size
    input_size = input_size

    # image mean
    img_mean = img_mean
    img_std = img_std
    norm = norm
    # image norm
    noise_img = np.random.randint(0, 256, (input_size))
    img_norm0 = np.linalg.norm(noise_img)
    img_norm0 = img_norm0/2.

    # initial input
    if initial_input is None:
        initial_input = np.random.randint(0, 256, (input_size))
    else:
        input_size = initial_input.shape

    if save_intermediate:
        if len(input_size) == 3:
            #image
            save_name = 'initial_image.jpg'
            if bgr:
                PIL.Image.fromarray(np.uint8(initial_input[...,[2,1,0]])).save(os.path.join(save_intermediate_path, save_name))
            else:
                PIL.Image.fromarray(np.uint8(initial_input)).save(os.path.join(save_intermediate_path, save_name))
        elif len(input_size) == 4:
            # video
            # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
            save_name = 'initial_video.avi'
            save_video(initial_input, save_name, save_intermediate_path, bgr)

            #save_name = 'initial_video.gif'
            #save_gif(initial_input, save_name, save_intermediate_path, bgr,
            #         fr_rate=150)

        else:
            print('Input size is not appropriate for save')
            assert len(input_size) not in [3,4]


    # layer_list
    layer_dict = features
    layer_list = list(features.keys())

    # number of layers
    num_of_layer = len(layer_list)

    # layer weight
    if layer_weight is None:
        weights = np.ones(num_of_layer)
        weights = np.float32(weights)
        weights = weights / weights.sum()
        layer_weight = {}
        for j, layer in enumerate(layer_list):
            layer_weight[layer] = weights[j]

    # feature mask
    if feature_masks is None:
        feature_masks = create_feature_masks(layer_dict, masks=mask, channels=channel)

    # iteration for gradient descent
    input = initial_input.copy().astype(np.float32)
    if len(input_size) == 3:
        input_stim = img_preprocess(input, img_mean, img_std, norm)
    else:
        input_stim = vid_preprocess(input, img_mean, img_std, norm)

    loss_list = np.zeros(iter_n, dtype='float32')

    for t in range(iter_n):
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        sigma = sigma_start + t * (sigma_end - sigma_start) / iter_n

        # shift
        if image_jitter:
            ox, oy = np.random.randint(-jitter_size, jitter_size+1, 2)
            input_stim = np.roll(np.roll(input_stim, ox, -1), oy, -2)

        # forward
        input_stim = torch.tensor(input_stim[np.newaxis], requires_grad=True)
        if opt_name == 'Adam':
            op = optim.Adam([input_stim], lr = lr)
        elif opt_name == 'SGD':
            op = optim.SGD([input_stim], lr=lr, momentum=momentum)
        elif opt_name == 'Adadelta':
            op = optim.Adadelta([input_stim])
        elif opt_name == 'Adagrad':
            op = optim.Adagrad([input_stim])
        elif opt_name == 'AdamW':
            op = optim.AdamW([input_stim])
        elif opt_name == 'SparseAdam':
            op = optim.SparseAdam([input_stim])
        elif opt_name == 'Adamax':
            op = optim.Adamax([input_stim])
        elif opt_name == 'ASGD':
            op = optim.ASGD([input_stim])

        elif opt_name == 'RMSprop':
            op = optim.RMSprop([input_stim])
        elif opt_name == 'Rprop':
            op = optim.Rprop([input_stim])
        fw = get_cnn_features(net, input_stim, features.keys(), prehook_dict)
        # backward for net
        err = 0.
        loss = 0.
        # set the grad of network to 0
        net.zero_grad()
        op.zero_grad()
        for j in range(num_of_layer):
      
            target_layer_id = num_of_layer -1 -j
            
            target_layer = layer_list[target_layer_id]
            
            # extract activation or mask at input true video, and mask
            act_j = fw[target_layer_id].clone()
            feat_j = features[target_layer].clone()
            mask_j = feature_masks[target_layer]

            layer_weight_j = layer_weight[target_layer]
            
            masked_act_j = torch.masked_select(act_j, torch.FloatTensor(mask_j).bool())
            masked_feat_j = torch.masked_select(feat_j, torch.FloatTensor(mask_j).bool())
            #if loss_type == 'FeatCorrLoss':
            masked_act_j = masked_act_j.view(act_j.shape)
            masked_feat_j = masked_feat_j.view(feat_j.shape)
            # calculate loss using pytorch loss function
            loss_j = loss_fun(masked_act_j, masked_feat_j) * layer_weight_j

            # backward the gradient to the video
            loss_j.backward(retain_graph=True)

            loss += loss_j.detach().numpy()
            
        if grad_normalize:
            grad_mean = torch.abs(input_stim.grad).mean()
            if grad_mean > 0:
                input_stim.grad /= grad_mean

        op.step()

        input_stim = input_stim.detach().numpy()[0]

        err = err + loss
        loss_list[t] = loss

        # clip pixels with extreme value
        if clip_extreme and (t+1) % clip_extreme_every == 0:
            e_pct = e_pct_start + t * (e_pct_end - e_pct_start) / iter_n
            input_stim = clip_extreme_pixel(input_stim, e_pct)

        # clip pixels with small norm
        if clip_small_norm and (t+1) % clip_small_norm_every == 0:
            n_pct = n_pct_start + t * (n_pct_end - n_pct_start) / iter_n
            input_stim = clip_small_norm_pixel(input_stim, n_pct)

        # unshift
        if image_jitter:
            input_stim = np.roll(np.roll(input_stim, -ox, -1), -oy, -2)

        # L_2 decay
        if pix_decay:
            input_stim = (1-decay) * input_stim

        # gaussian blur
        if image_blur:
            
            if len(input_size) == 3:
                input_stim = gaussian_blur(input_stim, sigma)
            else:
                for i in range(input_stim.shape[1]):
                    input_stim[:,i] = gaussian_blur(input_stim[:,i], sigma)

        # disp info
        if (t+1) % disp_every == 0:
            print('iter=%d; err=%g;' % (t+1, err))


        # save image
        if save_intermediate and ((t+1) % save_intermediate_every == 0):
            if len(input_size) == 3:
                save_name = '%05d.jpg' % (t+1)
                save_img = input_stim.copy()
                if bgr:
                    PIL.Image.fromarray(normalise_img(img_deprocess(save_img, img_mean, img_std, norm))[...,[2,1,0]]).save(
                        os.path.join(save_intermediate_path, save_name))
                    
                else:
                    PIL.Image.fromarray(normalise_img(img_deprocess(save_img, img_mean, img_std, norm))).save(
                        os.path.join(save_intermediate_path, save_name))
              
            else:
                save_stim = input_stim.copy()
                # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
                save_name = '%05d.avi' % (t + 1)
                save_video(normalise_vid(vid_deprocess(save_stim, img_mean, img_std, norm)), save_name,
                           save_intermediate_path, bgr,fr_rate=30)
                #save_name = '%05d.gif' % (t + 1)
                #save_gif(normalise_vid(vid_deprocess(save_stim, img_mean, img_std, norm)), save_name,
                #         save_intermediate_path, bgr,
                #         fr_rate=150)
    # return img
    if len(input_size) == 3:
        return img_deprocess(input_stim, img_mean, img_std, norm), loss_list
    else:
        return vid_deprocess(input_stim, img_mean, img_std, norm), loss_list
