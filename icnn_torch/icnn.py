'''Invert CNN feature to reconstruct image: Reconstruct image from CNN features using gradient descent with momentum.

Author: Ken SHIRAKAWA <shirakawa.ken.38w@st.kyoto-u.ac.jp.>
'''
import os
from datetime import datetime

import numpy as np
import PIL.Image
import torch.optim as optim
import torch
from loss import MSE_with_regulariztion
from utils import img_deprocess, img_preprocess, normalise_img, \
    vid_deprocess, normalise_vid, vid_preprocess, clip_extreme_pixel, get_cnn_features, create_feature_masks,\
    save_video, save_gif, gaussian_blur, clip_small_norm_pixel


def reconstruct_stim(features, net,
                     img_mean=np.array((0, 0, 0)).astype(np.float32),
                     img_std=np.array((1, 1, 1)).astype(np.float32),
                     norm=255,
                     bgr=False,
                     initial_input=None,
                     input_size=(224, 224, 3),
                     layer_weight=None, channel=None, mask=None,
                     opt_name='SGD',
                     prehook_dict = {},
                     lr_start=0.02, lr_end=1e-12,
                      momentum_start=0.9, momentum_end=0.9,
                      decay_start=0.02, decay_end=1e-11,
                      image_jitter=False, jitter_size=4,
                      image_blur=False, sigma_start=0.02, sigma_end=0.005,
                      p=3, lamda=0.5,
                      TVlambda = [0,0],
                      clip_extreme=False, clip_extreme_every=4, e_pct_start=1, e_pct_end=1,
                      clip_small_norm=False, clip_small_norm_every=4, n_pct_start=5., n_pct_end=5.,

                     loss_type='l2', iter_n=200,  save_intermediate=False,
                     save_intermediate_every=1, save_intermediate_path=None,
                     disp_every=1,
                     ):
    if loss_type == "l2":
        loss_fun = torch.nn.MSELoss(reduction='sum')
    elif loss_type == "L2_with_reg":
        loss_fun = MSE_with_regulariztion(L_lambda=lamda, alpha=p, TV_lambda=TVlambda)
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

            save_name = 'initial_video.gif'
            save_gif(initial_input, save_name, save_intermediate_path, bgr,
                     fr_rate=150)

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
    feature_masks = create_feature_masks(layer_dict, masks=mask, channels=channel)

    # iteration for gradient descent
    input = initial_input.copy().astype(np.float32)
    if len(input_size) == 3:
        input = img_preprocess(input, img_mean, img_std, norm)
    else:
        input = vid_preprocess(input, img_mean, img_std, norm)

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
            input = np.roll(np.roll(input, ox, -1), oy, -2)

        # forward
        input = torch.tensor(input[np.newaxis], requires_grad=True)
        if opt_name == 'Adam':
            op = optim.Adam([input], lr = lr)
        elif opt_name == 'SGD':
            op = optim.SGD([input], lr=lr, momentum=momentum)
        elif opt_name == 'Adadelta':
            op = optim.Adadelta([input])
        elif opt_name == 'Adagrad':
            op = optim.Adagrad([input])
        elif opt_name == 'AdamW':
            op = optim.AdamW([input])
        elif opt_name == 'SparseAdam':
            op = optim.SparseAdam([input])
        elif opt_name == 'Adamax':
            op = optim.Adamax([input])
        elif opt_name == 'ASGD':
            op = optim.ASGD([input])

        elif opt_name == 'RMSprop':
            op = optim.RMSprop([input])
        elif opt_name == 'Rprop':
            op = optim.Rprop([input])
        fw = get_cnn_features(net, input, features.keys(), prehook_dict)
        # backward for net
        err = 0.
        loss = 0.
        # set the grad of network to 0
        net.zero_grad()
        for j in range(num_of_layer):

            # op.zero_grad()
            op.zero_grad()
            target_layer_id = num_of_layer -1 -j
            target_layer = layer_list[target_layer_id]
            # extract activation or mask at input true video, and mask
            act_j = fw[target_layer_id].clone()
            feat_j = features[target_layer].clone()
            mask_j = feature_masks[target_layer]

            layer_weight_j = layer_weight[target_layer]

            masked_act_j = torch.masked_select(act_j, torch.FloatTensor(mask_j).bool())
            masked_feat_j = torch.masked_select(feat_j, torch.FloatTensor(mask_j).bool())
            # calculate loss using pytorch loss function
            loss_j = loss_fun(masked_act_j, masked_feat_j) * layer_weight_j

            # backward the gradient to the video
            loss_j.backward(retain_graph=True)

            loss += loss_j.detach().numpy()

        op.step()

        input = input.detach().numpy()[0]

        err = err + loss
        loss_list[t] = loss

        # clip pixels with extreme value
        if clip_extreme and (t+1) % clip_extreme_every == 0:
            e_pct = e_pct_start + t * (e_pct_end - e_pct_start) / iter_n
            input = clip_extreme_pixel(input, e_pct)

        # clip pixels with small norm
        if clip_small_norm and (t+1) % clip_small_norm_every == 0:
            n_pct = n_pct_start + t * (n_pct_end - n_pct_start) / iter_n
            input = clip_small_norm_pixel(input, n_pct)

        # unshift
        if image_jitter:
            input = np.roll(np.roll(input, -ox, -1), -oy, -2)

        # L_2 decay
        input = (1-decay) * input

        # gaussian blur
        if image_blur:
            if len(input_size) == 3:
                input = gaussian_blur(input, sigma)
            else:
                for i in range(input.shape[1]):
                    input[:, i] = gaussian_blur(input[:, i], sigma)

        # disp info
        if (t+1) % disp_every == 0:
            print('iter=%d; err=%g;' % (t+1, err))


        # save image
        if save_intermediate and ((t+1) % save_intermediate_every == 0):
            if len(input_size) == 3:
                save_name = '%05d.jpg' % (t+1)
                PIL.Image.fromarray(normalise_img(img_deprocess(input, img_mean, img_std, norm))).save(
                    os.path.join(save_intermediate_path, save_name))
            else:
                save_stim = input
                # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
                save_name = '%05d.avi' % (t + 1)
                save_video(normalise_vid(vid_deprocess(save_stim, img_mean, img_std, norm)), save_name,
                           save_intermediate_path, bgr, fr_rate=30)
                save_name = '%05d.gif' % (t + 1)
                save_gif(normalise_vid(vid_deprocess(save_stim, img_mean, img_std, norm)), save_name,
                         save_intermediate_path,
                         bgr, fr_rate=150)
    # return img
    if len(input_size) == 3:
        return img_deprocess(input, img_mean, img_std, norm), loss_list
    else:
        return vid_deprocess(input, img_mean, img_std, norm), loss_list
