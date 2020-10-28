import os
import sys
from datetime import datetime

import numpy as np
import PIL.Image
import torch.optim as optim
import torch
from torch.nn import functional as F
from loss import MSE_with_regulariztion
from merge_utils import create_layer_weight, calc_loss_one_model, torch_img_preprocess, torch_vid_preprocess, torch_vid2flow, flow2vid, torch_flow_preprocess, torch_flow_deprocess

sys.path.append('../icnn_torch')
from utils import img_deprocess, img_preprocess, normalise_img, \
    vid_deprocess, normalise_vid, vid_preprocess, clip_extreme_value, get_cnn_features, create_feature_masks,\
    save_video, save_gif, gaussian_blur, clip_small_norm_value



def reconstruct_stim(features_list, net_list, #[video, image] if video model was used 
                     
                     img_mean_list,
                     img_std_list,
                     norm_list,
                     bgr_list,
                     prep_list,
                     weight_list=None,
                     initial_input=None,
                     input_size=(11, 224, 224, 3),
                     feature_masks_list=None,
                     layer_weight_list=None, channel=None, mask=None,
                     opt_name='SGD',
                     
                     lr_start=2, lr_end=1e-10,
                      momentum_start=0.9, momentum_end=0.9,
                      pix_decay= True,
                      decay_start=0.01, decay_end=0.01,
                      grad_normalize = False,
                      image_jitter=False, jitter_size=4,
                      image_blur=True, sigma_start=2, sigma_end=0.5,
                     
                      clip_extreme=False, clip_extreme_every=4, e_pct_start=1, e_pct_end=1,
                      clip_small_norm = False,
                     
                    
                     

                     loss_type='l2', iter_n=200,  save_intermediate=False,
                     save_intermediate_every=1, save_intermediate_path=None,
                     disp_every=1,
                     ):
    
    loss_fun = torch.nn.MSELoss(reduction='sum')
    
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('./recon_stim_combine_snapshots' + datetime.now().strftime('%Y%m%dT%H%M%S'))
            os.makedirs(save_intermediate_path, exist_ok=True)
    
    
    if weight_list is None:
        weight_list = [1.0 /len(net_list) for i in range(len(net_list))]
        
    if layer_weight_list is None:
        layer_weight_list = [create_layer_weight(feat_dict) for feat_dict in features_list]
        
        
    if initial_input is None:
        initial_input = np.random.randint(0,256, (input_size)).astype(np.float32)
      
    #save if save intermediate is True
    if save_intermediate:
        if len(input_size) == 3:
            #image
            save_name = 'initial_image.jpg'
            if bgr_list[0]:
                PIL.Image.fromarray(np.uint8(initial_input[...,[2,1,0]])).save(os.path.join(save_intermediate_path, save_name))
            else:
                PIL.Image.fromarray(np.uint8(initial_input)).save(os.path.join(save_intermediate_path, save_name))
        elif len(input_size) == 4:
            # video
            # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
            save_name = 'initial_video.avi'
            save_video(initial_input, save_name, save_intermediate_path, bgr_list[0])

            #save_name = 'initial_video.gif'
            #save_gif(initial_input, save_name, save_intermediate_path, bgr_list[0],
            #         fr_rate=150)

        else:
            print('Input size is not appropriate for save')
            assert len(input_size) not in [3,4]
            
        
    #create feature mask
    if feature_masks_list is None:
        feature_masks_list = []
        for features in features_list:
            feat_mask = create_feature_masks(features)
            feature_masks_list.append(feat_mask)
            
                
   
    loss_list = np.zeros(iter_n, dtype='float32')
    
    input_ = initial_input.copy().astype(np.float32)
    
    
    for t in range(iter_n):
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        sigma = sigma_start + t * (sigma_end - sigma_start) / iter_n
        # optimize based on the first element
        input = torch.tensor(input_, requires_grad=True)

        if opt_name == 'Adam':
            op = optim.Adam([input], lr = lr)
        elif opt_name == 'SGD':
            op = optim.SGD([input], lr=lr, momentum=momentum)
            #op = optim.SGD([input], lr=lr)
        elif opt_name == 'Adadelta':
            op = optim.Adadelta([input],lr = lr)
        elif opt_name == 'Adagrad':
            op = optim.Adagrad([input], lr = lr)
        elif opt_name == 'AdamW':
            op = optim.AdamW([input], lr = lr)
        elif opt_name == 'SparseAdam':
            op = optim.SparseAdam([input], lr = lr)
        elif opt_name == 'Adamax':
            op = optim.Adamax([input], lr = lr)
        elif opt_name == 'ASGD':
            op = optim.ASGD([input], lr = lr)
        elif opt_name == 'RMSprop':
            op = optim.RMSprop([input], lr = lr)
        elif opt_name == 'Rprop':
            op = optim.Rprop([input], lr = lr)

        op.zero_grad()
        net_loss_list = []
        # preprocess input for each network
        feat_inp_list = []
        for i in range(len(net_list)):
            net = net_list[i]
            features = features_list[i]
            target_layer_list = features.keys()
            img_mean = torch.tensor(img_mean_list[i])
            img_std = torch.tensor(img_std_list[i])
            norm = norm_list[i]
            bgr = bgr_list[i]
            prep_l = prep_list[i]
            feature_masks = feature_masks_list[i]
            feature_weight = layer_weight_list[i]
            net_weight = weight_list[i]
            inp = input
            for prep in prep_l:
                if prep == 'mean':
                    inp = inp.mean(0)

                elif prep == "resize(224,224)":
                    #upsample
                    inp = inp

           
                elif prep == 'spatial':
                    inp = torch_img_preprocess(inp, img_mean, img_std, norm)
                elif prep == 'flow':
                    inp = torch_vid2flow(inp)
                    inp = torch_flow_preprocess(inp, img_mean, img_std, norm)
                
                for prep in prep_l:
                    if prep == "resize(56,56)":
                        inp = F.interpolate(inp, (56,56),mode='bilinear', align_corners =True)
               

            inp = inp.view([1]  + list(inp.shape))
            #obtain features
            feat_inp = get_cnn_features(net, inp, target_layer_list)
            feat_inp_list.append(feat_inp)
            #calc loss
            loss = calc_loss_one_model(net, feat_inp, features, feature_masks, feature_weight, loss_fun, w_model=net_weight)
            net_loss_list.append(loss)


        op.step()
        input_ = input.detach().numpy()
        err = np.sum(net_loss_list)
        loss_list[t] = err
        
        # clip pixels with extreme value
        if clip_extreme and (t+1) % clip_extreme_every == 0:
            e_pct = e_pct_start + t * (e_pct_end - e_pct_start) / iter_n
            input = clip_extreme_value(input, e_pct)

        # clip pixels with small norm
        if clip_small_norm and (t+1) % clip_small_norm_every == 0:
            n_pct = n_pct_start + t * (n_pct_end - n_pct_start) / iter_n
            input = clip_small_norm_value(input, n_pct)

        # unshift
        if image_jitter:
            input = np.roll(np.roll(input, -ox, -1), -oy, -2)

        # L_2 decay
        if pix_decay:
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
                PIL.Image.fromarray(normalise_img(clip_extreme_value(input_, pct=0.04))).save(
                    os.path.join(save_intermediate_path, save_name))
            else:
                save_stim = input_
                # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
                save_name = '%05d.avi' % (t + 1)
                save_video(normalise_vid(clip_extreme_value(save_stim, pct=0.04)), save_name,
                           save_intermediate_path, bgr_list[0], fr_rate=30)
                #save_name = '%05d.gif' % (t + 1)
                #save_gif(normalise_vid(clip_extreme_value(save_stim, pct=0.04)), save_name,
                #         save_intermediate_path,
                #         bgr_list[0], fr_rate=150)
    # return img
    if len(input_size) == 3:
        return input_, loss_list
    else:
        return input_, loss_list

        
    
   