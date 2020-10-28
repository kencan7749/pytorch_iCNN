import os
import sys
from datetime import datetime

import numpy as np
import PIL.Image
import torch.optim as optim
import torch
from torch.nn import functional as F
from loss import MSE_with_regulariztion
from merge_utils import create_layer_weight, calc_loss_one_model, torch_img_preprocess, torch_vid_preprocess

sys.path.append('../icnn_torch')
from utils import img_deprocess, img_preprocess, normalise_img, \
    vid_deprocess, normalise_vid, vid_preprocess, clip_extreme_value, get_cnn_features, create_feature_masks,\
    save_video, save_gif, gaussian_blur, clip_small_norm_value

sys.path.append('../icnn_torch_ME')
from utils_ME import vid_deprocess, vid_preprocess, normalise_vid, rgb2gray, create_feature_mask_ME, save_video, save_gif, gaussian_blur_ME

def reconstruct_stim(features_list, net_list, #[video, image] if video model was used 
                     dgn,
                     
                     img_mean_list,
                     img_std_list,
                     norm_list,
                     bgr_list,
                     prep_list,
                     weight_list=None,
                     initial_input=None,
                     eps_init =None,
                     z_C_init = None,
                     input_size=(16, 224, 224, 3),
                     feature_masks_list=None,
                     layer_weight_list=None, channel=None, mask=None,
                     opt_name='Adam',
                     initial_feat_gen = None,
                     feat_size_gen =  (16, 4096),
                     lr_start=2, lr_end=1e-10,
                     feat_upper_bound = 100., feat_lower_bound=0.,
                      momentum_start=0.9, momentum_end=0.9,
                      pix_decay=True,
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

    if initial_feat_gen is None:
        initial_feat_gen = np.random.normal(0,1, feat_size_gen).astype(np.float32)
        
    #save if save intermediate is True
    if save_intermediate:
        aa = dgn.generate(torch.tensor(initial_feat_gen))
        initial_input = aa.detach().numpy()
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
            
    if layer_weight_list is None:
        layer_weight_list = [create_layer_weight(feat_dict) for feat_dict in features_list]
        
    if weight_list is None:
        weight_list = np.ones(len(features_list)) / len(features_list)
            
                 
    #create feature mask
    if feature_masks_list is None:
        feature_masks_list = []
        for features in features_list:
            feat_mask = create_feature_masks(features)
            feature_masks_list.append(feat_mask)
            
                
   
    loss_list = np.zeros(iter_n, dtype='float32')
     
    
    loss_list = np.zeros(iter_n, dtype='float32')
    #input_ = initial_input.copy().astype(np.float32)
    
    feat_input_ = initial_feat_gen.copy()
    
    

    for t in range(iter_n):
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        
        # optimize based on the first element
        feat_input = torch.tensor(feat_input_, requires_grad=True)
        #eps_init = torch.tensor(eps_input_, requires_grad=True)
        #z_C_init = torch.tensor(z_C_input_, requires_grad=True)

        if opt_name == 'Adam':
            op = optim.Adam([feat_input], lr = lr)
        elif opt_name == 'SGD':
            op = optim.SGD([feat_input], lr=lr, momentum=momentum)
            #op = optim.SGD([input], lr=lr)
        elif opt_name == 'Adadelta':
            op = optim.Adadelta([feat_input],lr = lr)
        elif opt_name == 'Adagrad':
            op = optim.Adagrad([feat_input], lr = lr)
        elif opt_name == 'AdamW':
            op = optim.AdamW([feat_input], lr = lr)
        elif opt_name == 'SparseAdam':
            op = optim.SparseAdam([feat_input], lr = lr)
        elif opt_name == 'Adamax':
            op = optim.Adamax([feat_input], lr = lr)
        elif opt_name == 'ASGD':
            op = optim.ASGD([feat_input], lr = lr)
        elif opt_name == 'RMSprop':
            op = optim.RMSprop([feat_input], lr = lr)
        elif opt_name == 'Rprop':
            op = optim.Rprop([feat_input], lr = lr)

        op.zero_grad()
        #gen_v = dgn.generate(input)
        #input = F.interpolate(gen_v, input_size[1:3]).permute(1,2,3,0)
        input = dgn.generate(feat_input)
        net_loss_list = []
        # preprocess input for each network
        feat_inp_list = []
        for i in range(len(net_list)):
            net = net_list[i]
            features = features_list[i]
            target_layer_list = list(features.keys())
            img_mean = torch.tensor(img_mean_list[i])
            img_std = torch.tensor(img_std_list[i])
            norm = norm_list[i]
            bgr = bgr_list[i]
            prep_l = prep_list[i]
            feature_masks = feature_masks_list[i]
            feature_weight = layer_weight_list[i]
            net_weight = weight_list[i]
            inp = input.clone()
            if i == 0:
                #motion energy/
                if net.W[0].grad is not None:
                    net.W[0].grad.zero_
                    net.W[1].grad.zero_
                    net.W[2].grad.zero_
                    net.W[3].grad.zero_
                #temporal model
                inp = inp.permute(3,0, 1,2)#ch fr h w
                #assert 1== 0
                #inp = F.interpolate(inp, (56,56), mode='bicubic',align_corners=True)
                for prep in prep_l:
                    if prep == "resize(56,56)":
                        m =torch.nn.AdaptiveAvgPool2d( (56,56))
                        inp = m(inp)
                inp = inp.permute(0, 2,3,1) # ch h w fr
                
                inp= rgb2gray(inp)
                inp -= rgb2gray(img_mean)
                #inp -= params.zero_mean
                feature_temporal = net.predict(inp)
                loss =loss_fun(feature_temporal, features[target_layer_list[0]]) * net_weight


                loss.backward(retain_graph =True)
                net_loss_list.append(loss.detach().numpy())

            else:    
                for prep in prep_l:
                    
                    if prep == 'mean':
                        inp = inp.mean(0)

                # preprocess
                if len(inp.shape) == 3:
                    inp = torch_img_preprocess(inp, img_mean, img_std, norm)
                else:
                    inp = torch_vid_preprocess(inp, img_mean, img_std, norm)

                inp = inp.view([1]  + list(inp.shape))
                for prep in prep_l:
                    if prep == "resize(112,112)":
                        inp = inp
                    elif prep == "resize(224,224)":
                        inp = F.interpolate(inp, (224,224), mode='bicubic', align_corners=True)

                #obtain features
                feat_inp = get_cnn_features(net, inp, target_layer_list)
                feat_inp_list.append(feat_inp)
                #calc loss
                loss = calc_loss_one_model(net, feat_inp, features, feature_masks, feature_weight, loss_fun, w_model=net_weight)
                net_loss_list.append(loss)


        op.step()

        loss_list[t] = np.sum(net_loss_list)
        # disp info
        if (t+1) % disp_every == 0:
            print('iter=%d; err=%g;' % (t+1, loss_list[t]))

        input_ = input.detach().numpy().copy()  
        feat_input_ = feat_input.detach().numpy().copy()
        
        
        #clip feat
        if feat_lower_bound is not None:
            feat_input_ = np.maximum(feat_input_, feat_lower_bound).astype(np.float32)
        if feat_upper_bound is not None:
            feat_input_ = np.minimum(feat_input_, feat_upper_bound).astype(np.float32)
        
        # clip pixels with extreme value
        if clip_extreme and (t+1) % clip_extreme_every == 0:
            e_pct = e_pct_start + t * (e_pct_end - e_pct_start) / iter_n
            input_ = clip_extreme_value(input_, e_pct)

        # clip pixels with small norm
        if clip_small_norm and (t+1) % clip_small_norm_every == 0:
            n_pct = n_pct_start + t * (n_pct_end - n_pct_start) / iter_n
            input_ = clip_small_norm_value(input_, n_pct)

        # unshift
        if image_jitter:
            input_ = np.roll(np.roll(input_, -ox, -1), -oy, -2)

        
        # L_2 decay
        if pix_decay:
            input_ = (1-decay) * input_
        # save image
        if save_intermediate and ((t+1) % save_intermediate_every == 0):
            if len(input_size) == 3:
                save_name = '%05d.jpg' % (t+1)
                PIL.Image.fromarray(normalise_img(clip_extreme_value(input_, pct=0.04))).save(
                    os.path.join(save_intermediate_path, save_name))
            else:
                save_stim = input_.copy()
                # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
                save_name = '%05d.avi' % (t + 1)
                save_video(normalise_vid(clip_extreme_value(save_stim, pct=0.04)), save_name,
                           save_intermediate_path, bgr_list[0], fr_rate=30)
                #save_name = '%05d.gif' % (t + 1)
                #save_gif(normalise_vid(clip_extreme_value(save_stim, pct=0.04)), save_name,
                #         save_intermediate_path,
                #         bgr_list[0], fr_rate=150)



    save_stim = input_.copy()
    return save_stim, loss_list
   