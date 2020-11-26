'''Invert CNN feature to reconstruct image: Reconstruct image from CNN features using gradient descent with momentum.

Author: Ken SHIRAKAWA <shirakawa.ken.38w@st.kyoto-u.ac.jp.>
'''
import os
from datetime import datetime
from scipy.io import savemat
import numpy as np
import PIL.Image
import torch.optim as optim
import torch
from loss import *
from utils import img_deprocess, img_preprocess, normalise_img, \
    vid_deprocess, normalise_vid, vid_preprocess, clip_extreme_value, get_cnn_features, create_feature_masks,\
    save_video, save_gif, gaussian_blur, clip_small_norm_value



def reconstruct_stim(features, net, net_gen, 
                     img_mean=np.array((0, 0, 0)).astype(np.float32),
                     img_std=np.array((1, 1, 1)).astype(np.float32),
                     norm=255,
                     bgr=False,
                     layer_weight = None, 
                     channel= None, feature_masks = None, mask = None,
                     feat_size_gen =  (4096,),
                     image_input_size=(1, 3, 224, 224), # for pytorch input
                     img_size_gen = (1, 3, 256, 256),
                     initial_gen_feat = None,
                     feat_upper_bound = 100., feat_lower_bound=0.,
                     loss_type='l2', iter_n = 200,  loss_weight = None, 
                     lr_start = 2., lr_end=1e-10,
                     momentum_start=0.9, momentum_end=0.9,
                     decay_start=0.01, decay_end=0.01,
                     disp_every=1,
                     grad_normalize = True, # this is differ from original caffe model
                     save_intermediate=False, save_intermediate_every=1,save_intermediate_path = None,
                     return_gen_feat = False,
                     opt_name='SGD'):
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
    elif loss_type == "MSE_Corr":
        loss_fun = MergeLoss([torch.nn.MSELoss(reduction='sum'), CorrLoss() ], weight_list = loss_weight)
    else:
        assert loss_type + ' is not correct'
        
    # make save dir
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('..', 'recon_img_by_dgn_icnn' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)
            
    #initial feature

    if initial_gen_feat is None:
        initial_gen_feat = np.random.normal(0,1, feat_size_gen).astype(np.float32)

    if save_intermediate:
        save_name = 'initial_gen_feat.mat'
        savemat(os.path.join(save_intermediate_path, save_name), {'initial_gen_feat': initial_gen_feat})
        
    #image_size 
    img_size = image_input_size 

    #top left offset for cropping the output image to get match image size
    top_left = ((img_size_gen[2] - img_size[2])//2,
               (img_size_gen[3] -img_size[3])//2)
    
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
        
        
    #iteration for optimization
    feat_gen = initial_gen_feat.copy()


    loss_list = np.zeros(iter_n, dtype='float32')
    
    for t in range(iter_n):
        feat_gen = torch.tensor(feat_gen).requires_grad_()
        feat_gen.retain_grad()
        if feat_gen.grad is None:
            feat_gen.grad = torch.zeros_like(feat_gen).detach()

        #parameter 
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n


        if opt_name == 'Adam':
            #op = optim.Adam([input], lr = lr)
            op = optim.Adam([feat_gen], lr = lr)
        elif opt_name == 'SGD':
            op = optim.SGD([feat_gen], lr=lr, momentum=momentum)
            #op = optim.SGD([input], lr=lr)
        elif opt_name == 'Adadelta':
            op = optim.Adadelta([feat_gen],lr = lr)
        elif opt_name == 'Adagrad':
            op = optim.Adagrad([feat_gen], lr = lr)
        elif opt_name == 'AdamW':
            op = optim.AdamW([feat_gen], lr = lr)
        elif opt_name == 'SparseAdam':
            op = optim.SparseAdam([feat_gen], lr = lr)
        elif opt_name == 'Adamax':
            op = optim.Adamax([feat_gen], lr = lr)
        elif opt_name == 'ASGD':
            op = optim.ASGD([feat_gen], lr = lr)

        elif opt_name == 'RMSprop':
            op = optim.RMSprop([feat_gen], lr = lr)
        elif opt_name == 'Rprop':
            op = optim.Rprop([feat_gen], lr = lr)

        # forward for generator

        img0 = net_gen(feat_gen).requires_grad_()
        img0.retain_grad()

        final_gen_feat = feat_gen.clone() #keep featgen for return_gen_out
        
        
        if t==0: #and save_intermediate:
            if len(img_size) == 4:
                #image
                save_img = img_deprocess(img0[0].detach().numpy(), img_mean, img_std, norm)
                save_name = 'initial_image.jpg'
                if bgr:
                    PIL.Image.fromarray(np.uint8(save_img[...,[2,1,0]])).save(os.path.join(save_intermediate_path, save_name))
                else:
                    PIL.Image.fromarray(np.uint8(save_img)).save(os.path.join(save_intermediate_path, save_name))
            elif len(img_size) == 5:
                # video
                save_vid = vid_deprocess(img0[0].detach().numpy(), img_mean, img_std, norm)
                # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
                save_name = 'initial_video.avi'
                save_video(save_vid, save_name, save_intermediate_path, bgr)

                save_name = 'initial_video.gif'
                save_gif(save_vid, save_name, save_intermediate_path, bgr,
                         fr_rate=150)

            else:
                print('Input size is not appropriate for save')
                assert len(input_size) not in [3,4]
                
        #image clip
        img_mask = np.zeros_like(img0.detach().numpy())
        img_mask[:,:, top_left[0]: top_left[0] + img_size[2], 
            top_left[1]: top_left[1] + img_size[3]
            ] = 1
        input = torch.masked_select(img0, torch.FloatTensor(img_mask).bool()).view(img_size).requires_grad_()
        input.retain_grad()
        if input.grad is None:
            input.grad = torch.zeros_like(input).detach()
            
        # forward
        fw = get_cnn_features(net, input, features.keys())
        
        # backward for net
        err = 0.
        loss = 0.
        # set the grad of network to 0
        net.zero_grad()
        net_gen.zero_grad()
        op.zero_grad()
        for j in range(num_of_layer):

            # op.zero_grad()

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

            grad_mean = torch.abs(feat_gen.grad).mean()

            if grad_mean > 0:
                feat_gen.grad /= grad_mean
                
        op.step()
        feat_gen = feat_gen.detach().numpy()
        err = err + loss
        loss_list[t] = loss
        
        #L2 decay
        feat_gen = (1 - decay) * feat_gen
        
        #clip feat
        if feat_lower_bound is not None:
            feat_gen = np.maximum(feat_gen, feat_lower_bound)
        if feat_upper_bound is not None:
            feat_gen = np.minimum(feat_gen, feat_upper_bound)
            
            
        # disp info
        if (t+1) % disp_every == 0:
            print('iter=%d; err=%g;' % (t+1, err))
            
        feat_gen = feat_gen.astype(np.float32)
        # save image
        if save_intermediate and ((t+1) % save_intermediate_every == 0):
            input_ = input.detach().numpy()[0]
            if bgr:
                input_ = input_[[2,1,0]]
            if len(input_) == 3:
                save_name = '%05d.jpg' % (t+1)
                PIL.Image.fromarray(normalise_img(img_deprocess(input_, img_mean, img_std, norm))).save(
                    os.path.join(save_intermediate_path, save_name))
            else:
                save_stim = input_
                # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
                save_name = '%05d.avi' % (t + 1)
                save_video(normalise_vid(vid_deprocess(save_stim, img_mean, img_std, norm)), save_name,
                           save_intermediate_path, bgr, fr_rate=30)
                save_name = '%05d.gif' % (t + 1)
                save_gif(normalise_vid(vid_deprocess(save_stim, img_mean, img_std, norm)), save_name,
                         save_intermediate_path,
                         bgr, fr_rate=150)
                
    # return img
    input = input.detach().numpy()[0]
    
    
    if len(input) == 3:
        return img_deprocess(input, img_mean, img_std, norm), loss_list
    else:
        return vid_deprocess(input, img_mean, img_std, norm), loss_list



