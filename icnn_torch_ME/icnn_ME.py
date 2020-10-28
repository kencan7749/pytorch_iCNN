import os
from datetime import datetime
from scipy.io import savemat
import numpy as np
import PIL.Image
import torch.optim as optim
import torch
from utils_ME import vid_deprocess, vid_preprocess, normalise_vid, rgb2gray, create_feature_mask_ME, save_video, save_gif, gaussian_blur_ME




def reconstruct_stim(features, net, 
                     img_mean=np.array((0, 0, 0)).astype(np.float32),
                     initial_input=None,
                     input_size=(3, 56, 56, 16),
                     feature_masks=None,
                     layer_weight=None, channel=None, mask=None,
                     bgr = False,
                     opt_name='SGD',
                     prehook_dict = {},
                     lr_start=2., lr_end=1e-10,
                      momentum_start=0.9, momentum_end=0.9,
                      pix_decay = True, decay_start=0.2, decay_end=1e-10,
                      grad_normalize = True,
                      image_jitter=False, jitter_size=4,
                      image_blur=True, sigma_start=2, sigma_end=0.5,
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
    img_mean_gray = rgb2gray(img_mean)
    
    # image norm
    noise_img = np.random.randint(0, 256, (input_size))
    img_norm0 = np.linalg.norm(noise_img)
    img_norm0 = img_norm0/2.

    # initial input
    if initial_input is None:
        initial_input = np.random.randint(0, 256, (input_size)) # (3, 56, 56, 16)
    else:
        input_size = initial_input.shape

    if save_intermediate:
        s_video = initial_input.transpose(3, 1,2,0)
        # video
        # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
        save_name = 'initial_video.avi'
        save_video(s_video, save_name, save_intermediate_path, bgr)

        #save_name = 'initial_video.gif'
        #save_gif(s_video, save_name, save_intermediate_path, bgr,
        #         fr_rate=150)

    # layer_list
    layer_dict = features
    layer_list = list(features.keys())

    # number of layers
    num_of_layer = len(layer_list)

   
    # feature mask
    if feature_masks is None:
        feature_masks = create_feature_mask_ME(net, channel=channel)

    # iteration for gradient descent
    input = initial_input.copy().astype(np.float32)
    #input = torch.tensor(input_, requires_grad=True)


    loss_list = np.zeros(iter_n, dtype='float32')

    for t in range(iter_n):
        input = torch.tensor(input, requires_grad=True)
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        sigma = sigma_start + t * (sigma_end - sigma_start) / iter_n
        
        # shift
        if image_jitter:
            #not yet imprelement well
            ox, oy = np.random.randint(-jitter_size, jitter_size+1, 2)
            input = np.roll(np.roll(input, ox, -1), oy, -2)

        if opt_name == 'Adam':
            #op = optim.Adam([input], lr = lr)
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
            
        input_gray = rgb2gray(input)
        input_gray -= img_mean_gray
        fw = net.predict(input_gray)
        err = 0.
        loss = 0.
        
        op.zero_grad()
        if net.W[0].grad is not None:
            net.W[0].grad.zero_
            net.W[1].grad.zero_
            net.W[2].grad.zero_
            net.W[3].grad.zero_
            
        target_layer = layer_list[-1]
        act_j = fw.clone()
        feat_j = features[target_layer].clone()
        mask_j = feature_masks

        #layer_weight_j = layer_weight[target_layer]

        masked_act_j = torch.masked_select(act_j, torch.FloatTensor(mask_j).bool())
        masked_feat_j = torch.masked_select(feat_j, torch.FloatTensor(mask_j).bool())
        # calculate loss using pytorch loss function
        loss_j = loss_fun(masked_act_j, masked_feat_j) #* layer_weight_j

        # backward the gradient to the video
        loss_j.backward(retain_graph=True)
        loss = loss_j.detach().numpy()

            
        if grad_normalize:
            grad_mean = torch.abs(input.grad).mean()
            if grad_mean > 0:
                input.grad /= grad_mean
        op.step()
        
        input = input.detach().numpy()
        
        err = err + loss
        loss_list[t] = loss

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
           
            input = gaussian_blur_ME(input, sigma)
            

        # disp info
        if (t+1) % disp_every == 0:
            print('iter=%d; err=%g;' % (t+1, err))


        # save image
        if save_intermediate and ((t+1) % save_intermediate_every == 0):
            save_stim = input.copy().transpose(3, 1,2,0)
            # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
            save_name = '%05d.avi' % (t + 1)
            save_video(normalise_vid(save_stim), save_name,
                       save_intermediate_path, bgr, fr_rate=30)
            #save_name = '%05d.gif' % (t + 1)
            #save_gif(normalise_vid(save_stim), save_name,
            #         save_intermediate_path,
            #         bgr, fr_rate=150)
    

    return input, loss_list
