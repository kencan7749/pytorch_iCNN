import torch
import numpy as np

class MSE_with_regulariztion(torch.nn.Module):
    """
    loss function for hand made
    In pytorch, hand made loss function requires two method, __init__ and forward
    """

    def __init__(self, l_lambda=0.01):
        """
        
        """
        super(MSE_with_regulariztion, self).__init__()
        self.lam = l_lambda
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        
        

    def forward(self, act, feat, vid):

        loss = self.loss_fun(act, feat)
        
        img_mask = np.zeros_like(vid.detach().numpy())
        img_mask[-1] = 1
        
        diff = vid - torch.roll(vid, 1, 0)
        diff_masked = torch.masked_select(diff, torch.FloatTensor(img_mask).bool()).requires_grad_()
        

        loss += self.lam * torch.sum(diff_masked **2)

        return loss  # / (max_val * res_field_size **2
    
    
    
class CorrLoss(torch.nn.Module):
    """
    Loss function correlation coefficent replaced with MSE
    """
    
    def __init__(self):
        super(CorrLoss, self).__init__()
        
        
    def forward(self, act, feat):
        
        feat_shape = feat.shape
        act_flat = act.view(feat_shape[0], -1)
        feat_flat = feat.view(feat_shape[0], -1)
        
        x = act_flat
        y = feat_flat
        
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2))* torch.sqrt(torch.sum(vy**2)))
        return  - corr 
      
        
        