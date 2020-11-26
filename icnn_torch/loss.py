import torch
import numpy as np

class MSE_with_regulariztion(torch.nn.Module):
    """
    loss function for hand made
    In pytorch, hand made loss function requires two method, __init__ and forward
    """

    def __init__(self, vid, l_lambda=0.01):
        """
        
        """
        super(MSE_with_regulariztion, self).__init__()
        self.lam = l_lambda
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        self.vid = vid
        

    def forward(self, act, feat):
        vid = self.vid

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
      
        
        
def convert_for_corrmat(feat):
    feat_shape = feat.shape
    feat_flatten = feat.view(feat_shape[0], feat_shape[1], -1)
    return feat_flatten


def cov_torch(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)    
    m_exp = torch.mean(m,dim=2)
    x = m.permute(0,2,1) - m_exp[:, None]
    x = x.permute(0,2,1)
    #return x
    cov = 1 / (x.size(2) -1) * torch.einsum('bnm,bmk->bnk', x, x.permute(0,2,1))
    denom = 1. /torch.einsum('bn,bm->bnm',x.std(2),x.std(2))
    return cov * denom

class FeatCorrLoss(torch.nn.Module):
    """
    Loss function correlation coefficent replaced with MSE
    """
    
    def __init__(self):
        super(FeatCorrLoss, self).__init__()
        self.loss_fun = torch.nn.MSELoss()
        
        
    def forward(self, act, feat):
        
        feat_shape = feat.shape
        #act_flat = act.view(feat_shape[0], -1)
        #feat_flat = feat.view(feat_shape[0], -1)
        
        x = convert_for_corrmat(act)
        y = convert_for_corrmat(feat)
        
        x_mat = cov_torch(x)
        y_mat = cov_torch(y)

        return self.loss_fun(x_mat, y_mat)
    
    
    
class MergeLoss(torch.nn.Module):
    """
    Loss function with multiple loss 
    """
    
    def __init__(self, loss_list, weight_list = None):
        super(MergeLoss, self).__init__()
        self.loss_list = loss_list
        if weight_list is None:
            self.weight_list = torch.ones(len(loss_list))
            self.weight_list = self.weight_list / torch.tensor(len(self.weight_list))
        else:
            self.weight_list = weight_list
            
            
    def forward(self, act, feat):
        loss = 0
        for i in range(len(self.weight_list)):
            loss_fun = self.loss_list[i]
            weight = self.weight_list[i]
            loss += loss_fun(act,feat) * weight
            
        return loss 
   

        