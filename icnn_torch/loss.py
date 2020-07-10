import torch
import numpy as np

class MSE_with_regulariztion(torch.nn.Module):
    """
    loss function for hand made
    In pytorch, hand made loss function requires two method, __init__ and forward
    """

    def __init__(self, input, L_lambda=0, alpha=3, TV_lambda=[0, 0], B=160 / 255):
        """
        if R_b is not false, using L2 normalize
        """
        super(with_regulariztion, self).__init__()
        self.input = input

        self.l_lambda = L_lambda
        self.alpha = alpha
        self.TV_lambda = TV_lambda

        self.B = B
        self.V = B / 6.5
        self.loss_fun  = torch.nn.MSELoss(reduction='sum')

    def forward(self, act, feat, max_val=1, res_field_size=3 * 3 * 3):

        loss = self.loss_fun(act, feat)

        # regulaize
        if self.l_lambda > 0:
            local_reg = self.local_energy()
            loss += local_reg

        if np.sum(self.TV_lambda) > 0:
            tv_reg = self.local_frequency()
            loss += tv_reg

        return loss  # / (max_val * res_field_size **2)

    def local_energy(self):
        _, ch, T, H, W = self.input.shape
        R_b = torch.sum(torch.pow(torch.sum(torch.pow(self.input, 2), 1), self.alpha / 2))

        R_b /= T * H * W * self.B ** self.alpha * self.l_lambda
        return R_b

    def local_frequency(self):
        _, ch, T, H, W = self.input.shape
        d1 = torch.roll(self.input, -1, -1)
        d1[:, :, :, :, -1] = self.input[:, :, :, :, -1]
        d1 = d1 - self.input

        d2 = torch.roll(self.input, -1, -2)
        d2[:, :, :, -1, :] = self.input[:, :, :, -1, :]
        d2 = d2 - self.input
        tv_sp = (torch.pow(d1, 2) + torch.pow(d2, 2)) * self.TV_lambda[0]

        d3 = torch.roll(self.input, -1, -3)
        d3[:, :, -1, :, :] = self.input[:, :, -1, :, :]
        d3 = d3 - self.input

        tv_tm = torch.pow(d3, 2) * self.TV_lambda[1]

        R_tv = torch.sum(tv_sp + tv_tm)

        R_tv /= T * H * W * self.V ** 2

        return R_tv