''' from original FNO repo '''
import torch
import torch.nn as nn
from .basics import SpectralConv2dV2, _get_act


class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 dropout=0,
                 activation='tanh',
                 mean_constraint=False):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2dV2(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.dropout = nn.Dropout(p=dropout)

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint

    def forward(self, x):
        '''
        (b,c,h,w) -> (b,1,h,w)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x) # project
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
            x = self.dropout(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)

        if self.mean_constraint:
            x = x - torch.mean(x, dim=(-2,-1), keepdim=True)

        return x

def fno(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut]*len(params.modes1)
        params.modes2 = [params.mode_cut]*len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut]*len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    input_dim = params.in_dim

    return FNN2d(params.modes1, params.modes2, layers=params.layers, fc_dim=params.fc_dim,
                in_dim=input_dim, out_dim=params.out_dim, dropout=params.dropout,
                activation='gelu', mean_constraint=(params.loss_func == 'pde'))
