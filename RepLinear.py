import torch 
import torch.nn as nn
import numpy as np


class RepLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n=1, deploy=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.deploy = deploy
        self.n = n
        
        if self.deploy:
            self.rep_fc = nn.Linear(in_dim, out_dim, bias=True)
        else:
            self.branches = nn.ModuleList() 
            for i in range(n):
                self.branches.append(self.conv_bn(in_dim, out_dim, 1))
            self.branches.append(nn.BatchNorm1d(out_dim))

            self.fc = nn.Conv1d(in_dim, out_dim, 1, bias=True)
            
    @staticmethod
    def conv_bn(in_channels, out_channels, kernel_size=1):
        result = nn.Sequential()
        result.add_module('conv', nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, bias=False))
        result.add_module('bn', nn.BatchNorm1d(num_features=out_channels))
        return result

    def forward(self, x):
        if self.deploy:
            return self.rep_fc(x)
        
        else:
            x = x.permute(0, 2, 1)  
            branch_outputs = []
            for branch in self.branches:
                branch_outputs.append(branch(x))  

            x = self.fc(sum(branch_outputs))
            x = x.permute(0, 2, 1) 

            return x
    
    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):   
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm1d)    # 仅BN
            # print('fuse bn')
            kernel_value = np.zeros((self.in_dim, self.in_dim, 1), dtype=np.float32)
            for i in range(self.in_dim):
                kernel_value[i, i, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def _fuse_conv(self, w1, b1, conv2):
        '''
        x --> Conv1 --> Conv2 --> y
        Conv1 is the first filter,
        Conv2 is the second subsequent to Conv1.
        '''
        w2 = conv2.weight
        b2 = conv2.bias 
        
        w = torch.einsum('oi,icj->ocj', w2.squeeze().squeeze(), w1)
        b = (b1.view(1,-1,1)*w2).sum(2).sum(1) + b2
        
        return w, b 
    
    def switch_to_deploy(self):
        if hasattr(self, 'rep_fc'):
            return
        
        w = torch.zeros((self.out_dim, self.in_dim, 1)).to(self.fc.weight.device)
        b = torch.zeros((self.out_dim)).to(self.fc.weight.device)
        for i in range(self.n+1):
            wi, bi = self._fuse_bn(self.branches[i])
            w += wi
            b += bi

        w_rep, b_rep = self._fuse_conv(w, b, self.fc)
        self.rep_fc = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.rep_fc.weight.data = w_rep.squeeze(-1)
        self.rep_fc.bias.data = b_rep
        self.__delattr__('branches')
        self.__delattr__('fc')
        self.deploy = True        
        
        
if __name__ == '__main__':
    # Verify the equivalence of RepLinear before and after reparameterization.
    FC = RepLinear(3, 3, n=2, deploy=False)
    
    print('The original RepLinear is: \n', FC)   # print the complex structure of RepLinear
    x = torch.rand((1, 2, 3))   # [Batch, #patches, embed_dim]
    
    FC.eval()   # switch to eval mode (fix BN parameters)
    y = FC(x)
    print('\n The output of the RepLinear before reparameterization is: \n', y)
    
    # Reparameterize the RepLinear:
    FC.switch_to_deploy()
    print('\n The reparameterized RepLinear is: \n', FC)    # become a single linear now
    
    y_rep = FC(x)  
    print('\n The output of the Replinear after reparameterization is: \n', y_rep)