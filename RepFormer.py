import torch 
import torch.nn as nn
import numpy as np
from RepLinear import RepLinear
from torch.nn.init import trunc_normal_, _calculate_fan_in_and_fan_out


# The latest PyTorch version has surported RMSNorm as nn.RMSNorm(dim, eps)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 n_heads=5,
                 qkv_bias=False,
                ):
        super(Attention, self).__init__()

        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = (self.qkv(x)
               .reshape(b, n, 3, self.n_heads, c//self.n_heads)
               .permute(2, 0, 3, 1, 4)
               )
        q,k,v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)   
        x = (attn @ v).transpose(1,2).reshape(b,n,c)

        x = self.proj(x)

        return x
    

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hid_features,
                 out_features,
                 n_branch=1,
                 ):
        super(MLP, self).__init__()

        out_features = out_features or in_features
        hid_features = hid_features or in_features

        self.fc1 = RepLinear(in_features, hid_features, n=n_branch)    # Use RepLinear here
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x
    

class PatchEmbed(nn.Module):
    def __init__(self, img_size=16,
                 patch_size=8,
                 in_c=1,
                 embed_dim=30,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.n_patches = int((img_size[1] - patch_size[1]) / (patch_size[1] / 2)) + 1

        self.proj = nn.Conv2d(in_c, embed_dim,
                              kernel_size=patch_size,
                              stride=int(patch_size[1] / 2), bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):  
        x = self.proj(x).flatten(2).transpose(1, 2)       
        if self.norm is not None:
            x = self.norm(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
class EncoderBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=1.,
            qkv_bias=False,
            init_values=None,
            n_branch=1):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)   # if not surported, replaced it with RMSNorm
        self.attn = Attention(dim, n_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = nn.RMSNorm(dim, eps=1e-6)   # if not surported, replaced it with RMSNorm
        self.mlp = MLP(in_features=dim, hid_features=int(dim * mlp_ratio), out_features=dim, n_branch=n_branch)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        
        return x
    
    
def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

def init_vit_weights(module, name='', head_bias=0.):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
        

class RepFormer(nn.Module):
    def __init__(self,
                 img_size=(2, 128),
                 patch_size=(2, 8),
                 in_c=1,
                 n_classes=11,
                 embed_dim=30,
                 depth=5,
                 n_heads=5,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 init_values=None,
                 pe=False,
                 n_branch=1,
                 ):
        super(RepFormer, self).__init__()

        self.n_classes = n_classes
        self.n_features = self.embed_dim = embed_dim
        self.pe = pe
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
        
        if self.pe:   # positional encoding
            self.n_tokens = 0
            n_patches = self.patch_embed.n_patches
            self.pos_embed = nn.Parameter(torch.zeros(
                1, n_patches+self.n_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.Sequential(*[
            EncoderBlock(embed_dim, n_heads, mlp_ratio, qkv_bias, init_values, n_branch) for i in range(depth)])

        self.fc_norm = nn.RMSNorm(embed_dim, eps=1e-6)   # if not surported, replaced it with RMSNorm

        self.head = nn.Linear(self.n_features, n_classes)

        self.apply(init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)   
        if self.pe:
            x = x + self.pos_embed
        x = self.blocks(x)
        x = x.mean(dim=1)  # GAP
        x = self.fc_norm(x)
        x = self.head(x)

        return x
    
    def switch_to_deploy(self):
        for i in range(self.depth):
            self.blocks[i].mlp.fc1.switch_to_deploy()
            
            
if __name__ == '__main__':
    # Verify the equivalence of RepFormer before and after reparameterization.
    net = RepFormer(img_size=(2, 128),
                          patch_size=(2, 16),
                          n_classes=11,
                          depth=5,
                          n_heads=5,
                          embed_dim=30,
                          mlp_ratio=1,
                          qkv_bias=False,
                          init_values=1,
                          pe=False,
                          n_branch=2,
                          )
    
    print('The original RepFormer is: \n', net)   # print the complex structure of RepLinear
    x = torch.rand((1, 1, 2, 128))   # [Batch, #channel, height, width]
    
    net.eval()   # switch to eval mode (fix BN parameters)
    y = net(x)
    
    # Reparameterize the RepFormer:
    net.switch_to_deploy()
    print('\n The reparameterized RepFormer is: \n', net)    # become standard MLP now
    
    y_rep = net(x)  
    print('\n The output of the RepFormer before reparameterization is: \n', y)
    print('\n The output of the RepFormer after reparameterization is: \n', y_rep)