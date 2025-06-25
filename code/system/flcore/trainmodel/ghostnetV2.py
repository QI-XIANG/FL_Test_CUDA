# 2020.11.06-Changed for building GhostNetV2
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.registry import register_model

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    
  
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    
class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None,args=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            ) 
      
    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]         
        elif self.mode in ['attn']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))  
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 


class GhostBottleneckV2(nn.Module): 

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,layer_id=None,args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id<=1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='original',args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='attn',args=args) 

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
            
        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False,mode='original',args=args)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

cfgs_default = [   
        # k, t, c, SE, s 
        [[3,  16,  16, 0, 1]],
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
   
class GhostNetV2(nn.Module):
    def __init__(self, cfgs=cfgs_default, num_classes=1000, width=1.0, dropout=0.2,block=GhostBottleneckV2,args=None):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block==GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id,args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

@register_model
def ghostnetv2(**kwargs):
    cfgs = [   
        # k, t, c, SE, s 
        [[3,  16,  16, 0, 1]],
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNetV2(cfgs, num_classes=kwargs['num_classes'], 
                    width=kwargs['width'], 
                    dropout=kwargs['dropout'],
                    args=kwargs['args'])

#------------------------------------------------------------------------------------------------------------------------

class FedProtoCINIC10_GhostNetV2(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CINIC-10,
    using a pre-trained **GhostNetV2** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    This model leverages a custom GhostNetV2 implementation, adapting it for
    small image datasets and the unique requirements of federated learning.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. This significantly
        lowers FLOPs, memory usage, and communication costs.
    2.  **Leverage Pre-training (if available):** If you have pre-trained weights for this
        GhostNetV2, you can load them after instantiation to benefit from features learned
        on large datasets like ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CINIC-10 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **GhostNetV2** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。這顯著降低了 FLOPs、內存使用量和通信成本。
    2.  **利用預訓練（如果可用）：** 如果您有此 GhostNetV2 的預訓練權重，
        您可以在實例化後加載它們，以受益於在大型數據集（如 ImageNet）上學習到的特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    def __init__(self, num_classes=10):
        """
        Initializes the FedProtoCINIC10_GhostNetV2 with a GhostNetV2 backbone.

        Parameters:
            num_classes (int): The number of output classes. For CINIC-10, this is 10.
        """
        super(FedProtoCINIC10_GhostNetV2, self).__init__()

        # --- Instantiate the GhostNetV2 backbone ---
        # The provided GhostNetV2 code has a 'ghostnetv2' function at the end,
        # which acts as a factory.
        # It takes 'num_classes', 'width', 'dropout', and 'args' as kwargs.
        # For our use case, 'num_classes' for the backbone should be 1000 (ImageNet),
        # 'width' can be 1.0 (standard), 'dropout' can be 0.0 for feature extraction.
        # The 'args' parameter from your provided file isn't explicitly used by GhostNetV2's
        # __init__ or in GhostModuleV2 within this context, so we can pass a dummy object or None.

        class DummyArgs:
            # Placeholder for 'args' if the GhostNetV2 implementation internally expects it
            # but doesn't strictly use its attributes for the modules we care about.
            # You might need to add attributes here if your GhostNetV2 version
            # uses them for its 'mode' selection or other configurations.
            pass

        # Define cfgs as it's passed to GhostNetV2, and it's defined in the original `ghostnetv2` function.
        cfgs = [
            # k, t, c, SE, s
            [[3,  16,  16, 0, 1]],
            [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            [[5,  72,  40, 0.25, 2]],
            [[5, 120,  40, 0.25, 1]],
            [[3, 240,  80, 0, 2]],
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
            ],
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
            ]
        ]
        
        # Instantiate GhostNetV2. We set num_classes to 1000 to match ImageNet pre-training.
        backbone = GhostNetV2(cfgs=cfgs, num_classes=1000, width=1.0, dropout=0.0, args=DummyArgs())
        print("GhostNetV2 backbone initialized.")

        # --- Load Pre-trained Weights (if available) ---
        # If you have specific pre-trained weights (e.g., a .pth file), you would load them here.
        # Example:
        # try:
        #     state_dict = torch.load('path/to/your/ghostnetv2_weights.pth', map_location='cpu')
        #     # Remove 'classifier' weights from state_dict if you're replacing the head
        #     # state_dict.pop('classifier.weight', None)
        #     # state_dict.pop('classifier.bias', None)
        #     backbone.load_state_dict(state_dict, strict=False) # strict=False allows missing/unexpected keys
        #     print("Loaded pre-trained GhostNetV2 weights.")
        # except FileNotFoundError:
        #     print("No pre-trained GhostNetV2 weights found. Initializing from scratch.")
        # except Exception as e:
        #     print(f"Error loading GhostNetV2 weights: {e}. Initializing from scratch.")
        

        # --- Adaptation for 32x32 input ---
        # The `conv_stem` of GhostNetV2 typically has a stride of 2.
        # For small 32x32 images, we need to reduce this to stride=1 to avoid aggressive downsampling.
        # This modification re-initializes the weights of this specific convolutional layer.
        
        original_conv_stem = backbone.conv_stem
        
        # Create a new Conv2d layer with stride=1
        new_conv_stem = nn.Conv2d(
            in_channels=original_conv_stem.in_channels,
            out_channels=original_conv_stem.out_channels,
            kernel_size=original_conv_stem.kernel_size,
            stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
            padding=original_conv_stem.padding,
            bias=original_conv_stem.bias
        )
        # Replace the original conv_stem
        backbone.conv_stem = new_conv_stem
        
        # In this GhostNetV2 structure, `conv_stem` is followed by `bn1` and `act1`.
        # No explicit pooling layer needs to be removed/modified here.
        print("GhostNetV2: Adapted initial conv_stem stride for 32x32 inputs (no resizing).")

        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The feature extractor comprises `conv_stem`, `bn1`, `act1`, `blocks`, `global_pool`, `conv_head`, and `act2`.

        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # 特徵提取器包括 `conv_stem`、`bn1`、`act1`、`blocks`、`global_pool`、`conv_head` 和 `act2`。
        self.feature_extractor = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
            backbone.blocks,
            backbone.global_pool,
            backbone.conv_head,
            backbone.act2
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Set requires_grad to False to freeze parameters
        print("GhostNetV2 backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # The original GhostNetV2 classifier is `backbone.classifier`.
        # We replace it with a new linear layer suitable for our `num_classes`.
        # This is the only part of the model whose parameters will be updated during client training.

        # 定義新的、可訓練的分類頭。
        # 原始 GhostNetV2 分類器是 `backbone.classifier`。
        # 我們將用一個適合我們 `num_classes` 的新線性層替換它。
        # 這是模型中唯一在客戶端訓練期間更新其參數的部分。

        # Get the input feature dimension for the new classifier from the original model's last linear layer.
        feature_dim = backbone.classifier.in_features
        self.classifier = nn.Linear(feature_dim, num_classes)
        print(f"New classifier initialized with {feature_dim} input features and {num_classes} output classes.")

    def forward(self, x):
        """
        Performs a forward pass through the network, producing classification logits.

        Parameters:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: The classification logits from the model.
        """
        # Pass input through the frozen feature extractor (inference mode for these layers)
        # Even if the model is in training mode, these frozen layers should behave like evaluation mode.
        # 即使模型在訓練模式，這些凍結層也應行為如評估模式。
        self.feature_extractor.eval()
        with torch.no_grad():  # Ensure no gradients are computed for the frozen backbone
            x = self.feature_extractor(x)

        # The global_pool, conv_head, and act2 are already part of the feature_extractor.
        # After these operations, the tensor should be `(batch_size, channels, 1, 1)`.
        # Flatten it to `(batch_size, channels)` before passing to the final linear layer.
        x = x.view(x.size(0), -1)
        
        # Pass the flattened features through the trainable classification layer.
        # 將展平後的特徵通過可訓練的分類層。
        x = self.classifier(x)
        return x

    def get_features(self, x):
        """
        Extracts features (embeddings) from the input before the final classification layer.
        This method is highly useful in prototype-based federated learning (like FedProto)
        where class prototypes (centroids) are computed from these feature embeddings
        to mitigate issues like non-IID data and class imbalance.

        Parameters:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: The feature embeddings extracted from the input images.
        """
        # Ensure the feature extractor is in evaluation mode for consistent feature extraction
        # (e.g., BatchNorm layers use global mean/variance).
        # 確保特徵提取器處於評估模式以進行一致的特徵提取
        # (例如，BatchNorm 層使用全局均值/方差)。
        self.feature_extractor.eval()
        with torch.no_grad():  # No gradients needed for feature extraction
            x = self.feature_extractor(x)

        # The feature_extractor already includes global_pool and conv_head/act2,
        # which prepares the output as a feature vector.
        # Flatten the features to `(batch_size, embedding_dim)`.
        x = x.view(x.size(0), -1)
            
        return x