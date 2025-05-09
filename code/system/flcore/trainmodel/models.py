import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
from torchvision.models import resnet18, ResNet18_Weights

batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

###########################################################

# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=64*26, num_classes=6, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/digit5.py
class Digit5CNN(nn.Module):
    def __init__(self):
        super(Digit5CNN, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("gn1", nn.GroupNorm(8, 64))  # Changed to GroupNorm
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("gn2", nn.GroupNorm(8, 64))  # Changed to GroupNorm
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("gn3", nn.GroupNorm(8, 128))  # Changed to GroupNorm
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(8192, 3072))
        self.linear.add_module("gn4", nn.GroupNorm(8, 3072))  # Changed to GroupNorm
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(3072, 2048))
        self.linear.add_module("gn5", nn.GroupNorm(8, 2048))  # Changed to GroupNorm
        self.linear.add_module("relu5", nn.ReLU())

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return out

        
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.fc(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        

# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000), 
            # nn.BatchNorm1d(1000), 
            nn.ReLU(), 
            nn.Linear(1000, 500), 
            # nn.BatchNorm1d(500), 
            nn.ReLU(),
            nn.Linear(500, 100), 
            # nn.BatchNorm1d(100), 
            nn.ReLU()
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
        

# # https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
# class FedAvgCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features,
#                                32,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(32,
#                                64,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.fc1 = nn.Linear(dim, 512)
#         self.fc = nn.Linear(512, num_classes)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.act(self.conv2(x))
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         x = self.act(self.fc1(x))
#         x = self.fc(x)
#         return x

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super(FedAvgCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

    def feature_extractor(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out

class FedAvgCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgCifar10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1, bias=True),  # Changed in_features to 3, added padding=2
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1, bias=True),  # Added padding=2
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # For CIFAR-10 (32x32x3), output after conv2 is 64x8x8 (see below)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),  # Corrected dim to 4096
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

class FedAvgCifar100(nn.Module):
    def __init__(self, num_classes=100):  # Default to 100 for CIFAR-100
        super(FedAvgCifar100, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1, bias=True),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1, bias=True),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # For CIFAR-100 (32x32x3), output after conv3 is 128x4x4
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),  # Correct dim: 128 * 4 * 4 = 2048
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

#---------------------------------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out

class FedAvgCifar100ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(FedAvgCifar100ResNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

#---------------------------------------------------------------------------------------------------

class FedAvgCifar100Enhanced(nn.Module):
    def __init__(self, num_classes=100):
        super(FedAvgCifar100Enhanced, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
#---------------------------------------------------------------------------------------------------

class FedProtoCifar100(nn.Module):
    def __init__(self, num_classes=100):
        super(FedProtoCifar100, self).__init__()
        # Utilize a pretrained ResNet-18 backbone for feature extraction
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

#---------------------------------------------------------------------------------------------------

class FedAvgCNN2(nn.Module): #for CelebA Dataset
    def __init__(self, in_features, num_classes):
        super(FedAvgCNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2

        # Assuming input images are 128x128 pixels
        self.output_size = self._get_conv_output_size(input_height=128, input_width=128)

        # Fully connected layers
        self.fc1 = nn.Linear(self.output_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output_size(self, input_height, input_width):
        x = torch.rand(1, 3, input_height, input_width)  # Adjust for the number of channels
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        return x.numel()  # Total number of elements in the tensor

    def forward(self, x):
        # Ensure x is in the shape (batch_size, channels, height, width)
        x = x.view(x.size(0), 3, 128, 128)  # Change 3 to your actual number of input channels if different
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model for CelebA
class FedAvgCNN2_Light(nn.Module): 
    def __init__(self, in_features, num_classes):
        super(FedAvgCNN2_Light, self).__init__()
        # Convolutional layers with fewer filters
        self.conv1 = nn.Conv2d(in_features, 16, kernel_size=3, stride=1, padding=1)  # 32 -> 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)           # 64 -> 32 filters
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2

        # Compute output size of the convolutional layers
        self.output_size = self._get_conv_output_size(input_height=128, input_width=128)

        # Reduced size of fully connected layer
        self.fc1 = nn.Linear(self.output_size, 128)  # 512 -> 128 neurons
        self.fc2 = nn.Linear(128, num_classes)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

    def _get_conv_output_size(self, input_height, input_width):
        x = torch.rand(1, 3, input_height, input_width)  # 3 channels for RGB input
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = x.view(x.size(0), 3, 128, 128)  # Input assumed to be (batch_size, channels, height, width)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
    
# model for GTSRB and SVHN 
class FedAvgCNN3(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super(FedAvgCNN3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# 定義新的多標籤分類 CNN 類，適用於 CelebA 資料集（32x32 圖片）
class CelebAMultiLabelCNN(nn.Module):
    def __init__(self, in_features=3, num_attributes=40, dim=1600):
        """
        初始化 CelebA 多標籤分類 CNN，適配 32x32 輸入圖片。
        
        參數：
            in_features (int): 輸入通道數，預設為 3（RGB 圖片）
            num_attributes (int): 屬性數量，CelebA 為 40
            dim (int): 全連接層前的展平維度，預設為 1600（基於 32x32 輸入）
        """
        super(CelebAMultiLabelCNN, self).__init__()
        
        # 第一個卷積層：輸入通道為 3，輸出通道為 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.GroupNorm(8, 32),  # 使用 GroupNorm 正規化，提升穩定性
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # 池化後尺寸從 28x28 減至 14x14
        )
        
        # 第二個卷積層：輸入通道為 32，輸出通道為 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.GroupNorm(8, 64),  # 使用 GroupNorm 正規化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # 池化後尺寸從 10x10 減至 5x5
        )
        
        # 第一個全連接層：將卷積輸出展平後（5x5x64=1600）映射到 512 維
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),  # dim 從 1024 改為 1600
            nn.ReLU(inplace=True)
        )
        
        # 輸出層：映射到 40 個屬性（無激活函數，使用 BCEWithLogitsLoss）
        self.fc = nn.Linear(512, num_attributes)

    def forward(self, x):
        """
        前向傳播。
        
        參數：
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, in_features, 32, 32)
        
        返回：
            torch.Tensor: 輸出張量，形狀為 (batch_size, num_attributes)
        """
        out = self.conv1(x)  # 第一個卷積層，輸出形狀 (batch_size, 32, 14, 14)
        out = self.conv2(out)  # 第二個卷積層，輸出形狀 (batch_size, 64, 5, 5)
        out = torch.flatten(out, 1)  # 展平為 (batch_size, 1600)
        out = self.fc1(out)  # 第一個全連接層，輸出 (batch_size, 512)
        out = self.fc(out)  # 輸出層，返回 40 個屬性的 logits，形狀 (batch_size, 40)
        return out

# ====================================================================================================================

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================================================================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1*28*28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('GroupNorm') != -1:  # Adjusted for GroupNorm
        nn.init.ones_(m.weight)  # Setting weight to 1 for GroupNorm
        nn.init.zeros_(m.bias)   # GroupNorm does not have bias, but we can set it to zero
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.group_norm = nn.GroupNorm(10, bottleneck_dim)  # Using GroupNorm instead of BatchNorm
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.group_norm(x)  # Changed to GroupNorm
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]
                            
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ====================================================================================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        dims = hidden_dim*2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        text, text_lengths = x
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:,-1,:])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
            
        return out

# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out

# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.8, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels*len(kernel_size), num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text).permute(0,2,1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out

# ====================================================================================================================


# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input
  
#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output