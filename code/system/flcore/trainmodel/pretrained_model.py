import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights

batch_size = 10

class FedProtoCINIC10_ShuffleNetV2(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CINIC-10,
    using a pre-trained **ShuffleNetV2 0.5x** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. This significantly
        lowers FLOPs, memory usage, and communication costs.
    2.  **Leverage Pre-training:** Still benefits from rich features learned on ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CINIC-10 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **ShuffleNetV2 0.5x** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。這顯著降低了 FLOPs、內存使用量和通信成本。
    2.  **利用預訓練：** 仍然受益於在 ImageNet 上學習到的豐富特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    def __init__(self, num_classes=10):
        """
        Initializes the FedProtoCINIC10_ShuffleNetV2 with a pre-trained ShuffleNetV2 0.5x backbone.

        Parameters:
            num_classes (int): The number of output classes. For CINIC-10, this is 10.
        """
        super(FedProtoCINIC10_ShuffleNetV2, self).__init__()

        # Load pre-trained ShuffleNetV2 0.5x model weights from ImageNet.
        # 載入預訓練的 ShuffleNetV2 0.5x 模型權重 (來自 ImageNet)。
        backbone = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)

        # --- Adaptation for 32x32 input for ShuffleNetV2 without explicit resizing ---
        # ShuffleNetV2's first convolutional layer (`conv1`) typically has a stride of 2,
        # which aggressively downsamples inputs. For small 32x32 images, this can lead to loss of vital
        # spatial information. We modify its stride to 1 to retain more resolution in early layers.
        # Note: This will re-initialize the weights of this specific convolutional layer,
        # but the rest of the backbone maintains its pre-trained weights.

        # ShuffleNetV2 的第一個卷積層 (`conv1`) 通常步幅為 2，
        # 這會激進地下採樣輸入。對於 32x32 的小圖像，這可能導致丟失重要的空間信息。
        # 我們將其步幅更改為 1，以在早期層中保留更多分辨率。
        # 注意：這將重新初始化此特定卷積層的權重，
        # 但主幹的其餘部分仍保留其預訓練權重。
        
        original_first_conv_block = backbone.conv1 # This is an nn.Sequential block
        
        # Access the Conv2d layer within the Sequential block
        original_first_conv = original_first_conv_block[0] 
        
        # Create a new Conv2d layer with stride=1
        new_first_conv = nn.Conv2d(
            in_channels=original_first_conv.in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
            padding=original_first_conv.padding,
            bias=original_first_conv.bias
        )
        
        # Replace the first Conv2d layer in the original Sequential block
        backbone.conv1 = nn.Sequential(
            new_first_conv,
            original_first_conv_block[1], # BatchNorm
            original_first_conv_block[2]  # ReLU
        )
        
        # Also adjust the max pooling layer that typically follows the first conv.
        # For 32x32 input with stride 1, if maxpool stride remains 2, it still downsamples too much.
        # We replace it with an Identity to effectively remove it, relying on subsequent layers for downsampling.
        # 同時調整通常跟隨第一個卷積層的最大池化層。
        # 對於步幅為 1 的 32x32 輸入，如果最大池化步幅仍為 2，它仍然會下採樣過多。
        # 我們將其替換為 Identity 以有效移除它，依賴於後續層進行下採樣。
        backbone.maxpool = nn.Identity()

        print("ShuffleNetV2 0.5x: Adapted initial conv stride and removed maxpool for 32x32 inputs (no resizing).")

        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The feature extractor for ShuffleNetV2 consists of conv1, maxpool, stages, and conv5.
        # We group these components into our feature_extractor.

        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # ShuffleNetV2 的特徵提取器由 conv1、maxpool、stages 和 conv5 組成。
        # 我們將這些組件歸類到我們的 feature_extractor 中。
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.maxpool, # This is now nn.Identity()
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
            backbone.conv5
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Set requires_grad to False to freeze parameters
        print("ShuffleNetV2 0.5x backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # The original ShuffleNetV2 classifier is a single linear layer (`fc`).
        # We'll replace it with a new linear layer suitable for our `num_classes`.
        # This is the only part of the model whose parameters will be updated during client training.

        # 定義新的、可訓練的分類頭。
        # 原始 ShuffleNetV2 分類器是一個單個線性層 (`fc`)。
        # 我們將用一個適合我們 `num_classes` 的新線性層替換它。
        # 這是模型中唯一在客戶端訓練期間更新其參數的部分。

        # Get the input feature dimension for the new classifier from the original model's last layer.
        feature_dim = backbone.fc.in_features
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

        # Apply Adaptive Average Pooling to ensure 1x1 spatial dimensions before flattening.
        # This makes the output feature vector size independent of the input spatial dimensions
        # and prepares it for the linear classification layer.
        # 應用自適應平均池化以確保在展平之前具有 1x1 的空間維度。
        # 這使得輸出特徵向量的大小與輸入空間維度無關，
        # 並為線性分類層做好準備。
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten the tensor from (batch_size, channels, 1, 1) to (batch_size, channels).
        # 展平張量，從 (batch_size, channels, 1, 1) 展平為 (batch_size, channels)。
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

        # Apply AdaptiveAvgPool2d here as well for consistency in feature extraction.
        # 在此處也應用 AdaptiveAvgPool2d 以保持特徵提取的一致性。
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten the features to (batch_size, embedding_dim).
        # 將特徵展平為 (batch_size, embedding_dim)。
        x = x.view(x.size(0), -1)
        return x
    
#---------------------------------------------------------------------------------------------------

# Import SqueezeNet model
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

class FedProtoCINIC10_SqueezeNet(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CINIC-10.
    This model uses a pre-trained **SqueezeNet 1.1** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. SqueezeNet
        is inherently efficient, which further helps lower FLOPs, memory usage, and
        communication costs on client devices.
    2.  **Leverage Pre-training:** Still benefits from rich features learned on ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CINIC-10 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **SqueezeNet 1.1** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。SqueezeNet 本身效率很高，
        這進一步有助於降低客戶端設備上的 FLOPs、內存使用量和通信成本。
    2.  **利用預訓練：** 仍然受益於在 ImageNet 上學習到的豐富特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    def __init__(self, num_classes=10):
        """
        Initializes the FedProtoCINIC10_SqueezeNet with a pre-trained SqueezeNet 1.1 backbone.

        Parameters:
            num_classes (int): The number of output classes. For CINIC-10, this is 10.
        """
        super(FedProtoCINIC10_SqueezeNet, self).__init__()
        
        # Load pre-trained SqueezeNet 1.1 model weights from ImageNet.
        # 載入預訓練的 SqueezeNet 1.1 模型權重 (來自 ImageNet)。
        backbone = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        print("SqueezeNet 1.1 backbone loaded.")

        # --- Adaptation for 32x32 input for SqueezeNet ---
        # SqueezeNet's first convolutional layer (`features[0]`) has a stride of 2,
        # followed by a MaxPool2d (`features[2]`) with stride 2. This is too aggressive
        # for 32x32 images. We'll modify the initial conv stride to 1 and potentially
        # remove or reduce the first maxpool's stride to retain more resolution.
        
        # SqueezeNet 的第一個卷積層 (`features[0]`) 步幅為 2，
        # 接著是步幅為 2 的 MaxPool2d (`features[2]`)。這對於 32x32 圖像來說過於激進。
        # 我們將初始卷積層的步幅更改為 1，並可能移除或減小第一個最大池化層的步幅，以保留更多分辨率。

        # Get the original first Conv2d layer
        original_first_conv = backbone.features[0] 
        
        # Create a new Conv2d layer with stride=1
        backbone.features[0] = nn.Conv2d(
            in_channels=original_first_conv.in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
            padding=original_first_conv.padding,
            # FIX: Check if original_first_conv.bias is not None to get a boolean value
            bias=original_first_conv.bias is not None
        )
        
        # Remove or modify the first MaxPool2d layer (features[2])
        # For 32x32 input with initial conv stride 1, a MaxPool2d with kernel_size=3, stride=2, padding=1
        # would still heavily downsample. Replacing it with Identity or a smaller stride is better.
        # Replacing with Identity means we rely on subsequent Fire modules for downsampling.
        backbone.features[2] = nn.Identity() 

        print("SqueezeNet: Adapted initial conv stride and removed first maxpool for 32x32 inputs.")
        
        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The `features` module of SqueezeNet acts as the feature extractor.
        
        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # SqueezeNet 的 `features` 模組作為特徵提取器。
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Set requires_grad to False to freeze parameters
        print("SqueezeNet backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # SqueezeNet's original classifier is a Sequential module ending with a Conv2d.
        # The last convolutional layer in the `features` module (e.g., `features[12]`)
        # outputs 512 channels before global pooling.
        # We replace the entire classifier of SqueezeNet with a simple linear layer.
        
        # 定義新的、可訓練的分類頭。
        # SqueezeNet 的原始分類器是一個以 Conv2d 結尾的 Sequential 模組。
        # `features` 模組中的最後一個卷積層 (例如，`features[12]`) 在全局池化之前輸出 512 個通道。
        # 我們將 SqueezeNet 的整個分類器替換為一個簡單的線性層。
        
        # The output channels of the last Fire module (features[12]) are usually 512 for SqueezeNet 1.1
        # The original classifier's first layer is features[12] (Conv2d with 512 out_channels in SqueezeNet1_1)
        # However, it's more robust to grab the actual last output channel from the feature_extractor.
        # For SqueezeNet 1.1, the output channel before the final classifier part is 512.
        feature_dim = 512 # Standard output channels from SqueezeNet's feature extractor
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
        # 即使模型在訓練模式，這些凍結層也應行為如評估模式
        self.feature_extractor.eval() 
        with torch.no_grad(): # Ensure no gradients are computed for the frozen backbone
            x = self.feature_extractor(x)
        
        # Apply Adaptive Average Pooling to ensure 1x1 spatial dimensions before flattening.
        # This makes the output feature vector size independent of the input spatial dimensions
        # and prepares it for the linear classification layer.
        # 應用自適應平均池化以確保在展平之前具有 1x1 的空間維度。
        # 這使得輸出特徵向量的大小與輸入空間維度無關，
        # 並為線性分類層做好準備。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the tensor from (batch_size, channels, 1, 1) to (batch_size, channels).
        # 展平張量，從 (batch_size, channels, 1, 1) 展平為 (batch_size, channels)。
        x = x.view(x.size(0), -1)
        
        # Pass the flattened features through the trainable classification layer
        # 將展平後的特徵通過可訓練的分類層
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
        # (e.g., BatchNorm layers use global mean/variance)
        self.feature_extractor.eval() 
        with torch.no_grad(): # No gradients needed for feature extraction
            x = self.feature_extractor(x)
        
        # Apply AdaptiveAvgPool2d here as well for consistency in feature extraction.
        # 在此處也應用 AdaptiveAvgPool2d 以保持特徵提取的一致性。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the features to (batch_size, embedding_dim)
        # 將特徵展平為 (batch_size, embedding_dim)
        x = x.view(x.size(0), -1)
        return x

#---------------------------------------------------------------------------------------------------

# Import EfficientNetV2-S model
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class FedProtoCINIC10_EfficientNetV2S(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CINIC-10.
    This model uses a pre-trained **EfficientNetV2-S** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. EfficientNetV2-S
        offers a great balance of performance and efficiency, suitable for FL clients.
    2.  **Leverage Pre-training:** Still benefits from rich features learned on ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CINIC-10 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **EfficientNetV2-S** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。EfficientNetV2-S
        在性能和效率之間提供了很好的平衡，適合聯邦學習客戶端。
    2.  **利用預訓練：** 仍然受益於在 ImageNet 上學習到的豐富特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    def __init__(self, num_classes=10):
        """
        Initializes the FedProtoCINIC10_EfficientNetV2S with a pre-trained EfficientNetV2-S backbone.

        Parameters:
            num_classes (int): The number of output classes. For CINIC-10, this is 10.
        """
        super(FedProtoCINIC10_EfficientNetV2S, self).__init__()
        
        # Load pre-trained EfficientNetV2-S model weights from ImageNet.
        # 載入預訓練的 EfficientNetV2-S 模型權重 (來自 ImageNet)。
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        print("EfficientNetV2-S backbone loaded.")

        # --- Adaptation for 32x32 input for EfficientNetV2-S ---
        # EfficientNetV2-S's initial convolutional layer (`features[0]`) has a stride of 2.
        # For small 32x32 images, this is too aggressive. We'll modify its stride to 1.
        # This modification re-initializes the weights of this specific convolutional layer,
        # but the rest of the backbone maintains its pre-trained weights.

        # EfficientNetV2-S 的第一個卷積層 (`features[0]`) 步幅為 2。
        # 對於 32x32 的小圖像，這過於激進。我們將其步幅更改為 1。
        # 注意：這將重新初始化此特定卷積層的權重，
        # 但主幹的其餘部分仍保留其預訓練權重。
        
        original_first_conv = backbone.features[0] 
        
        # EfficientNetV2-S's first layer is typically an `nn.Sequential` containing a Conv2d and BatchNorm/Activation.
        # We need to access and modify the Conv2d directly within that Sequential.
        # Assuming the Conv2d is the first module in features[0]
        conv2d_in_features0 = original_first_conv[0]

        backbone.features[0] = nn.Sequential(
            nn.Conv2d(
                in_channels=conv2d_in_features0.in_channels,
                out_channels=conv2d_in_features0.out_channels,
                kernel_size=conv2d_in_features0.kernel_size,
                stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
                padding=conv2d_in_features0.padding,
                bias=conv2d_in_features0.bias
            ),
            *list(original_first_conv.children())[1:] # Keep the subsequent layers (BatchNorm, activation)
        )
        print("EfficientNetV2-S: Adapted initial conv stride for 32x32 inputs (no resizing).")
        
        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The `features` module of EfficientNetV2-S acts as the feature extractor.
        
        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # EfficientNetV2-S 的 `features` 模組作為特徵提取器。
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Set requires_grad to False to freeze parameters
        print("EfficientNetV2-S backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # The original EfficientNetV2-S classifier (`backbone.classifier`) usually
        # consists of a `Sequential` with `Dropout` and a `Linear` layer.
        # We'll replace the entire classifier with a single linear layer suitable for `num_classes`.
        
        # 定義新的、可訓練的分類頭。
        # 原始 EfficientNetV2-S 分類器 (`backbone.classifier`) 通常由
        # 帶有 `Dropout` 和 `Linear` 層的 `Sequential` 組成。
        # 我們將用一個適合 `num_classes` 的單個線性層替換整個分類器。
        
        # Get the input feature dimension for the new classifier from the original model's last layer.
        # The first layer in backbone.classifier is typically the Dropout, then the Linear layer.
        # So, backbone.classifier[1] should be the Linear layer, and we need its in_features.
        feature_dim = backbone.classifier[1].in_features 
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
        # 即使模型在訓練模式，這些凍結層也應行為如評估模式
        self.feature_extractor.eval() 
        with torch.no_grad(): # Ensure no gradients are computed for the frozen backbone
            x = self.feature_extractor(x)
        
        # Apply Adaptive Average Pooling to ensure 1x1 spatial dimensions before flattening.
        # This makes the output feature vector size independent of the input spatial dimensions
        # and prepares it for the linear classification layer.
        # 應用自適應平均池化以確保在展平之前具有 1x1 的空間維度。
        # 這使得輸出特徵向量的大小與輸入空間維度無關，
        # 並為線性分類層做好準備。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the tensor from (batch_size, channels, 1, 1) to (batch_size, channels).
        # 展平張量，從 (batch_size, channels, 1, 1) 展平為 (batch_size, channels)。
        x = x.view(x.size(0), -1)
        
        # Pass the flattened features through the trainable classification layer
        # 將展平後的特徵通過可訓練的分類層
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
        # (e.g., BatchNorm layers use global mean/variance)
        self.feature_extractor.eval() 
        with torch.no_grad(): # No gradients needed for feature extraction
            x = self.feature_extractor(x)
        
        # Apply AdaptiveAvgPool2d here as well for consistency in feature extraction.
        # 在此處也應用 AdaptiveAvgPool2d 以保持特徵提取的一致性。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the features to (batch_size, embedding_dim)
        # 將特徵展平為 (batch_size, embedding_dim)
        x = x.view(x.size(0), -1)
        return x
#---------------------------------------------------------------------------------------------------

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class FedProtoCifar100_MobileNetV2_new(nn.Module):
    def __init__(self, num_classes=100):
        super(FedProtoCifar100_MobileNetV2_new, self).__init__()
        # Utilize a pretrained MobileNetV2 backbone for feature extraction
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Remove the final classification layer.
        # MobileNetV2's feature extraction part is typically everything before the 'classifier' module.
        self.feature_extractor = mobilenet.features
        
        # The last layer in mobilenet.features (features.18) outputs 1280 channels.
        # This will be the input to our new classification head.
        self.fc = nn.Linear(mobilenet.last_channel, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        # MobileNetV2's feature extractor output needs to be flattened.
        # It typically ends with a 1x1 average pooling, so we can use adaptive average pooling
        # before flattening for consistency with how MobileNetV2 is often used,
        # or simply flatten the global average pooled output.
        # The original MobileNetV2 applies an adaptive average pool as part of its forward pass
        # before the classifier. We'll replicate that.
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1) # Flatten the tensor
        x = self.fc(x)
        return x

class FedProtoCIFAR100_MobileNetV2(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CIFAR-100.
    This model uses a pre-trained **MobileNetV2** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. This significantly
        lowers FLOPs, memory usage, and communication costs, ideal for FL clients.
    2.  **Leverage Pre-training:** Still benefits from rich features learned on ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CIFAR-100 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **MobileNetV2** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。這顯著降低了 FLOPs、內存使用量和通信成本，
        非常適合聯邦學習客戶端。
    2.  **利用預訓練：** 仍然受益於在 ImageNet 上學習到的豐富特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    def __init__(self, num_classes=100):
        """
        Initializes the FedProtoCIFAR100_MobileNetV2 with a pre-trained MobileNetV2 backbone.

        Parameters:
            num_classes (int): The number of output classes. For CIFAR-100, this is 100.
        """
        super(FedProtoCIFAR100_MobileNetV2, self).__init__()
        
        # Load pre-trained MobileNetV2 model weights from ImageNet.
        # 載入預訓練的 MobileNetV2 模型權重 (來自 ImageNet)。
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        print("MobileNetV2 backbone loaded.")

        # --- Adaptation for 32x32 input for MobileNetV2 ---
        # MobileNetV2's first layer (`features[0]`, which is an InvertedResidual)
        # starts with a Conv2d that typically has a stride of 2.
        # For small 32x32 images, this is too aggressive. We'll modify its stride to 1.
        # Note: This will re-initialize the weights of this specific convolutional layer,
        # but the rest of the backbone maintains its pre-trained weights.

        # MobileNetV2 的第一個層 (`features[0]`，是一個 InvertedResidual)
        # 以通常步幅為 2 的 Conv2d 開始。
        # 對於 32x32 的小圖像，這過於激進。我們將其步幅更改為 1。
        # 注意：這將重新初始化此特定卷積層的權重，
        # 但主幹的其餘部分仍保留其預訓練權重。
        
        # Access the Conv2d inside the first InvertedResidual block
        original_first_conv = backbone.features[0][0]
        
        # Create a new Conv2d layer with stride=1
        new_first_conv = nn.Conv2d(
            in_channels=original_first_conv.in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
            padding=original_first_conv.padding,
            bias=original_first_conv.bias
        )
        # Replace the original Conv2d within the InvertedResidual block
        backbone.features[0][0] = new_first_conv
        
        print("MobileNetV2: Adapted initial conv stride for 32x32 inputs (no resizing).")
        
        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The `features` module of MobileNetV2 acts as the feature extractor.
        
        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # MobileNetV2 的 `features` 模組作為特徵提取器。
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Set requires_grad to False to freeze parameters
        print("MobileNetV2 backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # The original MobileNetV2 classifier (`backbone.classifier`) usually
        # consists of a `Sequential` with a `Linear` layer.
        # We'll replace the entire classifier with a single linear layer suitable for `num_classes`.
        
        # 定義新的、可訓練的分類頭。
        # 原始 MobileNetV2 分類器 (`backbone.classifier`) 通常由
        # 帶有 `Linear` 層的 `Sequential` 組成。
        # 我們將用一個適合 `num_classes` 的單個線性層替換整個分類器。
        
        # Get the input feature dimension for the new classifier from the original model's last layer.
        # The last layer in MobileNetV2's features before the classifier is typically a Conv2d
        # that outputs 1280 channels.
        # The classifier itself starts with a Conv2d followed by a Linear layer.
        # The input to the final Linear layer (backbone.classifier[1]) will be 1280.
        feature_dim = backbone.classifier[1].in_features # This will be 1280 for standard MobileNetV2
        self.classifier = nn.Linear(feature_dim, num_classes) # num_classes will be 100 for CIFAR-100
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
        # 即使模型在訓練模式，這些凍結層也應行為如評估模式
        self.feature_extractor.eval() 
        with torch.no_grad(): # Ensure no gradients are computed for the frozen backbone
            x = self.feature_extractor(x)
        
        # Apply Adaptive Average Pooling to ensure 1x1 spatial dimensions before flattening.
        # This makes the output feature vector size independent of the input spatial dimensions
        # and prepares it for the linear classification layer.
        # 應用自適應平均池化以確保在展平之前具有 1x1 的空間維度。
        # 這使得輸出特徵向量的大小與輸入空間維度無關，
        # 並為線性分類層做好準備。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the tensor from (batch_size, channels, 1, 1) to (batch_size, channels).
        # 展平張量，從 (batch_size, channels, 1, 1) 展平為 (batch_size, channels)。
        x = x.view(x.size(0), -1)
        
        # Pass the flattened features through the trainable classification layer
        # 將展平後的特徵通過可訓練的分類層
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
        # (e.g., BatchNorm layers use global mean/variance)
        self.feature_extractor.eval() 
        with torch.no_grad(): # No gradients needed for feature extraction
            x = self.feature_extractor(x)
        
        # Apply AdaptiveAvgPool2d here as well for consistency in feature extraction.
        # 在此處也應用 AdaptiveAvgPool2d 以保持特徵提取的一致性。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the features to (batch_size, embedding_dim)
        # 將特徵展平為 (batch_size, embedding_dim)
        x = x.view(x.size(0), -1)
        return x

#---------------------------------------------------------------------------------------------------

class FedProtoCIFAR100_EfficientNetV2S(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CIFAR-100.
    This model uses a pre-trained **EfficientNetV2-S** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. EfficientNetV2-S
        offers a great balance of performance and efficiency, suitable for FL clients.
    2.  **Leverage Pre-training:** Still benefits from rich features learned on ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CIFAR-100 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **EfficientNetV2-S** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。EfficientNetV2-S
        在性能和效率之間提供了很好的平衡，適合聯邦學習客戶端。
    2.  **利用預訓練：** 仍然受益於在 ImageNet 上學習到的豐富特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    # Renamed class to reflect CIFAR-100 dataset
    def __init__(self, num_classes=100): # KEY CHANGE: Default num_classes set to 100 for CIFAR-100
        """
        Initializes the FedProtoCIFAR100_EfficientNetV2S with a pre-trained EfficientNetV2-S backbone.

        Parameters:
            num_classes (int): The number of output classes. For CIFAR-100, this is 100.
        """
        super(FedProtoCIFAR100_EfficientNetV2S, self).__init__()
        
        # Load pre-trained EfficientNetV2-S model weights from ImageNet.
        # 載入預訓練的 EfficientNetV2-S 模型權重 (來自 ImageNet)。
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        print("EfficientNetV2-S backbone loaded.")

        # --- Adaptation for 32x32 input for EfficientNetV2-S ---
        # EfficientNetV2-S's initial convolutional layer (`features[0]`) has a stride of 2.
        # For small 32x32 images, this is too aggressive. We'll modify its stride to 1.
        # This modification re-initializes the weights of this specific convolutional layer,
        # but the rest of the backbone maintains its pre-trained weights.

        # EfficientNetV2-S 的第一個卷積層 (`features[0]`) 步幅為 2。
        # 對於 32x32 的小圖像，這過於激進。我們將其步幅更改為 1。
        # 注意：這將重新初始化此特定卷積層的權重，
        # 但主幹的其餘部分仍保留其預訓練權重。
        
        original_first_conv = backbone.features[0] 
        
        # EfficientNetV2-S's first layer is typically an `nn.Sequential` containing a Conv2d and BatchNorm/Activation.
        # We need to access and modify the Conv2d directly within that Sequential.
        # Assuming the Conv2d is the first module in features[0]
        conv2d_in_features0 = original_first_conv[0]

        backbone.features[0] = nn.Sequential(
            nn.Conv2d(
                in_channels=conv2d_in_features0.in_channels,
                out_channels=conv2d_in_features0.out_channels,
                kernel_size=conv2d_in_features0.kernel_size,
                stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
                padding=conv2d_in_features0.padding,
                bias=conv2d_in_features0.bias
            ),
            *list(original_first_conv.children())[1:] # Keep the subsequent layers (BatchNorm, activation)
        )
        print("EfficientNetV2-S: Adapted initial conv stride for 32x32 inputs (no resizing).")
        
        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The `features` module of EfficientNetV2-S acts as the feature extractor.
        
        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # EfficientNetV2-S 的 `features` 模組作為特徵提取器。
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Set requires_grad to False to freeze parameters
        print("EfficientNetV2-S backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # The original EfficientNetV2-S classifier (`backbone.classifier`) usually
        # consists of a `Sequential` with `Dropout` and a `Linear` layer.
        # We'll replace the entire classifier with a single linear layer suitable for `num_classes`.
        
        # 定義新的、可訓練的分類頭。
        # 原始 EfficientNetV2-S 分類器 (`backbone.classifier`) 通常由
        # 帶有 `Dropout` 和 `Linear` 層的 `Sequential` 組成。
        # 我們將用一個適合 `num_classes` 的單個線性層替換整個分類器。
        
        # Get the input feature dimension for the new classifier from the original model's last layer.
        # The first layer in backbone.classifier is typically the Dropout, then the Linear layer.
        # So, backbone.classifier[1] should be the Linear layer, and we need its in_features.
        feature_dim = backbone.classifier[1].in_features 
        self.classifier = nn.Linear(feature_dim, num_classes) # KEY CHANGE: num_classes passed here will be 100
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
        # 即使模型在訓練模式，這些凍結層也應行為如評估模式
        self.feature_extractor.eval() 
        with torch.no_grad(): # Ensure no gradients are computed for the frozen backbone
            x = self.feature_extractor(x)
        
        # Apply Adaptive Average Pooling to ensure 1x1 spatial dimensions before flattening.
        # This makes the output feature vector size independent of the input spatial dimensions
        # and prepares it for the linear classification layer.
        # 應用自適應平均池化以確保在展平之前具有 1x1 的空間維度。
        # 這使得輸出特徵向量的大小與輸入空間維度無關，
        # 並為線性分類層做好準備。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the tensor from (batch_size, channels, 1, 1) to (batch_size, channels).
        # 展平張量，從 (batch_size, channels, 1, 1) 展平為 (batch_size, channels)。
        x = x.view(x.size(0), -1)
        
        # Pass the flattened features through the trainable classification layer
        # 將展平後的特徵通過可訓練的分類層
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
        # (e.g., BatchNorm layers use global mean/variance)
        self.feature_extractor.eval() 
        with torch.no_grad(): # No gradients needed for feature extraction
            x = self.feature_extractor(x)
        
        # Apply AdaptiveAvgPool2d here as well for consistency in feature extraction.
        # 在此處也應用 AdaptiveAvgPool2d 以保持特徵提取的一致性。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the features to (batch_size, embedding_dim)
        # 將特徵展平為 (batch_size, embedding_dim)
        x = x.view(x.size(0), -1)
        return x

#---------------------------------------------------------------------------------------------------

class FedProtoCIFAR100_SqueezeNet(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CIFAR-100.
    This model uses a pre-trained **SqueezeNet 1.1** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. SqueezeNet
        is inherently efficient, which further helps lower FLOPs, memory usage, and
        communication costs on client devices.
    2.  **Leverage Pre-training:** Still benefits from rich features learned on ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CIFAR-100 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **SqueezeNet 1.1** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。SqueezeNet 本身效率很高，
        這進一步有助於降低客戶端設備上的 FLOPs、內存使用量和通信成本。
    2.  **利用預訓練：** 仍然受益於在 ImageNet 上學習到的豐富特徵。
    3.  **提高小圖像性能：：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    # Renamed class to reflect CIFAR-100 dataset
    def __init__(self, num_classes=100): # KEY CHANGE: Default num_classes set to 100 for CIFAR-100
        """
        Initializes the FedProtoCIFAR100_SqueezeNet with a pre-trained SqueezeNet 1.1 backbone.

        Parameters:
            num_classes (int): The number of output classes. For CIFAR-100, this is 100.
        """
        super(FedProtoCIFAR100_SqueezeNet, self).__init__()
        
        # Load pre-trained SqueezeNet 1.1 model weights from ImageNet.
        # 載入預訓練的 SqueezeNet 1.1 模型權重 (來自 ImageNet)。
        backbone = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        print("SqueezeNet 1.1 backbone loaded.")

        # --- Adaptation for 32x32 input for SqueezeNet ---
        # SqueezeNet's first convolutional layer (`features[0]`) has a stride of 2,
        # followed by a MaxPool2d (`features[2]`) with stride 2. This is too aggressive
        # for 32x32 images. We'll modify the initial conv stride to 1 and potentially
        # remove or reduce the first maxpool's stride to retain more resolution.
        
        # SqueezeNet 的第一個卷積層 (`features[0]`) 步幅為 2，
        # 接著是步幅為 2 的 MaxPool2d (`features[2]`)。這對於 32x32 圖像來說過於激進。
        # 我們將初始卷積層的步幅更改為 1，並可能移除或減小第一個最大池化層的步幅，以保留更多分辨率。

        # Get the original first Conv2d layer
        original_first_conv = backbone.features[0] 
        
        # Create a new Conv2d layer with stride=1
        backbone.features[0] = nn.Conv2d(
            in_channels=original_first_conv.in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
            padding=original_first_conv.padding,
            # Ensure 'bias' argument is a boolean. original_first_conv.bias is a Tensor, so check if it's not None.
            bias=original_first_conv.bias is not None 
        )
        
        # Remove or modify the first MaxPool2d layer (features[2])
        # For 32x32 input with initial conv stride 1, a MaxPool2d with kernel_size=3, stride=2, padding=1
        # would still heavily downsample. Replacing it with Identity or a smaller stride is better.
        # Replacing with Identity means we rely on subsequent Fire modules for downsampling.
        backbone.features[2] = nn.Identity() 

        print("SqueezeNet: Adapted initial conv stride and removed first maxpool for 32x32 inputs.")
        
        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The `features` module of SqueezeNet acts as the feature extractor.
        
        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # SqueezeNet 的 `features` 模組作為特徵提取器。
        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Set requires_grad to False to freeze parameters
        print("SqueezeNet backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # SqueezeNet's original classifier is a Sequential module ending with a Conv2d.
        # The last convolutional layer in the `features` module (e.g., `features[12]`)
        # outputs 512 channels before global pooling.
        # We replace the entire classifier of SqueezeNet with a simple linear layer.
        
        # 定義新的、可訓練的分類頭。
        # SqueezeNet 的原始分類器是一個以 Conv2d 結尾的 Sequential 模組。
        # `features` 模組中的最後一個卷積層 (例如，`features[12]`) 在全局池化之前輸出 512 個通道。
        # 我們將 SqueezeNet 的整個分類器替換為一個簡單的線性層。
        
        # The output channels of the last Fire module (features[12]) are usually 512 for SqueezeNet 1.1
        # The original classifier's first layer is features[12] (Conv2d with 512 out_channels in SqueezeNet1_1)
        # However, it's more robust to grab the actual last output channel from the feature_extractor.
        # For SqueezeNet 1.1, the output channel before the final classifier part is 512.
        feature_dim = 512 # Standard output channels from SqueezeNet's feature extractor
        self.classifier = nn.Linear(feature_dim, num_classes) # KEY CHANGE: num_classes passed here will be 100
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
        # 即使模型在訓練模式，這些凍結層也應行為如評估模式
        self.feature_extractor.eval() 
        with torch.no_grad(): # Ensure no gradients are computed for the frozen backbone
            x = self.feature_extractor(x)
        
        # Apply Adaptive Average Pooling to ensure 1x1 spatial dimensions before flattening.
        # This makes the output feature vector size independent of the input spatial dimensions
        # and prepares it for the linear classification layer.
        # 應用自適應平均池化以確保在展平之前具有 1x1 的空間維度。
        # 這使得輸出特徵向量的大小與輸入空間維度無關，
        # 並為線性分類層做好準備。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the tensor from (batch_size, channels, 1, 1) to (batch_size, channels).
        # 展平張量，從 (batch_size, channels, 1, 1) 展平為 (batch_size, channels)。
        x = x.view(x.size(0), -1)
        
        # Pass the flattened features through the trainable classification layer
        # 將展平後的特徵通過可訓練的分類層
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
        # (e.g., BatchNorm layers use global mean/variance)
        self.feature_extractor.eval() 
        with torch.no_grad(): # No gradients needed for feature extraction
            x = self.feature_extractor(x)
        
        # Apply AdaptiveAvgPool2d here as well for consistency in feature extraction.
        # 在此處也應用 AdaptiveAvgPool2d 以保持特徵提取的一致性。
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        
        # Flatten the features to (batch_size, embedding_dim)
        # 將特徵展平為 (batch_size, embedding_dim)
        x = x.view(x.size(0), -1)
        return x

#---------------------------------------------------------------------------------------------------

class FedProtoCIFAR100_ShuffleNetV2(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CIFAR-100,
    using a pre-trained **ShuffleNetV2 0.5x** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. This significantly
        lowers FLOPs, memory usage, and communication costs.
    2.  **Leverage Pre-training:** Still benefits from rich features learned on ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CIFAR-100 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **ShuffleNetV2 0.5x** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。這顯著降低了 FLOPs、內存使用量和通信成本。
    2.  **利用預訓練：** 仍然受益於在 ImageNet 上學習到的豐富特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    # Renamed class to reflect CIFAR-100 dataset
    def __init__(self, num_classes=100): # KEY CHANGE: Default num_classes set to 100 for CIFAR-100
        """
        Initializes the FedProtoCIFAR100_ShuffleNetV2 with a pre-trained ShuffleNetV2 0.5x backbone.

        Parameters:
            num_classes (int): The number of output classes. For CIFAR-100, this is 100.
        """
        super(FedProtoCIFAR100_ShuffleNetV2, self).__init__()

        # Load pre-trained ShuffleNetV2 0.5x model weights from ImageNet.
        # 載入預訓練的 ShuffleNetV2 0.5x 模型權重 (來自 ImageNet)。
        backbone = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)

        # --- Adaptation for 32x32 input for ShuffleNetV2 without explicit resizing ---
        # ShuffleNetV2's first convolutional layer (`conv1`) typically has a stride of 2,
        # which aggressively downsamples inputs. For small 32x32 images, this can lead to loss of vital
        # spatial information. We modify its stride to 1 to retain more resolution in early layers.
        # Note: This will re-initialize the weights of this specific convolutional layer,
        # but the rest of the backbone maintains its pre-trained weights.

        # ShuffleNetV2 的第一個卷積層 (`conv1`) 通常步幅為 2，
        # 這會激進地下採樣輸入。對於 32x32 的小圖像，這可能導致丟失重要的空間信息。
        # 我們將其步幅更改為 1，以在早期層中保留更多分辨率。
        # 注意：這將重新初始化此特定卷積層的權重，
        # 但主幹的其餘部分仍保留其預訓練權重。
        
        original_first_conv_block = backbone.conv1 # This is an nn.Sequential block
        
        # Access the Conv2d layer within the Sequential block
        original_first_conv = original_first_conv_block[0] 
        
        # Create a new Conv2d layer with stride=1
        new_first_conv = nn.Conv2d(
            in_channels=original_first_conv.in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
            padding=original_first_conv.padding,
            bias=original_first_conv.bias
        )
        
        # Replace the first Conv2d layer in the original Sequential block
        backbone.conv1 = nn.Sequential(
            new_first_conv,
            original_first_conv_block[1], # BatchNorm
            original_first_conv_block[2]  # ReLU
        )
        
        # Also adjust the max pooling layer that typically follows the first conv.
        # For 32x32 input with stride 1, if maxpool stride remains 2, it still downsamples too much.
        # We replace it with an Identity to effectively remove it, relying on subsequent layers for downsampling.
        # 同時調整通常跟隨第一個卷積層的最大池化層。
        # 對於步幅為 1 的 32x32 輸入，如果最大池化步幅仍為 2，它仍然會下採樣過多。
        # 我們將其替換為 Identity 以有效移除它，依賴於後續層進行下採樣。
        backbone.maxpool = nn.Identity()

        print("ShuffleNetV2 0.5x: Adapted initial conv stride and removed maxpool for 32x32 inputs (no resizing).")

        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The feature extractor for ShuffleNetV2 consists of conv1, maxpool, stages, and conv5.
        # We group these components into our feature_extractor.

        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # ShuffleNetV2 的特徵提取器由 conv1、maxpool、stages 和 conv5 組成。
        # 我們將這些組件歸類到我們的 feature_extractor 中。
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.maxpool, # This is now nn.Identity()
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
            backbone.conv5
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Set requires_grad to False to freeze parameters
        print("ShuffleNetV2 0.5x backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # The original ShuffleNetV2 classifier is a single linear layer (`fc`).
        # We'll replace it with a new linear layer suitable for our `num_classes`.
        # This is the only part of the model whose parameters will be updated during client training.

        # 定義新的、可訓練的分類頭。
        # 原始 ShuffleNetV2 分類器是一個單個線性層 (`fc`)。
        # 我們將用一個適合我們 `num_classes` 的新線性層替換它。
        # 這是模型中唯一在客戶端訓練期間更新其參數的部分。

        # Get the input feature dimension for the new classifier from the original model's last layer.
        feature_dim = backbone.fc.in_features
        self.classifier = nn.Linear(feature_dim, num_classes) # KEY CHANGE: num_classes passed here will be 100
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

        # Apply Adaptive Average Pooling to ensure 1x1 spatial dimensions before flattening.
        # This makes the output feature vector size independent of the input spatial dimensions
        # and prepares it for the linear classification layer.
        # 應用自適應平均池化以確保在展平之前具有 1x1 的空間維度。
        # 這使得輸出特徵向量的大小與輸入空間維度無關，
        # 並為線性分類層做好準備。
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten the tensor from (batch_size, channels, 1, 1) to (batch_size, channels).
        # 展平張量，從 (batch_size, channels, 1, 1) 展平為 (batch_size, channels)。
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

        # Apply AdaptiveAvgPool2d here as well for consistency in feature extraction.
        # 在此處也應用 AdaptiveAvgPool2d 以保持特徵提取的一致性。
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten the features to (batch_size, embedding_dim).
        # 將特徵展平為 (batch_size, embedding_dim)。
        x = x.view(x.size(0), -1)
        return x

#---------------------------------------------------------------------------------------------------

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class FedProtoCifar100_MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=100):
        super(FedProtoCifar100_MobileNetV3_Small, self).__init__()
        # Utilize a pretrained MobileNetV3-Small backbone for feature extraction
        mobilenet_v3 = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # MobileNetV3 models have a 'features' and a 'classifier' module.
        # We take all layers up to the 'avgpool' layer within the features,
        # and then typically the 'classifier' module starts with a 1x1 conv or a linear layer.
        # For feature extraction, we want to capture the output right before the final classification head.
        # MobileNetV3's classifier usually starts with a Conv2d layer followed by pooling and then Linear.
        # The 'features' part of MobileNetV3-Small culminates in a layer that outputs 576 channels.
        self.feature_extractor = mobilenet_v3.features
        
        # The MobileNetV3 models have a `_forward_impl` method that includes an
        # `avgpool` and `flatten` before the `classifier`.
        # We need to find the output features of the `features` module before the final classifier.
        # Looking at the source, `mobilenet_v3_small` has a `last_channel` attribute
        # which is the number of channels output by the `features` block.
        self.fc = nn.Linear(mobilenet_v3.classifier[0].in_features, num_classes) # Access the in_features of the first layer in classifier


    def forward(self, x):
        x = self.feature_extractor(x)
        # MobileNetV3 also applies adaptive average pooling and flattening internally
        # before its classifier. We replicate this.
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1) # Flatten the tensor
        x = self.fc(x)
        return x

#---------------------------------------------------------------------------------------------------

from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

class FedProtoCifar100_SqueezeNet_new(nn.Module):
    def __init__(self, num_classes=100):
        super(FedProtoCifar100_SqueezeNet_new, self).__init__()
        # Utilize a pretrained SqueezeNet 1.0 backbone for feature extraction
        squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

        # The feature extractor is the 'features' module of SqueezeNet
        self.feature_extractor = squeezenet.features

        # SqueezeNet's classifier is a Sequential module.
        # The final classification layer is a Conv2d layer within this classifier.
        # For squeezenet1_0, the original Conv2d layer is at index 1 of the classifier.
        # Its input channels are 512.
        
        # We replace the classifier with our custom one.
        # The original SqueezeNet classifier looks like:
        # (classifier): Sequential(
        #   (0): Dropout(p=0.5, inplace=False)
        #   (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
        #   (2): ReLU(inplace=True)
        #   (3): AdaptiveAvgPool2d(output_size=(1, 1))
        # )
        
        # We will take the output of the features and then apply our new classifier.
        # The output channels of the last layer in `squeezenet.features` for `squeezenet1_0` is 512.
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), # Optional: keep dropout from original SqueezeNet classifier
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        x = torch.flatten(x, 1) # Flatten the tensor after the final adaptive average pooling
        return x

#---------------------------------------------------------------------------------------------------

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FedProtoCifar100_EfficientNet_B0(nn.Module):
    def __init__(self, num_classes=100):
        super(FedProtoCifar100_EfficientNet_B0, self).__init__()
        # Utilize a pretrained EfficientNet_B0 backbone for feature extraction
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # The feature extractor is the 'features' module of EfficientNet_B0
        self.feature_extractor = efficientnet.features
        
        # The EfficientNet classifier typically starts with a `nn.Linear` layer
        # after an adaptive average pooling and flatten operation.
        # We need to get the `in_features` of this initial linear layer.
        # For EfficientNet_B0, the classifier is a Sequential module, and the
        # first element (at index 1) is typically the Linear layer after dropout.
        # The in_features for EfficientNet_B0's classifier[1] is 1280.
        self.fc = nn.Linear(efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        # EfficientNet models typically use AdaptiveAvgPool2d and flatten
        # before the final classification layer within their own forward pass.
        # We need to replicate this behavior for our custom head.
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)
        return x

#---------------------------------------------------------------------------------------------------

from torchvision.models import mnasnet1_0, MNASNet1_0_Weights

class FedProtoCifar100_MNASNet(nn.Module):
    def __init__(self, num_classes=100):
        super(FedProtoCifar100_MNASNet, self).__init__()
        # Utilize a pretrained MNASNet backbone for feature extraction
        mnasnet = mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)

        # For MNASNet, the main convolutional backbone, including the
        # global average pooling, is typically contained within the 'layers' attribute.
        # The 'classifier' module then follows.
        self.feature_extractor = mnasnet.layers

        # The input features for the final classification layer come from the
        # output of the 'layers' sequence. The original MNASNet's classifier[1]
        # is the Linear layer. We can get its in_features directly.
        # This part should be correct if mnasnet.classifier[1] is indeed the Linear layer.
        self.fc = nn.Linear(mnasnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        # The 'layers' module in MNASNet usually includes the final
        # AdaptiveAvgPool2d, so the output is already pooled.
        # We just need to flatten it before passing to the FC layer.
        x = torch.flatten(x, 1)  # Flatten the tensor after the feature extraction
        x = self.fc(x)
        return x

#---------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights

class FedProtoCifar100_MNASNet_new(nn.Module):
    def __init__(self, num_classes=100):
        super(FedProtoCifar100_MNASNet_new, self).__init__()
        
        # Load the pretrained MNASNet model
        mnasnet = mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)

        # --- Modify the first convolutional layer ---
        # MNASNet's first layer is typically mnasnet.layers[0]
        original_first_conv = mnasnet.layers[0]

        # For CIFAR-100 (32x32, 3 channels):
        # We need to change the stride of the first convolutional layer
        # to (1,1) to prevent rapid downsampling.
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        new_first_conv = nn.Conv2d(
            in_channels=original_first_conv.in_channels, # Keep 3 channels for CIFAR-100
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=(1, 1), # <-- Change stride to 1x1
            padding=original_first_conv.padding,
            bias=original_first_conv.bias
        )
        
        # Optionally, transfer weights for the common part of the kernel (3 channels)
        # This is a common practice to retain some pre-trained knowledge
        new_first_conv.weight.data[:, :3, :, :] = original_first_conv.weight.data[:, :3, :, :]
        if original_first_conv.bias is not None:
            new_first_conv.bias.data = original_first_conv.bias.data

        # Replace the original first layer with our modified one
        mnasnet.layers[0] = new_first_conv

        # Assign the modified layers as the feature extractor
        self.feature_extractor = mnasnet.layers

        # The final classification layer remains the same, as its input features
        # depend on the output of the 'layers' sequence, which has been adjusted.
        self.fc = nn.Linear(mnasnet.classifier[1].in_features, num_classes)

        # Note: You still need to apply ImageNet normalization in your DataLoader
        # even if you don't resize, as the pre-trained weights expect this.

    def forward(self, x):
        # Assuming input 'x' is already normalized by the DataLoader
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # Flatten the tensor after the feature extraction
        x = self.fc(x)
        return x

#---------------------------------------------------------------------------------------------------