import torch
import torch.nn as nn
from torchviz import make_dot

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

def filtered_make_dot(output, params=None):
    """
    Custom make_dot function to filter out specific operation types.
    """
    if params is None:
        params = {}
    dot = make_dot(output, params=params)
    dot.attr('node', style='filled', shape='box', align='left', fontsize='12', ranksep='0.1', height='0.2')
    filtered_nodes = []
    for node in dot.body:
        if isinstance(node, str):
            filtered_nodes.append(node)
        else:
            label = node.attr('label')
            if label is not None:
                op_type = label.split('(')[0]
                if op_type in ['Conv2d', 'MaxPool2d', 'ReLU', 'GroupNorm', 'Linear', 'Flatten']:
                    filtered_nodes.append(node)
    dot.body = filtered_nodes
    return dot

if __name__ == '__main__':
    # Create an instance of the model
    model = FedAvgCifar100(num_classes=100)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image size

    # Use the custom filtered_make_dot function
    dot = filtered_make_dot(model(dummy_input), params=dict(model.named_parameters()))

    # Customize global graph attributes for vertical (Up to Down) layout
    dot.attr('graph', rankdir='TB')  # Top to Bottom layout
    dot.attr('node', shape='box')
    dot.attr('edge', arrowhead='vee')

    # Save the graph as a PNG image in the current directory
    dot.render("fedavg_cifar100_vertical_architecture", format="png")

    print("Filtered neural network architecture image saved as fedavg_cifar100_vertical_architecture.png in the current directory.")