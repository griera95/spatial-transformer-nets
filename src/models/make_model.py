from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def cnn_output_dim(input_dim, kernel_size, pooling_size, pooling_stride):
    """Calculate output dimension of the feature maps of convolutional layer with pooling"""
    h, w = input_dim
    kernel_h, kernel_w = kernel_size
    pooling_h, pooling_w = pooling_size

    conv_h = h - kernel_h + 1
    conv_w = w - kernel_w + 1

    out_h = (conv_h-pooling_h) // pooling_stride + 1
    out_w = (conv_w-pooling_w) // pooling_stride + 1

    return (out_h, out_w)

def localization_output_dim(cfg: DictConfig):
    """Calculate output dimension of the convolutional part of localization"""

    # first conv block
    conv1_out_h, conv1_out_w = cnn_output_dim(
        (cfg.data.height, cfg.data.width),
        (cfg.model.stn.kernel1_size, cfg.model.stn.kernel1_size), 
        (cfg.model.stn.pooling1_size, cfg.model.stn.pooling1_size),
        cfg.model.stn.pooling1_stride
    )

    # second conv block
    conv2_out_h, conv2_out_w = cnn_output_dim(
        (conv1_out_h, conv1_out_w),
        (cfg.model.stn.kernel2_size, cfg.model.stn.kernel2_size), 
        (cfg.model.stn.pooling2_size, cfg.model.stn.pooling2_size), 
        cfg.model.stn.pooling2_stride
    )

    return (conv2_out_h, conv2_out_w)


class STN(nn.Module):
    """Spatial Transformer Network"""
    def __init__(self, cfg: DictConfig):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(cfg.data.n_channels, cfg.model.conv1_channels, cfg.model.kernel1_size)
        self.conv2 = nn.Conv2d(cfg.model.conv1_channels, cfg.model.conv2_channels, cfg.model.kernel2_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, cfg.model.dense1_dim)
        self.fc2 = nn.Linear(cfg.model.dense1_dim, cfg.model.dense2_dim)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, cfg.model.stn.conv1_channels, kernel_size=cfg.model.stn.kernel1_size),
            nn.MaxPool2d(cfg.model.stn.pooling1_size, stride=cfg.model.stn.pooling1_stride),
            nn.ReLU(True),
            nn.Conv2d(cfg.model.stn.conv1_channels, cfg.model.stn.conv2_channels, kernel_size=cfg.model.stn.kernel2_size),
            nn.MaxPool2d(cfg.model.stn.pooling2_size, stride=cfg.model.stn.pooling2_stride),
            nn.ReLU(True)
        )

        # calculate output dimension of convolutional localization part
        h, w = localization_output_dim(cfg)

        # number of neurons of flattened feature maps
        self.linear_in = cfg.model.stn.conv2_channels * h * w

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.linear_in, cfg.model.stn.lin_size),
            nn.ReLU(True),
            nn.Linear(cfg.model.stn.lin_size, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.linear_in)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_model(cfg):
    """Generate model with cuda support if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STN(cfg).to(device)

    summary(model, (cfg.data.batch_size_train, 1, cfg.data.height, cfg.data.width))

    return model