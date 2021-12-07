from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def cnn_output_dim(input_dim, kernel_size, pooling_size, pooling_stride):
    """Calculate output dimension of the feature maps of convolutional layer with pooling"""
    w, h = input_dim
    kernel_w, kernel_h = kernel_size
    pooling_w, pooling_h = pooling_size

    conv_w = w - kernel_w + 1
    conv_h = h - kernel_h + 1

    out_w = (conv_w-pooling_w) // pooling_stride + 1
    out_h = (conv_h-pooling_h) // pooling_stride + 1

    return (out_w, out_h)

def localization_output_dim(cfg: DictConfig):
    """Calculate output dimension of the convolutional part of localization"""

    # first conv block
    conv1_out_w, conv1_out_h = cnn_output_dim(
        (cfg.data.height, cfg.data.width),
        (cfg.model.stn.kernel1_size, cfg.model.stn.kernel1_size), 
        (cfg.model.stn.pooling1_size, cfg.model.stn.pooling1_size),
        cfg.model.stn.pooling1_stride
    )

    # second conv block
    conv2_out_w, conv2_out_h = cnn_output_dim(
        (conv1_out_w, conv1_out_h),
        (cfg.model.stn.kernel2_size, cfg.model.stn.kernel2_size), 
        (cfg.model.stn.pooling2_size, cfg.model.stn.pooling2_size), 
        cfg.model.stn.pooling2_stride
    )

    return (conv2_out_w, conv2_out_h)

def stn_conv_output_dim(cfg: DictConfig):
    """Calculate output dimension of the conv part of stn (not localization)"""

    # first conv block
    conv1_out_w, conv1_out_h = cnn_output_dim(
        (cfg.data.height, cfg.data.width),
        (cfg.model.kernel1_size, cfg.model.kernel1_size), 
        (cfg.model.pooling1_size, cfg.model.pooling1_size),
        cfg.model.pooling1_size
    )

    # second conv block
    conv2_out_w, conv2_out_h = cnn_output_dim(
        (conv1_out_w, conv1_out_h),
        (cfg.model.kernel2_size, cfg.model.kernel2_size), 
        (cfg.model.pooling2_size, cfg.model.pooling2_size), 
        cfg.model.pooling2_size
    )

    return (conv2_out_w, conv2_out_h)

class AddCoords(nn.Module):
    def __init__(self, with_r=False, skiptile=False):
        """
        In the constructor we 
        """
        super(AddCoords, self).__init__()
        self.with_r = with_r
        self.skiptile = skiptile

    def forward(self, x):
        """
        input_tensor: (batch, 1, 1, c), or (batch, x_dim, y_dim, c)
        In the first case, first tile the input_tensor to be (batch, x_dim, y_dim, c)
        In the second case, skiptile, just concat
        """

        x_dim = x.shape[2]
        y_dim = x.shape[3]

        x = x.permute((0,2,3,1))    # put channels into second dimension

        if not self.skiptile:
            x = torch.tile(x, (1, x_dim, y_dim, 1))   # (batch, 64, 64, 2)

        batch_size_tensor = x.shape[0]                          # get batch size

        xx_ones = torch.ones([batch_size_tensor, x_dim])   # e.g. (batch, 64)
        xx_ones = xx_ones.unsqueeze(-1)                         # e.g. (batch, 64, 1)
        xx_range = torch.tile(torch.arange(y_dim).unsqueeze(0), 
                            (batch_size_tensor, 1))             # e.g. (batch, 64)
        xx_range = xx_range.unsqueeze(1).float()                # e.g. (batch, 1, 64)

        xx_channel = torch.matmul(xx_ones, xx_range)            # e.g. (batch, 64, 64)
        xx_channel = xx_channel.unsqueeze(-1)                   # e.g. (batch, 64, 64, 1)


        yy_ones = torch.ones([batch_size_tensor, y_dim])   # e.g. (batch, 64)
        yy_ones = yy_ones.unsqueeze(1)                          # e.g. (batch, 1, 64)
        yy_range = torch.tile(torch.arange(x_dim).unsqueeze(0),
                              (batch_size_tensor, 1))             # (batch, 64)
        yy_range = yy_range.unsqueeze(-1).float()               # e.g. (batch, 64, 1)

        yy_channel = torch.matmul(yy_range, yy_ones)            # e.g. (batch, 64, 64)
        yy_channel = yy_channel.unsqueeze(-1)                   # e.g. (batch, 64, 64, 1)

        xx_channel = xx_channel / (x_dim - 1)
        yy_channel = yy_channel / (y_dim - 1)
        xx_channel = xx_channel*2 - 1                           # [-1,1]
        yy_channel = yy_channel*2 - 1

        ret = torch.cat([x, 
                         xx_channel, 
                         yy_channel], axis=-1)                  # e.g. (batch, 64, 64, c+2)

        if self.with_r:
            rr = torch.sqrt(torch.square(xx_channel)
                    + torch.square(yy_channel)
                    )
            ret = torch.cat([ret, rr], axis=-1)                 # e.g. (batch, 64, 64, c+3)

        return ret.permute((0, 3, 1, 2))    # reformat for pytorch (channels in last dim)


class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, with_r=False, *args,  **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r, skiptile=True)
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class STN(nn.Module):
    """Spatial Transformer Network"""
    def __init__(self, cfg: DictConfig):
        super(STN, self).__init__()

        # select coordconv or normal convolution depending on config
        ConvLayer = CoordConv if cfg.model.coordconv else nn.Conv2d

        # if coordconv selected, input n_channels is 2 more
        n_channels_in = cfg.data.n_channels
        n_channels2 = cfg.model.conv1_channels
        n_channels2_loc = cfg.model.stn.conv1_channels

        if cfg.model.coordconv:
            n_channels_in += 2
            n_channels2 += 2
            n_channels2_loc += 2

        self.conv1 = ConvLayer(
            in_channels=n_channels_in, 
            out_channels=cfg.model.conv1_channels, 
            kernel_size=cfg.model.kernel1_size
        )
        self.conv2 = ConvLayer(
            in_channels=n_channels2, 
            out_channels=cfg.model.conv2_channels, 
            kernel_size=cfg.model.kernel2_size
        )
        self.conv2_drop = nn.Dropout2d()

        self.pooling1_size = cfg.model.pooling1_size
        self.pooling2_size = cfg.model.pooling2_size

        # calculate output dimension of convolutional part
        w, h = stn_conv_output_dim(cfg)

        # number of neurons of flattened feature maps
        self.conv_out = cfg.model.conv2_channels * w * h

        self.fc1 = nn.Linear(self.conv_out, cfg.model.dense1_dim)
        self.fc2 = nn.Linear(cfg.model.dense1_dim, cfg.model.dense2_dim)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            ConvLayer(
                in_channels=n_channels_in, 
                out_channels=cfg.model.stn.conv1_channels, 
                kernel_size=cfg.model.stn.kernel1_size
            ),
            nn.MaxPool2d(cfg.model.stn.pooling1_size, stride=cfg.model.stn.pooling1_stride),
            nn.ReLU(True),
            ConvLayer(
                in_channels=n_channels2_loc, 
                out_channels=cfg.model.stn.conv2_channels, 
                kernel_size=cfg.model.stn.kernel2_size
            ),
            nn.MaxPool2d(cfg.model.stn.pooling2_size, stride=cfg.model.stn.pooling2_stride),
            nn.ReLU(True)
        )

        # calculate output dimension of convolutional localization part
        w, h = localization_output_dim(cfg)

        # number of neurons of flattened feature maps
        self.linear_in = cfg.model.stn.conv2_channels * w * h

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

        # generate affine grid for coordinate transformation
        grid = F.affine_grid(theta, x.size())

        # spatial sampling of points based on grid
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), self.pooling1_size))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), self.pooling2_size))
        x = x.view(-1, self.conv_out)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_model(cfg):
    """Generate model with cuda support if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STN(cfg).to(device)

    summary(model, (cfg.data.batch_size_train, cfg.data.n_channels, cfg.data.height, cfg.data.width))

    return model