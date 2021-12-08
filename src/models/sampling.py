import torch
from torch.nn.functional import pad


def affine_grid_generator(theta, size, align_corners=False):
    """Generate affine transformation for later sampling"""
    N, C, H, W = size
    
    base_grid = make_base_grid(theta, N, C, H, W, align_corners)
    
    grid = affine_grid(base_grid, theta, N, H, W)

    return grid.view([N, H, W, 2])


def affine_grid(base_grid, theta, N, H, W):
    """Performs matrix multiplication for generating affine grid"""
    return base_grid.view([N, H * W, 3]).bmm(theta.transpose(1, 2))


def affine_diffeo_grid_generator(theta, size, align_corners=False):
    """Generate affine diffeomorphism transformation for later sampling"""
    N, C, H, W = size
    
    base_grid = make_base_grid(theta, N, C, H, W, align_corners)
    
    grid = affine_diffeo_grid(base_grid, theta, N, H, W)

    return grid.view([N, H, W, 2])


def affine_diffeo_grid(base_grid, theta, N, H, W):
    """
    Generates affine diffeomorphism grid
    as in https://www2.compute.dtu.dk/~sohau/papers/cvpr2018/detlefsen_cvpr_2018.pdf
    """
    theta = pad(theta,[0,0,0,1])
    # calculate matrix exponential to obtain velocity field
    theta = torch.matrix_exp(theta)

    res = base_grid.view([N, H * W, 3]).bmm(theta.transpose(1, 2))
    
    # ignore last dimension of 1s and return
    return res[:, :, :-1]


def make_base_grid(theta, N, C, H, W, align_corners):
    """Generate base grid compatible for computation of the different grids"""
    base_grid = torch.empty([N, H, W, 3])

    base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners))
    base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1))
    base_grid.select(-1, 2).fill_(1)

    return base_grid


def linspace_from_neg_one(grid, num_steps, align_corners):
    """Generates a grid of points of equal separation between -1 and 1"""
    if num_steps <= 1:
        return torch.tensor(0)

    range = torch.linspace(-1, 1, num_steps)
    if not align_corners:
        range = range * (num_steps - 1) / num_steps

    return range