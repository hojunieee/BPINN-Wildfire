"""
Datasets and dataset utilities

[1] Joel Janek Dabrowski, Daniel Edward Pagendam, James Hilton, Conrad Sanderson, 
    Daniel MacKinlay, Carolyn Huston, Andrew Bolt, Petra Kuhnert, "Bayesian 
    Physics Informed Neural Networks for Data Assimilation and Spatio-Temporal 
    Modelling of Wildfires", Spatial Statistics, Volume 55, June 2023, 100746
    https://www.sciencedirect.com/science/article/pii/S2211675323000210
"""

import numpy as np
import torch
import torch.nn as nn

__author__      = "Joel Janek Dabrowski"
__license__     = "MIT license"
__version__     = "0.0.0"

def level_set_function(x, y, x0=0.0, y0=0.0, offset=0.0):
    """
    Generate the level set function in the form of a signed distance function.

    :param x: grid over the x-dimension with shape [Nx]
    :param y: grid over the y-dimension with shape [Ny]
    :param x0: location of the centre of the signed distance function on x
    :param y0: location of the centre of the signed distance function on y
    :param offset: offset of the signed distance function below the zero level
        set plane.
    :return: the level set function with shape [Nx, Ny, 1]
    """
    with torch.no_grad():
        Nx = x.shape[0]
        Ny = y.shape[0]
        X, Y = torch.meshgrid(x, y)
        u = torch.zeros((Nx, Ny, 1))
        # Signed distance function
        u[:, :, 0] = torch.sqrt((X-x0)**2 + (Y-y0)**2) - offset
    return u

def c_wind_obstruction(t, x, y):
    """
    Compute the speed constant c in the level-set equation. This constant 
    comprises both the wind speed and the fire-front speed

    :param t: grid over the time-dimension with shape [Nt]
    :param x: grid over the x-dimension with shape [Nx]
    :param y: grid over the x-dimension with shape [Ny]
    :return: the firefront speed s, and the wind speed in the x, and y 
        directions. These are returned as tensors with shape [Nx, Ny, Nt]
    """
    with torch.no_grad():
        if torch.is_tensor(t):
            X, Y, T = torch.meshgrid(x, y, t)
        else:
            X, Y = torch.meshgrid(x, y)
        
        # Firefront speed
        s = 0.15 * torch.ones_like(X)
        mask = X<0.5
        s[mask] = 0.25
        
        # Wind
        w_x = 1e-6 * torch.ones_like(X)
        w_y = 0.1 * torch.ones_like(X)
        
        if t.shape[0] > 1:
            np.random.seed(0)
            for ti in range(1,t.shape[0]):
                w_x[:,:,ti] = w_x[:,:,ti-1] + 0.001 * np.random.randn(1)
                w_y[:,:,ti] = w_y[:,:,ti-1] + 0.005 * np.random.randn(1)
        
        # Obstructions
        x1_1 = -0.2
        x1_2 = 0.3
        y1_1 = 0.2
        y1_2 = 0.8
        x2_1 = 0.7
        x2_2 = 0.8
        y2_1 = 0.4
        y2_2 = 0.5
        x3_1 = 0.7
        x3_2 = 0.8
        y3_1 = 0.6
        y3_2 = 0.7
        mask = ((X>x1_1) & (X<x1_2) & (Y>y1_1) & (Y<y1_2)) | ((X>x2_1) & (X<x2_2) & (Y>y2_1) & (Y<y2_2)) | ((X>x3_1) & (X<x3_2) & (Y>y3_1) & (Y<y3_2))
        s[mask] = 1e-9
        w_x[mask] = 1e-9
        w_y[mask] = 1e-9
    return s, w_x, w_y

def c_wind_obstruction_complex(t, x, y):
    """
    Complex, smooth, time-varying wind field with evolving vortices.
    Fully compatible with BPINN single or multiple time steps.
    Returns tensors of shape [Nx, Ny, Nt].
    """
    with torch.no_grad():
        # Ensure float tensors
        x = x.float()
        y = y.float()
        t = t.float()
        Nx, Ny, Nt = len(x), len(y), len(t)

        # Meshgrid for spatial domain only (not time)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Allocate outputs
        s = torch.zeros((Nx, Ny, Nt), dtype=torch.float32)
        w_x = torch.zeros((Nx, Ny, Nt), dtype=torch.float32)
        w_y = torch.zeros((Nx, Ny, Nt), dtype=torch.float32)

        # Firefront spread rate (terrain dependent)
        s_base = 0.15 + 0.1 * torch.sin(2 * np.pi * X) * torch.cos(2 * np.pi * Y)
        s_base = torch.clamp(s_base, 0.05, 0.3)

        def smooth_step(t, center, width=0.05):
            return torch.sigmoid((t - center) / width)

        for i in range(Nt):
            t_scalar = float(t[i])
            t_t = torch.tensor(t_scalar, dtype=torch.float32)

            # Smooth transition between temporal regimes
            w1 = smooth_step(t_t, 0.2) * (1 - smooth_step(t_t, 0.4))
            w2 = smooth_step(t_t, 0.4) * (1 - smooth_step(t_t, 0.7))
            w3 = smooth_step(t_t, 0.7)

            scale = 1.0 + 0.5 * w1 - 0.4 * w2 + 0.2 * w3
            phase_shift = 0.3 * w1 - 0.5 * w2 + 1.0 * w3

            # Global oscillatory wind field
            vx_global = scale * (1.0 + 0.25 * torch.sin(20.0 * Y + 40.0 * (t_t + phase_shift)))
            vy_global = scale * 0.3 * torch.cos(20.0 * X - 35.0 * (t_t + phase_shift))

            # Vortices
            cx1, cy1 = 0.55, 0.55
            cx2, cy2 = 0.35, 0.35
            dx1, dy1 = X - cx1, Y - cy1
            dx2, dy2 = X - cx2, Y - cy2
            r1 = torch.sqrt(dx1**2 + dy1**2) + 1e-6
            r2 = torch.sqrt(dx2**2 + dy2**2) + 1e-6

            amp1 = 0.5 * (0.5 + 0.5 * torch.sin(2 * np.pi * (t_t / 0.6)))
            amp2 = 0.3 * (0.5 + 0.5 * torch.cos(2 * np.pi * (t_t / 0.8)))

            strength1 = amp1 * torch.exp(-r1**2 / (2 * 0.12**2))
            strength2 = amp2 * torch.exp(-r2**2 / (2 * 0.15**2))
            core1, core2 = 0.05, 0.06

            vx_spiral1 = -strength1 * dy1 / (r1**2 + core1**2)
            vy_spiral1 =  strength1 * dx1 / (r1**2 + core1**2)
            vx_spiral2 =  strength2 * dy2 / (r2**2 + core2**2)
            vy_spiral2 = -strength2 * dx2 / (r2**2 + core2**2)

            vx = vx_global + 0.3 * (vx_spiral1 + vx_spiral2)
            vy = vy_global + 0.3 * (vy_spiral1 + vy_spiral2)

            # Assign per-time slice
            s[:, :, i] = s_base
            w_x[:, :, i] = vx
            w_y[:, :, i] = vy

        # Terrain obstructions
        X3, Y3, _ = torch.meshgrid(x, y, t, indexing='ij')
        mask = (
            ((X3 > -0.2) & (X3 < 0.3) & (Y3 > 0.2) & (Y3 < 0.8))
            | ((X3 > 0.7) & (X3 < 0.8) & (Y3 > 0.4) & (Y3 < 0.5))
            | ((X3 > 0.7) & (X3 < 0.8) & (Y3 > 0.6) & (Y3 < 0.7))
        )
        s[mask] = 1e-9
        w_x[mask] = 1e-9
        w_y[mask] = 1e-9

    return s, w_x, w_y


class DatasetScaler(nn.Module):
    def __init__(self, x_min, x_max):
        """
        Utility for scaling a dataset to a given minimum and maximum value.

        :param x_min: desired minimum value of the scaled data
        :param x_max: desired maximum value of the scaled data
        """
        super(DatasetScaler, self).__init__()
        self.x_min = x_min
        self.x_max = x_max
    
    def forward(self, x):
        """
        Apply the scaling to an input tensor x

        :param x: input tensor to be scaled
        :return: scaled tensor with range [self.x_min, self.x_max]
        """
        if self.x_min == self.x_max:
            return x
        else:
            with torch.no_grad():
                x = (x - self.x_min) / (self.x_max - self.x_min)
            return x