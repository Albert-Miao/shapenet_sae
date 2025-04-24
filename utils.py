import torch
import torch.nn as nn
import numpy as np


def apply_random_rotation(pc, rot_axis=1):

    theta = np.random.rand(1) * 2 * np.pi
    cos = np.cos(theta)
    sin = np.sin(theta)

    if rot_axis == 0:
        rot = np.array([
            cos, -sin, 0.0,
            sin, cos, 0.0,
            0.0, 0.0, 1.0
        ]).T.reshape(3, 3)
    elif rot_axis == 1:
        rot = np.array([
            cos, 0, -sin,
            0, 1, 0,
            sin, 0, cos
        ]).T.reshape(3, 3)
    elif rot_axis == 2:
        rot = np.array([
            1.0, 0.0, 0.0,
            0.0, cos, -sin,
            0.0, sin, cos
        ]).T.reshape(3, 3)
    else:
        raise Exception("Invalid rotation axis")
    # rot = torch.from_numpy(rot).to(pc.device)

    # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
    # pc_rotated = torch.matmul(pc, rot)
    pc_rotated = pc @ rot
    return pc_rotated, rot, theta


def hard_threshold(arr, thresh=0.0):
      arr[arr <= thresh] = 0.0
      #arr[arr <= np.random.uniform(thresh*0.6, thresh)] = 0.0
      return arr


class JumpReLU(nn.Module):
    r"""Applies the jump rectified linear unit function element-wise
    
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, jump=0.0, episilon=1e-10):
        super(JumpReLU, self).__init__()
        self.jump = jump

    def forward(self, input):
        return hard_threshold(input, thresh=(torch.e ** self.jump))

    def __repr__(self):
        return self.__class__.__name__ + '()'