import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- C3 connection table (columns 0..15). Rows are S2 maps {0..5}. ----
# Source: LeCun et al. 1998, Table I

C3_TABLE = [
    [0,1,2],        # 0
    [1,2,3],        # 1
    [2,3,4],        # 2
    [3,4,5],        # 3
    [0,4,5],        # 4
    [0,1,5],        # 5
    [0,1,2,3],      # 6
    [1,2,3,4],      # 7
    [2,3,4,5],      # 8
    [0,3,4,5],      # 9
    [0,1,4,5],      # 10
    [0,1,2,5],      # 11
    [0,1,3,4],      # 12
    [1,2,4,5],      # 13
    [0,2,3,5],      # 14
    [0,1,2,3,4,5],  # 15 (fully connected to S2)
]

# normalize input pixels. background is -0.1 and the foreground is 1.175

class ScaledTanh(nn.Module):
    """
    Scaled Hyperbolic Tangent activation function as used in the original LeNet5 paper.

    The original LeNet5 paper used f(x) = A * tanh(S * x) instead of the standard tanh(x).
    Values from the paper: A = 1.7159, S = 2/3 ≈ 0.6667
    """

    def __init__(self, A=1.7159, S=0.6667):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, x):
        return self.A * torch.tanh(self.S * x)
    
class Cx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=True),
            ScaledTanh(),
        )
    
    def forward(self, x):
        return self.layer(x)

class Sx(nn.Module):
    """
    Custom pooling layer. Sum the input units. Then scale by learnable coefficient
    and add a learnable bias. End with a scaled hyperbolic tangent activation.
    """

    def __init__(self, channels, kernel_size=2, stride=2):
        super().__init__()
        self.a = nn.Parameter(torch.ones(channels))  # per-channel scale
        self.b = nn.Parameter(torch.zeros(channels))  # per-channel bias
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, divisor_override=1)
        self.activation = ScaledTanh()
    
    def forward(self, x):
        s = self.pool(x)
        y = s * self.a.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)
        return self.activation(y)
    
class PartialConv2d(nn.Module):
    """
    A conv layer where each output channel connects only to a subset of input channels.
    The connectivity is given by `connection_table`: list of length out_channels; entry i
    is a list of input-channel indices allowed for output channel i.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True, connection_table=None):
        super().__init__()
        k = kernel_size
        self.stride = stride
        self.padding = 0
        self.kernel_size = k

        # learnables
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        mask = torch.zeros(out_channels, in_channels, 1, 1)
        if connection_table is None:
            mask[:] = 1.0
        else:
            for o, ins in enumerate(connection_table):
                mask[o, ins, 0, 0] = 1.0
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(x, w, bias=self.bias, stride=self.stride)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self.pad32,
            Cx(in_channels=1, out_channels=6, kernel_size=5, stride=1),  # C1
            Sx(channels=6, kernel_size=2, stride=2),  # S2
            nn.Sequential(  # C3
                PartialConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, connection_table=C3_TABLE),
                ScaledTanh(),
            ),
            Sx(channels=16, kernel_size=2, stride=2),  # S4
            Cx(in_channels=16, out_channels=120, kernel_size=5, stride=1),  # C5
            nn.Flatten(),
            nn.Linear(120, 84),
            ScaledTanh(),
            nn.Linear(84, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)
    
def lenet5_init_(m):
    """
    Initialize weights ~ U(-2.4/F_i, +2.4/F_i), biases = 0.
    F_i is fan-in (#inputs to a unit).
      * Conv2d: F_i = k*k*in_channels
      * PartialConv2d: F_i per output map = k*k*#connected_in_channels
      * Linear: F_i = in_features
    """
    # PartialConv2d: per-output fan-in based on the mask
    if isinstance(m, PartialConv2d):
        k = m.kernel_size
        with torch.no_grad():
            m.weight.zero_()
            # mask2d: (out, in)
            mask2d = m.mask[..., 0, 0]
            for o in range(mask2d.size(0)):
                n_in = int(mask2d[o].sum().item())
                if n_in == 0:
                    continue
                bound = 2.4 / (k * k * n_in)
                ins = mask2d[o].nonzero(as_tuple=False).flatten().tolist()
                for c in ins:
                    nn.init.uniform_(m.weight[o, c], -bound, +bound)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        k_h, k_w = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size))
        fan_in = m.in_channels * k_h * k_w
        bound = 2.4 / fan_in
        nn.init.uniform_(m.weight, -bound, +bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        fan_in = m.in_features
        bound = 2.4 / fan_in
        nn.init.uniform_(m.weight, -bound, +bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
