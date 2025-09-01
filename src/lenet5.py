import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- C3 connection table (columns 0..15). Rows are S2 maps {0..5}. ----
# Source: LeCun et al., 1998, Proc. IEEE, Table I
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

class ScaledTanh(nn.Module):
    """ A * tanh(S * x)  with A=1.7159, S=2/3 per LeCun-5. """
    def __init__(self, A=1.7159, S=2.0/3.0):
        super().__init__()
        self.A, self.S = A, S

    def forward(self, x):
        return self.A * torch.tanh(self.S * x)

class Cx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=True)
        self.act  = ScaledTanh()
        
    def forward(self, x):
        return self.act(self.conv(x))

class Sx(nn.Module):
    """
    LeNet subsampling: average pool 2x2 (stride 2), then per-channel learnable
    scale 'a' and bias 'b', then scaled tanh.
    """
    def __init__(self, channels, kernel_size=2, stride=2):
        super().__init__()
        self.a = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, divisor_override=1)
        self.act  = ScaledTanh()

    def forward(self, x):
        s = self.pool(x)
        y = s * self.a.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)
        return self.act(y)

class PartialConv2d(nn.Module):
    """
    Conv layer with sparse S2->C3 connectivity.
    connection_table: list of len(out_channels); each entry lists input maps used.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True, connection_table=None):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.kernel_size = k
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias   = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # binary mask (out,in,1,1)
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

class RBFClassifier(nn.Module):
    """
    Euclidean RBF output layer (one prototype per class).
    Returns negative squared distances so CrossEntropyLoss can be used.
    Prototypes are initialized to +-1 with equal probability.
    The paper calls for initializing the logits to represent the bitmap
    of a 7 x 12 number. This is done for a variety of reasons in the actual paper.
    But here we just choose another viable option.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        # Initialize prototypes to +-1 with equal probability
        self.prototypes = nn.Parameter((torch.randint(0, 2, (num_classes, in_features)) * 2 - 1).float())

    def forward(self, x):  # x: (N, in_features)
        # logits = -(||x||^2 - 2 x*w^T + ||w||^2)
        x2 = (x * x).sum(dim=1, keepdim=True)         # (N,1)
        w2 = (self.prototypes * self.prototypes).sum(dim=1).unsqueeze(0)  # (1,C)
        xw = x @ self.prototypes.t()                  # (N,C)
        return -(x2 - 2*xw + w2)                      # (N,C)

class LeNet5(nn.Module):
    def __init__(self, use_rbf=True, num_classes=10):
        super().__init__()
        self.c1  = Cx(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.s2  = Sx(channels=6, kernel_size=2, stride=2)
        self.c3  = nn.Sequential(
            PartialConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, connection_table=C3_TABLE),
            ScaledTanh(),
        )
        self.s4  = Sx(channels=16, kernel_size=2, stride=2)
        self.c5  = Cx(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.act = ScaledTanh()
        self.out = RBFClassifier(84, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.act(self.f6(x))
        x = self.out(x)
        return x

def lenet5_init_(m):
    """
    Initialize weights ~ U(-2.4/F_i, +2.4/F_i), biases = 0.
    F_i is fan-in (#inputs to a unit).
    """
    if isinstance(m, PartialConv2d):
        k = m.kernel_size
        with torch.no_grad():
            m.weight.zero_()
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
