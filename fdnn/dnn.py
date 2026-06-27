"""dnn.py — the Static Delay-Summation Network (SDSNN) frame-level dash detector.

Copied verbatim from C:/Adhitya/Coding/test/FDNN/dash_code/dnn.py (inference
parts). Each FixedDelayLayer reads from a few fixed points in the recent past
via grouped time shifts; stacking them with base-2 dilation spans the ~27-frame
dash event. The readout emits one per-frame logit; sigmoid + NMS turns the
probability track into a dash count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Fixed Delay Layer --------------------------------------------------------
class FixedDelayLayer(nn.Module):
    def __init__(self, c_in, c_out, dilation=1):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)
        self.dilation = dilation

    def forward(self, x):
        # x: [B, T, c_in]
        h = self.linear(x)
        a = F.relu(h)

        # Apply delays
        B, T, C = a.shape
        groups = 4
        group_size = C // groups

        max_delay = 3 * self.dilation
        # pad only time dimension: [B, T+max_delay, C]
        a_padded = F.pad(a, (0, 0, max_delay, 0))

        out = torch.empty_like(a)
        for i in range(4):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < 3 else C
            delay = i * self.dilation

            # For delay d, we want time t to take value from time t-d.
            # In padded array, t-d is at index (t - d + max_delay)
            # So the sequence starts at (max_delay - delay) and ends at (max_delay - delay + T)
            start_t = max_delay - delay
            out[:, :, start_idx:end_idx] = a_padded[:, start_t : start_t + T, start_idx:end_idx]

        return out


# --- frame-level detector head --------------------------------------------
class DNN(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=4, **kwargs):
        super().__init__()
        # Ignore legacy kwargs (max_delay, residual, norm) for load compatibility.
        self.layers = nn.ModuleList()
        c_in = in_dim
        for i in range(layers):
            # Base-2 dilation: 1, 2, 4, 8...
            dilation = 2 ** i
            self.layers.append(FixedDelayLayer(c_in, hidden, dilation=dilation))
            c_in = hidden
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x):                          # x: [B, T, in_dim]
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.readout(h).squeeze(-1)         # logits [B, T]
