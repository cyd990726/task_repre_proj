import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self,  surrogate_decomposer=None, main_args=None):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch, task=None):
        return th.sum(agent_qs, dim=2, keepdim=True)