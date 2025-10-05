import torch
import torch.nn as nn
import EquiNetChannelsPooling

class PermEqui2_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    xm = self.Lambda(xm)
    x = self.Gamma(x)
    x = x - xm
    return x
class PermutationEquivariantModel(torch.nn.Module):
    def __init__(self, set_size:int, channels_in: int, channels_out: int):
        super(PermutationEquivariantModel, self).__init__()
        self.set_size = set_size
        self.channels_in =  channels_in
        self.channels_out = channels_out
        layer1 = EquiNetChannelsPooling.PermutationClosedLayer(set_size,set_size*4,channels_in,channels_out,"max",None, False)
        layer2 = EquiNetChannelsPooling.PermutationClosedLayer(set_size*4,set_size*8,channels_out,channels_out,"max",layer1, False)
        layer3 = EquiNetChannelsPooling.PermutationClosedLayer(set_size*8,set_size*4,channels_out,channels_out,"max",layer2,False)
        layer4 =EquiNetChannelsPooling.PermutationClosedLayer(set_size*4,set_size,channels_out,1,"max",layer3,False)
        self.network = nn.Sequential(
                layer1,
                nn.Tanh(),
                layer2,
                nn.Tanh(),
                layer3,
                nn.Tanh(),
                layer4
            )


    def forward(self, x):
        phi_output = self.network(x)
        samples, set_size, channels = phi_output.shape
        return torch.Tensor.view(phi_output,(samples,set_size))


class DeepSetsModel(torch.nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(DeepSetsModel, self).__init__()
        self.x_dim =  channels_in
        self.d_dim = channels_out
        self.network = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, 1),
        )


    def forward(self, x):
        phi_output = self.network(x)
        samples, set_size, channels = phi_output.shape
        return torch.Tensor.view(phi_output,(samples,set_size))