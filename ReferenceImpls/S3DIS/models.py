import torch
import torch.nn as nn
import EquiNetChannelsPooling

class PermEqui2_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    xm = self.Lambda(xm)
    x = self.Gamma(x)
    x = x - xm
    return x

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

# class PointNetModel(nn.Module):
#     def __init__(self,N, in_dim=3, hidden = 64, k_parts=50):
#         super().__init__()
#
#         self.embed = nn.Sequential(
#             nn.Linear(in_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden), nn.ReLU(),
#         )
#         layer1 = EquiNetChannelsPooling.PermutationClosedLayer(N,N*2, hidden, hidden, "mean", None, False)
#         layer2 =  EquiNetChannelsPooling.PermutationClosedLayer(N*2, N*4, hidden, hidden, "mean", layer1, False)
#         layer3 = EquiNetChannelsPooling.PermutationClosedLayer(N*4, N*2, hidden, hidden, "mean", layer2, False)
#         layer4 = EquiNetChannelsPooling.PermutationClosedLayer(N*2, N, hidden, hidden, "mean", layer3, False)
#         self.equi = nn.Sequential(
#             layer1, nn.ReLU(),
#            layer2, nn.ReLU(),
#             layer3, nn.ReLU(),
#             layer4,
#         )
#
#         self.head = nn.Sequential(
#             nn.Linear(2*hidden, hidden), nn.ReLU(),
#             nn.Linear(hidden, k_parts)
#         )
#
#     def forward(self, x):
#         h = self.embed(x)
#         h = h + self.equi(h)
#         g = h.max(dim=1, keepdim=True).values
#         hg = torch.cat([h, g.expand_as(h)], dim=-1)
#         logits = self.head(hg)
#         return logits.transpose(1, 2)
#
# class DefaultModel(nn.Module):
#     def __init__(self, N, in_dim=3, hidden=64, k_parts=50):
#         super().__init__()
#
#         self.embed = nn.Sequential(
#             nn.Linear(in_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden), nn.ReLU(),
#         )
#
#         self.equi = nn.Sequential(
#             PermEqui2_mean(hidden, hidden), nn.ReLU(),
#             PermEqui2_mean(hidden, hidden), nn.ReLU(),
#             PermEqui2_mean(hidden, hidden), nn.ReLU(),
#             PermEqui2_mean(hidden, hidden),
#         )
#
#         self.head = nn.Sequential(
#             nn.Linear(2 * hidden, hidden), nn.ReLU(),
#             nn.Linear(hidden, k_parts)
#         )
#
#     def forward(self, x):
#         h = self.embed(x)
#         h = h + self.equi(h)
#         g = h.max(dim=1, keepdim=True).values
#         hg = torch.cat([h, g.expand_as(h)], dim=-1)
#         logits = self.head(hg)
#         return logits.transpose(1, 2)
#


class PointNetModel(nn.Module):
    def __init__(self,N, in_dim=3, hidden = 64, k_parts=50):
        super().__init__()

        layer1 = EquiNetChannelsPooling.PermutationClosedLayer(N,N*4, in_dim, hidden, "mean", None, False)
        layer2 =  EquiNetChannelsPooling.PermutationClosedLayer(N*4, N*16, hidden, hidden, "mean", layer1, False)
        layer3 = EquiNetChannelsPooling.PermutationClosedLayer(N*16, N*4, hidden, hidden, "mean", layer2, False)
        layer4 = EquiNetChannelsPooling.PermutationClosedLayer(N*4, N, hidden, k_parts, "mean", layer3, False)
        self.equi = nn.Sequential(
            layer1, nn.ReLU(),
           layer2, nn.ReLU(),
            layer3, nn.ReLU(),
            layer4,
        )


    def forward(self, x):
        logits = self.equi(x)
        return logits.transpose(1, 2)


class DefaultModel(nn.Module):
    def __init__(self, N, in_dim=3, hidden=64, k_parts=50):
        super().__init__()



        self.equi = nn.Sequential(
            PermEqui2_mean(in_dim, hidden), nn.ReLU(),
            PermEqui2_mean(hidden, hidden), nn.ReLU(),
            PermEqui2_mean(hidden, hidden), nn.ReLU(),
            PermEqui2_mean(hidden, k_parts),
        )


    def forward(self, x):
        logits = self.equi(x)
        return logits.transpose(1, 2)


