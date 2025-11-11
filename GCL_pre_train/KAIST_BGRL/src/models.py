"""Model definitions for Bootstrap Graph Representation Learning."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    """
    Two-layer perceptron used for projection and prediction heads.
    投影头
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),#是 PyTorch 的一维批量归一化层（Batch Normalization 1D）
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Forward the features through the MLP."""

        return self.net(x)


class GCNEncoder(nn.Module):
    """Simple Graph Convolutional Network encoder."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.activations = nn.ModuleList(nn.PReLU() for _ in range(num_layers))
        self.dropout = nn.Dropout(p=0.5)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for activation in self.activations:
            activation.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_layers = len(self.convs)
        for layer_idx, (conv, activation) in enumerate(zip(self.convs, self.activations)):
            x = activation(conv(x, edge_index))
            if layer_idx < num_layers - 1:
                x = self.dropout(x)
        return x


class BGRL(nn.Module):
    """Implementation of the BGRL architecture."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        projection_dim: int,
        prediction_dim: int,
    ):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = self._copy_encoder(encoder)
        self.target_encoder.eval()

        self.online_projector = MLP(hidden_dim, hidden_dim, projection_dim)
        self.target_projector = MLP(hidden_dim, hidden_dim, projection_dim)
        self.target_projector.eval()
        self.online_predictor = MLP(projection_dim, hidden_dim, prediction_dim)

        for param in self.target_projector.parameters():
            param.requires_grad_(False)

        self._update_target_network(1.0)

    @staticmethod
    def _copy_encoder(encoder: nn.Module) -> nn.Module:
        target = copy.deepcopy(encoder) #这里使用了 Python 的深拷贝函数：
                                        # copy.deepcopy() 会 完整复制 一个模型，包括所有层、权重、超参数；
                                        # 复制后的 target 与原 encoder 互不影响。
        for param in target.parameters():
            param.requires_grad_(False) #不允许这些参数被反向传播更新。
        return target

    @torch.no_grad()
    def _update_target_network(self, momentum: float) -> None:
        """
        这段代码通过 指数滑动平均（Exponential Moving Average, EMA） 的方式，
        将在线网络的参数“缓慢”同步到目标网络。
        参数：
            momentum -> 动量系数

        更新公式：
            θtarget ​← m⋅θtarget​+(1−m)⋅θonline
        """
        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):#encoder参数更新
            target_param.data.mul_(momentum).add_(online_param.data * (1.0 - momentum))
        for target_param, online_param in zip(self.target_projector.parameters(), self.online_projector.parameters()):#投影头参数更新
            target_param.data.mul_(momentum).add_(online_param.data * (1.0 - momentum))
        for target_buffer, online_buffer in zip(self.target_encoder.buffers(), self.online_encoder.buffers()):#“buffers” 是指模型中不是参数（parameter），但仍要保存的状态数据
            target_buffer.copy_(online_buffer)
        for target_buffer, online_buffer in zip(self.target_projector.buffers(), self.online_projector.buffers()):
            target_buffer.copy_(online_buffer)

    def update_momentum(self, momentum: float) -> None:
        """Update target network parameters via exponential moving average."""

        with torch.no_grad():
            self._update_target_network(momentum)

    def forward_online(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.online_encoder(x, edge_index)
        z = self.online_projector(h)
        p = self.online_predictor(z)
        return h, p

    @torch.no_grad()
    def forward_target(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.target_encoder(x, edge_index)
        z = self.target_projector(h)
        return z


def cosine_loss(p: Tensor, z: Tensor) -> Tensor:
    """Compute the negative cosine similarity between predictions and targets."""

    p = nn.functional.normalize(p, dim=-1)
    z = nn.functional.normalize(z.detach(), dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

def augment_graph(data, feature_drop_prob=0.1, edge_drop_prob=0.1, noise_std=0.01):
    """
    对输入的 PyG Data 对象进行增强
    --------------------------------
    feature_drop_prob : float
        随机mask节点特征的比例
    edge_drop_prob : float
        随机删除边的比例
    noise_std : float
        特征加性噪声标准差
    """
    import torch
    import copy
    import numpy as np

    # 深拷贝一份，防止修改原图
    aug_data = copy.deepcopy(data)

    # ---------- (1) 特征增强 ----------
    x = aug_data.x.clone()

    # 随机mask部分节点特征
    mask = torch.rand_like(x) > feature_drop_prob
    x = x * mask

    # 加噪声（模拟测量误差）
    noise = noise_std * torch.randn_like(x)
    x = x + noise

    aug_data.x = x

    # ---------- (2) 结构增强 ----------
    edge_index = aug_data.edge_index.clone()
    num_edges = edge_index.shape[1]

    # 随机删除一部分边
    keep_mask = torch.rand(num_edges) > edge_drop_prob
    edge_index = edge_index[:, keep_mask]
    aug_data.edge_index = edge_index

    return aug_data


__all__ = ["BGRL", "GCNEncoder", "cosine_loss"]
