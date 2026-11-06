"""参数化网络与候选方向（方法设计 §3）。

网络在算例给出的采样点 X 上输出 m 个原始场 χ_k(x; η)，并由 η 给出读出系数 w(η) = exp(ℓ(η)) r(η)（ℓ 为标量幅值）。
算例的容许映射 admissible 把采样点上的原始场（列）线性映到容许坐标，候选方向 Φ(η) = admissible(χ(X; η))。
训练只用主场快照，中心化线性读出 û(η) = ū + Φ(η) w(η)，损失为逐工况相对误差平方的平均（M 范数，经 whiten）：
  L = mean_i ‖L(Φ(η_i) w(η_i) − (u_i − ū))‖² / ‖L u_i‖²
因 admissible 线性，Φ w = admissible(χ w)，训练时只需映射一列。每步随机取 batch 个训练工况，在全部采样点上计算。
"""
import numpy as np
import torch
import torch.nn as nn


def _mlp(dims, last_act):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if last_act or i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Net(nn.Module):
    """主干 (x, η) → m 个原始场（tanh）；读出 η → w(η) = exp(ℓ(η)) r(η)。x、η 已缩放到 [−1, 1] 附近。"""

    def __init__(self, dx=2, deta=2, m=128, layers=5, width=128, rlayers=2, rwidth=64):
        super().__init__()
        self.trunk = _mlp([dx + deta] + [width] * (layers - 1) + [m], True)
        self.r = _mlp([deta] + [rwidth] * rlayers + [m], False)
        self.amp = _mlp([deta] + [rwidth] * rlayers + [1], False)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)
        self.cfg = dict(dx=dx, deta=deta, m=m, layers=layers, width=width, rlayers=rlayers, rwidth=rwidth)

    def fields(self, X, eta):
        """采样点 X (P, dx) 上的原始场 (P, m)。"""
        return self.trunk(torch.cat([X, eta.expand(len(X), -1)], 1))

    def readout(self, eta):
        return torch.exp(self.amp(eta)) * self.r(eta)


def train(net, X, etas, U, mean, admissible, whiten, steps, lr, batch, seed=0, log=1000):
    """etas (Ns, deta)、U (n, Ns)：训练工况的参数坐标与增量快照（GPU float64）；mean：ū。
    学习率在 steps 的 1/2、3/4 处各降为 1/10。返回每 log 步的平均损失。"""
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    E = torch.as_tensor(np.asarray(etas), dtype=torch.float32, device=X.device)
    D = U - mean[:, None]
    scale = torch.linalg.norm(whiten(U), dim=0) ** 2
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [steps // 2, 3 * steps // 4], 0.1)
    hist, acc = [], 0.0
    for it in range(1, steps + 1):
        loss = 0.0
        for i in gen.choice(len(E), min(batch, len(E)), replace=False):
            f = (net.fields(X, E[i]) @ net.readout(E[i])).double()
            loss = loss + torch.linalg.norm(whiten(admissible(f) - D[:, i])) ** 2 / scale[i]
        loss = loss / min(batch, len(E))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        acc += float(loss)
        if it % log == 0:
            hist.append(acc / log)
            acc = 0.0
    return hist


@torch.no_grad()
def directions(net, X, eta, admissible):
    """查询参数坐标 eta 下的候选方向 Φ (n, m)，float64。"""
    e = torch.as_tensor(np.asarray(eta), dtype=torch.float32, device=X.device)
    return admissible(net.fields(X, e).double())


def save(path, net, mean):
    torch.save({"cfg": net.cfg, "state": net.state_dict(), "mean": mean.cpu()}, path)


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(**ck["cfg"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck["mean"].to(device)
