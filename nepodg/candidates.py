"""参数化网络、训练与候选方向（方法设计 §3）。

网络在算例给出的采样点 X 上输出 m 个原始场 χ_k(x; η)，并由 η 给出读出系数 w(η) = exp(ℓ(η)) r(η)（ℓ 为标量幅值）。
算例的容许映射 admissible 把采样点上的原始场（列）线性映到容许坐标，候选方向 Φ(η) = admissible(χ(X; η))。
网络预测为中心化线性读出 û(η) = ū + f(η)，f = Φ w。损失为快照值与灵敏度的逐项相对误差平方（M 范数，经 whiten）：
  L = mean_i ‖L(f(η_i) − (u_i − ū))‖² / ‖L u_i‖² + λ mean_{i,j} ‖L(∂f/∂η_j(η_i) − s_ij)‖² / ‖L s_ij‖²
因 admissible 线性，f = admissible(χ w)、∂f/∂η = admissible(∂(χ w)/∂η)；一批工况的 ∂(χ w)/∂η 以前向模式自动微分一次求出。
每步随机取 batch 个训练工况，在全部采样点上计算。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacfwd, vmap


def _mlp(dims, last_act):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if last_act or i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Net(nn.Module):
    """主干 (x, η) → m 个原始场（tanh）；读出 η → w(η) = exp(ℓ(η)) r(η)。x、η 已缩放到 [−1, 1] 附近。"""

    def __init__(self, dx=2, deta=2, m=128, layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32):
        super().__init__()
        self.trunk = _mlp([dx + deta] + [width] * (layers - 1) + [m], True)
        self.r = _mlp([deta] + [rwidth] * rlayers + [m], False)
        self.amp = _mlp([deta] + [awidth] * alayers + [1], False)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)
        self.cfg = dict(dx=dx, deta=deta, m=m, layers=layers, width=width, rlayers=rlayers, rwidth=rwidth,
                        alayers=alayers, awidth=awidth)

    def fields(self, X, eta):
        """采样点 X (P, dx) 上的原始场 (P, m)。"""
        return self.trunk(torch.cat([X, eta.expand(len(X), -1)], 1))

    def readout(self, eta):
        return torch.exp(self.amp(eta)) * self.r(eta)


def predict(net, X, E, admissible, grad=False):
    """一批参数坐标 E (b, deta) 上的 f = Φ w（容许坐标，(n, b)，float64）；grad 时另返回 ∂f/∂η (n, b, deta)，否则为 None。"""
    def g(e):
        y = net.fields(X, e) @ net.readout(e)
        return y, y
    if not grad:
        return admissible(vmap(lambda e: g(e)[0])(E).T.double()), None
    J, y = vmap(jacfwd(g, has_aux=True))(E)                                       # (b, P, deta)、(b, P)
    b, d = E.shape
    return admissible(y.T.double()), admissible(J.permute(1, 0, 2).reshape(-1, b * d).double()).reshape(-1, b, d)


def train(net, X, etas, U, mean, admissible, whiten, steps, lr, batch, seed=0, log=1000, S=None, lam=1.0):
    """etas (Ns, deta)、U (n, Ns)：训练工况的参数坐标与增量快照（GPU float64）；mean：ū；
    S (n, Ns, deta)：训练点对 η 的灵敏度，None 时只监督快照值（λ = 0）。学习率在 steps 的 1/2、3/4 处各降为 1/10。
    返回每 log 步的平均损失 (值项, 导数项)。"""
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    E = torch.as_tensor(np.asarray(etas), dtype=torch.float32, device=X.device)
    D = U - mean[:, None]
    su = torch.linalg.norm(whiten(U), dim=0) ** 2
    if S is not None:
        ss = torch.stack([torch.linalg.norm(whiten(S[:, :, j]), dim=0) ** 2 for j in range(S.shape[2])], 1)   # (Ns, deta)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [steps // 2, 3 * steps // 4], 0.1)
    hist, acc = [], np.zeros(2)
    for it in range(1, steps + 1):
        idx = torch.as_tensor(gen.choice(len(E), min(batch, len(E)), replace=False), device=X.device)
        f, df = predict(net, X, E[idx], admissible, S is not None)
        lv = (torch.linalg.norm(whiten(f - D[:, idx]), dim=0) ** 2 / su[idx]).mean()
        loss, ls = lv, torch.zeros((), dtype=lv.dtype, device=X.device)
        if S is not None:
            ls = torch.stack([torch.linalg.norm(whiten(df[:, :, j] - S[:, idx, j]), dim=0) ** 2 / ss[idx, j]
                              for j in range(S.shape[2])]).mean()
            loss = lv + lam * ls
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        acc += [lv.item(), ls.item()]
        if it % log == 0:
            hist.append(tuple(acc / log))
            acc[:] = 0
    return hist


@torch.no_grad()
def directions(net, X, eta, admissible):
    """查询参数坐标 eta 下的候选方向 Φ (n, m)，float64。"""
    e = torch.as_tensor(np.asarray(eta), dtype=torch.float32, device=X.device)
    return admissible(net.fields(X, e).double())


def save(path, net, mean, hist):
    """hist：训练损失曲线（train 的返回值）。"""
    torch.save({"cfg": net.cfg, "state": net.state_dict(), "mean": mean.cpu(), "hist": hist}, path)


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(**ck["cfg"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck["mean"].to(device)
