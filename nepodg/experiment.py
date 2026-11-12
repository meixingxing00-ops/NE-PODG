"""算例实验的通用入口（方法设计 §3、§5）：训练候选方向网络；在测试工况上降阶求解并记录误差。

算例模块（cases.<算例>.case）提供 cases()（cases.csv 的行，含 id、role 与预算标记 n<B>）、eta(μ)、MODELS、ROOT；
算例对象提供 n、X、whiten、admissible、lift、load(编号)（首项为主场快照，末项为参数 μ）与 sens(编号列表)。

降阶空间 space：
  pod                         V：快照空间
  sapodg                      W = V⊕S：加预算内训练点的稳态灵敏度（SA-PODG）
  <基>_<网络>[_fix<编号>][_z<r>]
                              在基 V 或 VS（= W）上加网络方向（去基的分量）。网络名见 candidates.kind：
                                v、v<种子>、v…m<m>  训练的网络 MODELS/<网络名>/n<B>.pt
                                r<种子>、r…m<m>     同结构、以该种子随机初始化的未训练网络
                              fix<编号>：方向取自 cases.csv 中该工况的参数，所有查询相同（默认取查询参数）
                              z<r>：候选空间只取 Φ(η) 的前 r 个奇异方向（M 范数）再去基的分量，n_Z ≤ r
  主方法为 VS_v（默认）。例：VS_r1、V_v、VS_v_fixP018、VS_v_z32、VS_vm64。
结果 ROOT/results/n<B>/<space>_<role>.csv，每个工况一行：编号、参数、n_V、n_S、n_Z、主场误差 e、所用空间与 POD 空间的最佳逼近误差
（按闭合所用的状态基，metrics.basis）、e_real、算例输出量的误差、收敛与迭代数（非线性闭合）、单次查询耗时 time 及其中网络方向与
去投影的耗时 time_basis；非线性闭合另记末次相对残差 res = ‖g‖/‖g(c⁽⁰⁾)‖；有网络方向时另记保留方向主角正弦 s 的最小值、中位数与 s > 0.1、s > 0.01 的个数。"""
import csv
import sys
import time

import numpy as np
import torch

from nepodg import candidates, closure, metrics, space

STEPS, LR, BATCH = 30000, 1e-3, 8
NET = dict(layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)


def train(case, g):
    """命令行 B [kind] [steps]：用预算 N<B> 的训练快照训练网络 kind（默认 v），存 MODELS/<kind>/n<B>.pt；
    损失曲线（每 steps/100 步的平均损失）另存为同目录的 n<B>_loss.csv。"""
    budget = int(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "v"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    _, seed, m = candidates.kind(name)
    ids = [c["id"] for c in case.cases() if c[f"n{budget}"] == "1"]
    snaps = [g.load(i) for i in ids]
    U = torch.stack([s[0] for s in snaps], 1)
    torch.manual_seed(seed)
    net = candidates.Net(m=m, **NET).to(g.X.device)
    t0 = time.perf_counter()
    log = max(1, steps // 100)
    hist = candidates.train(net, g.X, [case.eta(s[-1]) for s in snaps], U, g.admissible, g.whiten, steps, LR, BATCH, seed,
                            log=log)
    print(f"n{budget} {name}：{len(ids)} 个训练快照，m = {m}，{steps} 步，{time.perf_counter() - t0:.0f} s，末段损失 {hist[-1]:.3e}")
    (case.MODELS / name).mkdir(parents=True, exist_ok=True)
    candidates.save(case.MODELS / name / f"n{budget}.pt", net, hist)
    with open(case.MODELS / name / f"n{budget}_loss.csv", "w") as fh:
        fh.write("step,loss\n")
        fh.writelines(f"{(k + 1) * log},{a:.6e}\n" for k, a in enumerate(hist))


class Setup:
    """预算 N<B>、空间 mode 下的 POD 空间 pod、初值插值 init、基础空间 base（V 或 W）、n_S 与网络方向的设置。"""

    def __init__(self, case, g, budget, mode, params):
        self.case, self.g = case, g
        ids = [c["id"] for c in case.cases() if c[f"n{budget}"] == "1"]
        train = [g.load(i) for i in ids]
        self.pod = space.POD(torch.stack([s[0] for s in train], 1), g.whiten)
        self.init = closure.Init([case.eta(s[-1]) for s in train], self.pod.a)
        self.base, self.ns = self.pod, 0
        if mode == "sapodg" or mode.startswith("VS"):
            self.base, self.ns, _ = space.sensitivity_space(self.pod, g.sens(ids).reshape(g.n, -1), g.whiten)
        self.net, self.fix, self.zmax = None, None, None
        if "_" in mode:
            parts = mode.split("_")
            trained, seed, m = candidates.kind(parts[1])
            if trained:
                self.net = candidates.load(case.MODELS / parts[1] / f"n{budget}.pt", g.X.device)
            else:
                torch.manual_seed(seed)
                self.net = candidates.Net(m=m).to(g.X.device).eval()
            for p in parts[2:]:
                if p.startswith("fix"):
                    self.fix = next(case.eta(tuple(float(c[k]) for k in params)) for c in case.cases() if c["id"] == p[3:])
                else:
                    self.zmax = int(p[1:])

    def basis(self, mu):
        """查询参数 μ 的状态基 B 与保留方向主角正弦的统计。"""
        B, sst = self.base.V, [""] * 4
        if self.net is not None:
            g = self.g
            eta = self.case.eta(mu) if self.fix is None else self.fix
            Z, _, s = space.complement(candidates.directions(self.net, g.X, eta, g.admissible), g.whiten, self.base, r=self.zmax)
            B = torch.cat([B, Z], 1)
            s = s[:Z.shape[1]].cpu().numpy()
            sst = [s.min(), np.median(s), int((s > 0.1).sum()), int((s > 0.01).sum())]
        return B, sst

    def c0(self, mu, B):
        return np.concatenate([self.init(self.case.eta(mu)), np.zeros(B.shape[1] - self.pod.V.shape[1])])


def solve(case, g, budget, cid, mode, rom):
    """单个工况 cid 在预算 N<B>、空间 mode（pod、sapodg 或 <基>_<网络>…）下的降阶解：(主场, 辅助场, 闭合状态或 None)。"""
    st = Setup(case, g, budget, mode, ())
    mu = g.load(cid)[-1]
    B, _ = st.basis(mu)
    return rom(st.pod.mean, B, mu, st.c0(mu, B))


def run(case, g, params, rom, outputs):
    """命令行 B role [space]：预算 N<B> 在 cases.csv 中 role 为该值的工况上降阶求解。params：参数列名；
    rom(y0, B, μ, c0) → (主场, 辅助场, 闭合状态或 None)，c0 为初值系数；outputs(主场, 辅助场, 参考快照, μ) → {列名: 值}。"""
    budget, role = int(sys.argv[1]), sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "VS_v"
    S = Setup(case, g, budget, mode, params)
    pod, base = S.pod, S.base
    Vw_pod = metrics.basis(pod.V, g.whiten)
    out = case.ROOT / "results" / f"n{budget}" / f"{mode}_{role}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = None
        for c in [c for c in case.cases() if c["role"] == role]:
            ref = g.load(c["id"])
            mu = ref[-1]
            t0 = time.perf_counter()
            B, sst = S.basis(mu)
            torch.cuda.synchronize()
            tb = time.perf_counter() - t0
            y, aux, st = rom(pod.mean, B, mu, S.c0(mu, B))
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
            e, ep, er = metrics.errors(y, ref[0], pod.mean, metrics.basis(B, g.whiten), g.whiten, g.lift)
            ep_pod = metrics.errors(y, ref[0], pod.mean, Vw_pod, g.whiten, g.lift)[1]
            row = dict(id=c["id"], **{k: c[k] for k in params}, nV=pod.V.shape[1], nS=S.ns, nZ=B.shape[1] - base.V.shape[1],
                       e=e, e_proj=ep, e_real=er, e_proj_pod=ep_pod, **outputs(y, aux, ref, mu))
            if st is not None:
                row.update(converged=st["converged"], iters=st["iters"], res=st["res"])
            row.update(time=t, time_basis=tb, s_min=sst[0], s_med=sst[1], n_s01=sst[2], n_s001=sst[3])
            if wr is None:
                wr = csv.DictWriter(fh, list(row))
                wr.writeheader()
            wr.writerow(row)
            fh.flush()
    print(f"→ {out}")
