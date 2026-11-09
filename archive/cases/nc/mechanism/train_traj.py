"""训练过程中的候选空间（cases/nc/实验记录.md「训练过程中的候选空间」）：读 models/<kind>/n<B>_ckpt/<步数>.pt 与 loss.csv，
在外推 27 点与内插 50 点上计可分辨方向数与余量捕获率 Γ，所列步数另做闭合。结果见 results/traj_n<B>_<kind>.csv。
用法（仓库根目录，PINNGPU）：PYTHONPATH=. python archive/cases/nc/mechanism/train_traj.py B kind out.csv [闭合步数,...]"""
import csv, sys
import numpy as np, torch
from cases.nc.case import MODELS, NC, cases, eta
from nepodg import candidates, closure, metrics, space

B, k, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
cl = {int(x) for x in sys.argv[4].split(",")} if len(sys.argv) > 4 else set()
nc = NC()
ids = [c["id"] for c in cases() if c[f"n{B}"] == "1"]
train = [nc.load(i) for i in ids]
pod = space.POD(torch.stack([s[0] for s in train], 1), nc.whiten)
init = closure.Init([eta(s[2]) for s in train], pod.a)
base = space.sensitivity_space(pod, nc.sens(ids).reshape(nc.n, -1), nc.whiten)[0]
Q = [c for c in cases() if c["role"] == "extrap"] + [c for c in cases() if c["role"] == "interp"][::2]
ref = {}
for c in Q:
    psi, _, mu = nc.load(c["id"])
    d = nc.whiten(psi - pod.mean)
    ref[c["id"]] = (psi, mu, base.project(d[:, None])[:, 0])
d = MODELS / k / f"n{B}_ckpt"
steps = sorted(int(p.stem) for p in d.glob("*.pt"))
loss = {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(d / "loss.csv"))}
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["step", "loss", "id", "Ra", "Pr", "n2", "n4", "n6", "n_s01", "Gamma", "e", "e_proj"])
    for st in steps:
        net = candidates.load(d / f"{st}.pt", nc.X.device)
        for c in Q:
            psi, mu, rho = ref[c["id"]]
            Phi = candidates.directions(net, nc.X, eta(mu), nc.admissible)
            s = torch.linalg.svdvals(nc.whiten(Phi).cpu()); s = s / s[0]
            Z, Zw, sz = space.complement(Phi, nc.whiten, base)
            G = float(torch.linalg.norm(Zw.T @ rho) ** 2 / torch.linalg.norm(rho) ** 2)
            e = ep = ""
            if st in cl:
                Bm = torch.cat([base.V, Z], 1)
                c0 = np.concatenate([init(eta(mu)), np.zeros(Bm.shape[1] - pod.V.shape[1])])
                y = closure.solve(nc.reduced, pod.mean, Bm, mu, c0)[0]
                e, ep, _ = metrics.errors(y, psi, pod.mean, metrics.basis(Bm, nc.whiten), nc.whiten, nc.lift)
            wr.writerow([st, loss[min(loss, key=lambda x: abs(x - st))] if st else "", c["id"], c["Ra"], c["Pr"], *[int((s > t).sum()) for t in (1e-2, 1e-4, 1e-6)],
                         int((sz[:Z.shape[1]] > 0.1).sum()), G, e, ep])
        fh.flush()
        print(st, flush=True)
