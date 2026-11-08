"""训练候选方向网络：快照预算 N<B> 的训练快照与灵敏度（方法设计 §3、实例化设计 §8）。kind：
  d（默认）  快照值与灵敏度共同监督（λ = 1），存 cases/graetz/models/n<B>.pt
  v          只用快照值监督（λ = 0），存 cases/graetz/models/n<B>_v.pt
用法（仓库根目录，PINNGPU）：python -m cases.graetz.scripts.train B [kind] [steps]"""
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import torch

from cases.graetz.case import ROOT, Graetz, cases, eta
from nepodg import candidates

STEPS, LR, BATCH, SEED, LAM = 100000, 1e-3, 8, 1234, 1.0
NET = dict(m=128, layers=5, width=128, rlayers=2, rwidth=64, alayers=2, awidth=32)

budget = int(sys.argv[1])
kind = sys.argv[2] if len(sys.argv) > 2 else "d"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
g = Graetz()
ids = [c["id"] for c in cases() if c[f"n{budget}"] == "1"]
snaps = [g.load(i) for i in ids]
U = torch.stack([s[0] for s in snaps], 1)
mean = U.mean(1)
torch.manual_seed(SEED)
net = candidates.Net(**NET).to(g.X.device)
t0 = time.perf_counter()
hist = candidates.train(net, g.X, [eta(s[1]) for s in snaps], U, mean, g.admissible, g.whiten, steps, LR, BATCH, SEED,
                        log=max(1, steps // 100), S=g.sens(ids) if kind == "d" else None, lam=LAM)
print(f"n{budget} {kind}：{len(ids)} 个训练快照，{steps} 步，{time.perf_counter() - t0:.0f} s，"
      f"末段损失（值、导数）{hist[-1][0]:.3e} {hist[-1][1]:.3e}")
(ROOT / "models").mkdir(exist_ok=True)
candidates.save(ROOT / "models" / f"n{budget}{'' if kind == 'd' else '_v'}.pt", net, mean, hist)
