"""训练候选方向网络（nepodg.experiment.train）：预算 N<B> 的训练快照，存 cases/mixed/models/<kind>/n<B>.pt 与损失曲线 n<B>_loss.csv。
用法（仓库根目录，PINNGPU）：python -m cases.mixed.scripts.train B [kind] [steps]"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from cases.mixed import case
from nepodg import experiment

experiment.train(case, case.Mixed())
