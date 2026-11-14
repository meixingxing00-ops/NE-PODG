# NE-PODG 核心方法与科学论证基准

## 神经互补空间增广与 Galerkin 物理投影求解

NE-PODG（Neural-Enriched Proper Orthogonal Decomposition–Galerkin）是一种面向参数化稳态偏微分方程（PDE）的神经互补空间增广与 Galerkin 投影降阶方法。其基本构造是：以高保真快照及其一阶参数灵敏度形成基础降阶空间；利用场值监督训练的神经网络生成参数相关的多方向候选空间；提取该候选空间相对于基础空间的非冗余方向；最后由原离散控制方程在增广空间中的 Galerkin 投影确定降阶解。

该方法对神经网络的使用并不止于直接预测一个解场。网络提供可供物理求解利用的空间表示自由度，而控制方程负责在给定试探空间内确定状态：

$$
\boxed{
\text{神经网络提供互补表示自由度}
\quad+\quad
\text{Galerkin 投影确定空间内的解}
}
$$

NE-PODG 的科学问题是：**在已有快照—灵敏度降阶空间的基础上，如何利用有限场值数据学习能够补充其未表示解结构的方向，以及这些方向的最佳逼近收益如何影响实际物理投影求解精度？**

对这一问题，本文区分三个层次：神经候选空间的数学来源；互补空间对基础空间剩余解分量的捕获；Galerkin 闭合对最终误差的影响。其中，空间包含关系与误差分解给出严格的数学性质，而训练是否产生有效方向及实际误差是否下降，由数值证据判定。

---

## 一、科学问题：参数化降阶求解中的空间表示限制

### 1.1 参数化稳态问题与容许状态

考虑参数向量 $\mu\in\mathcal P_\mu\subset\mathbb R^d$，并记用于网络输入和参数分析的坐标为 $\eta=\eta(\mu)\in\mathcal P$。下文以 $\eta$ 统一表示参数依赖。

经过空间离散、非齐次边界条件处理及必要的约束消元后，物理状态表示为

$$
\boxed{
\mathbf y(\eta)=\boldsymbol\ell_h(\eta)+\mathbf u(\eta),
\qquad \mathbf u(\eta)\in\mathbb R^n.
}
$$

其中，$\boldsymbol\ell_h(\eta)$ 为边界提升，$\mathbf u$ 是共同容许坐标下的增量状态。被消去的离散约束已经包含在这一坐标选择中；涉及不同物理几何的算例，须通过一致的参考表示建立共同状态坐标。本文讨论的快照、灵敏度、神经候选方向和误差投影均针对 $\mathbf u$；物理场由 $\mathbf y=\boldsymbol\ell_h+\mathbf u$ 恢复。

记目标高保真离散解为 $\mathbf u^\star(\eta)$。参数变化使其形成解流形

$$
\mathcal M=\{\mathbf u^\star(\eta):\eta\in\mathcal P\}\subset\mathbb R^n.
$$

采用同一对称正定度量 $M\in\mathbb R^{n\times n}$ 定义内积与范数：

$$
\langle\mathbf a,\mathbf b\rangle_M=\mathbf a^{\mathsf T}M\mathbf b,
\qquad
\|\mathbf a\|_M^2=\mathbf a^{\mathsf T}M\mathbf a.
$$

$M$ 用于 POD、空间正交化、最优逼近与主状态误差分析，其具体选择由离散问题确定。

### 1.2 快照—灵敏度基础空间

给定 $N_s$ 个训练参数及其容许状态快照 $\mathbf u_i=\mathbf u^\star(\eta_i)$，定义快照均值与中心化矩阵：

$$
\bar{\mathbf u}=\frac1{N_s}\sum_{i=1}^{N_s}\mathbf u_i,
\qquad
D=[\mathbf u_1-\bar{\mathbf u},\ldots,\mathbf u_{N_s}-\bar{\mathbf u}].
$$

从 $D$ 的 $M$-加权 POD 分解中构造空间 $V$。当前方法保留数值可辨识的快照方向，而非以较低的累计能量比例进行额外的物理截断。进一步，在各训练点由相容的离散稳态方程求得一阶参数灵敏度

$$
\mathbf s_{ij}=\partial_{\eta_j}\mathbf u^\star(\eta_i),
\qquad i=1,\ldots,N_s,\quad j=1,\ldots,d.
$$

将灵敏度集合去除 $V$ 的投影并进行数值秩筛选，所得正交空间记为 $S$。基础空间为

$$
\boxed{
W=V\oplus_M S,
\qquad W^{\mathsf T}MW=I,
\qquad P_W=WW^{\mathsf T}M.
}
$$

这里 $V$ 提供训练解的空间结构，$S$ 提供训练点附近的一阶参数响应方向。仅在这一空间上实施 Galerkin 投影，构成灵敏度增广基线 SA-PODG。与之对应的仿射容许状态空间为

$$
\mathcal T_W^{(u)}=\bar{\mathbf u}+\operatorname{span}W.
$$

对于给定参数，基础空间未表示的解余量定义为

$$
\boxed{
\boldsymbol\rho_W(\eta)
=(I-P_W)\big[\mathbf u^\star(\eta)-\bar{\mathbf u}\big].
}
$$

该余量的 $M$-范数恰为基础仿射空间的最佳逼近误差（未归一化）。因此，固定空间中的系数预测或方程求解即使达到该空间内的最优状态，也无法表示 $\boldsymbol\rho_W$。

### 1.3 局部参数响应与剩余表示误差

若目标解分支在训练参数 $\eta_i$ 附近足够光滑，且基础空间完整包含该点的中心化快照与各一阶参数导数，则

$$
\begin{aligned}
\boldsymbol\rho_W(\eta_i+\Delta\eta)
={}&\frac12(I-P_W)
D_\eta^2\mathbf u^\star(\eta_i)
[\Delta\eta,\Delta\eta]\\
&+O(\|\Delta\eta\|^3).
\end{aligned}
$$

这说明快照—灵敏度基础空间能够消除训练点处的零阶及一阶投影误差，但并不自动包含全部高阶参数响应。该关系是局部性质，其二阶主项也可能继续落在 $W$ 中；不能据此将所有高阶变化都视为必然缺失的方向。

由此，NE-PODG 面向的具体表示问题为：**不显式计算高阶参数导数，也不在在线阶段使用未知参考解余量，能否从有限场值监督中获得对 $\boldsymbol\rho_W(\eta)$ 有效的附加空间方向？**

---

## 二、核心表示：场值监督与神经输出可达空间

### 2.1 多方向候选场与参数化读出

神经网络在查询参数 $\eta$ 下产生 $m$ 个候选场。原始网络输出经过算例给定的容许映射 $\mathcal A_h$ 后，形成容许状态坐标下的矩阵

$$
\Phi_\theta(\eta)
=[\boldsymbol\phi_{\theta,1}(\eta),\ldots,
  \boldsymbol\phi_{\theta,m}(\eta)]\in\mathbb R^{n\times m}.
$$

网络同时产生参数化读出系数 $w_\theta(\eta)\in\mathbb R^m$，组合预测为

$$
\boxed{
\widehat{\mathbf u}_\theta(\eta)
=\Phi_\theta(\eta)w_\theta(\eta).
}
$$

候选场依赖空间坐标和参数，读出系数依赖参数；两者对参数均可具有非线性依赖。当前网络采用标量幅值分支与仿射读出结构：

$$
w_\theta(\eta)=e^{\ell_\theta(\eta)}r_\theta(\eta),
\qquad
r_\theta(\eta)=K_\theta h_\theta(\eta)+\mathbf b.
$$

其中 $\ell_\theta$ 为标量幅值分支，$\mathbf b$ 为最后一层读出偏置；候选场分支和幅值分支均不依赖这一偏置。

对训练快照采用相对场值监督：

$$
\boxed{
\mathcal L(\theta)=\frac1{N_s}\sum_{i=1}^{N_s}
\frac{\|\Phi_\theta(\eta_i)w_\theta(\eta_i)-\mathbf u_i\|_M^2}
{\|\mathbf u_i\|_M^2}.
}
$$

上述写法要求训练快照的分母非零；具体算例按同一相对误差口径处理。训练目标只约束候选场的组合输出，不为每一个候选方向单独指定监督目标，也不在网络损失中直接加入在线未知的真实解余量。

### 2.2 读出 Jacobian 与可达空间

固定查询参数 $\eta$ 以及除 $\mathbf b$ 外的网络参数，有

$$
\boxed{
\frac{\partial\widehat{\mathbf u}_\theta(\eta)}{\partial\mathbf b}
=e^{\ell_\theta(\eta)}\Phi_\theta(\eta).
}
$$

由于 $e^{\ell_\theta(\eta)}>0$，得到

$$
\boxed{
\operatorname{range}\left(
\frac{\partial\widehat{\mathbf u}_\theta(\eta)}{\partial\mathbf b}
\right)
=\operatorname{span}\Phi_\theta(\eta).
}
$$

因此，定义参数相关的**神经输出可达空间**

$$
\boxed{
\mathcal E_\theta(\eta)=\operatorname{span}\Phi_\theta(\eta).
}
$$

这一空间是固定其他网络参数时，改变最后一层读出偏置所能产生的输出变化方向张成的空间。网络的当前预测 $\widehat{\mathbf u}_\theta$ 只是其中一个向量：

$$
\widehat{\mathbf u}_\theta(\eta)\in\mathcal E_\theta(\eta).
$$

“可达空间”是由当前网络结构严格定义的数学对象，不表示任意网络中间特征都天然对应有效的 PDE 试探方向，也不表示网络已经准确预测该空间中的最佳组合。

### 2.3 场值监督与空间有效性

候选场和读出系数通过同一个场值目标联合优化。训练在改善组合预测的同时改变候选空间，但损失函数并未唯一确定候选空间：不同的方向组合可能产生相同或相近的预测结果。

因此，需要区分三项事实：

1. 网络结构**定义**了多方向输出可达空间；
2. 场值监督**优化**了产生这一空间的网络参数；
3. 候选方向能否有效表示基础空间遗漏的解分量，需要通过余量捕获与求解结果**检验**。

NE-PODG 采用第二项形成的候选空间作为空间增广资源，而不将网络直接预测的准确程度等同于整个候选空间的表示能力。

---

## 三、空间构造：相对于基础空间的神经互补增广

### 3.1 非冗余方向提取

神经可达空间 $\mathcal E_\theta(\eta)$ 可能与基础空间 $W$ 重叠。直接拼接候选场和基础基向量可能引入线性相关与不良条件数。NE-PODG 因而提取可达空间中位于 $W$ 正交补内的部分。

在不进行数值秩截断时，理想互补空间为

$$
\mathcal Z_\theta^{\mathrm{ideal}}(\eta)
=(I-P_W)\mathcal E_\theta(\eta).
$$

实际计算先形成候选空间的 $M$-正交基 $Q_E$，再分解 $(I-P_W)Q_E$，对数值不可辨识的方向进行秩筛选，得到 $Z_\theta(\eta)$。记保留的互补空间为 $\mathcal Z_\theta(\eta)=\operatorname{span}Z_\theta(\eta)$，则

$$
\mathcal Z_\theta(\eta)
\subseteq\mathcal Z_\theta^{\mathrm{ideal}}(\eta),
$$

并满足

$$
\boxed{
Z_\theta^{\mathsf T}MZ_\theta=I,
\qquad W^{\mathsf T}MZ_\theta=0.
}
$$

数值秩筛选意味着原始候选方向数 $m$ 不等于一定保留的互补维数 $n_Z(\eta)$。正交化解决方向冗余及数值可辨识性问题，但**非冗余不等于对真实解余量有效**；有效性由第五节的捕获率刻画。

### 3.2 参数相关的增广试探空间

定义增广基

$$
B_\theta(\eta)=[W,Z_\theta(\eta)],
\qquad B_\theta^{\mathsf T}MB_\theta=I.
$$

容许增量状态的仿射试探空间为

$$
\mathcal T_\theta^{(u)}(\eta)
=\bar{\mathbf u}+\operatorname{span}B_\theta(\eta),
$$

相应物理状态的试探空间为

$$
\boxed{
\mathcal T_\theta^{(y)}(\eta)
=\boldsymbol\ell_h(\eta)+\bar{\mathbf u}
+\operatorname{span}[W,Z_\theta(\eta)].
}
$$

基础空间 $W$ 不依赖查询参数，神经互补空间 $\mathcal Z_\theta(\eta)$ 则随查询参数生成。由于

$$
\operatorname{span}W\subseteq\operatorname{span}B_\theta(\eta),
$$

增广空间完整保留基础空间的信息，并提供其之外的数值可辨识自由度。

### 3.3 不依赖未知真实余量的在线构造

互补方向由训练好的网络、已知基础空间和给定查询参数确定：

$$
\boxed{
Z_\theta(\eta)
=\operatorname{orth}_M\big[(I-P_W)\Phi_\theta(\eta)\big],
}
$$

其中 $\operatorname{orth}_M$ 包含上述数值秩筛选。实际实现可先正交化候选空间，再进行去投影，以提高数值稳定性。

这一过程不需要在线高保真参考解，也不需要以 $\boldsymbol\rho_W(\eta)$ 为输入。真实余量仅用于事后的空间表示分析。因而神经互补增广是一种可独立执行的试探空间构造，而不是依赖测试真值的逐工况误差修正。

---

## 四、物理闭合：增广空间上的 Galerkin 投影

### 4.1 与高保真离散相容的有效残差

记处理约束后的稳态控制方程为

$$
r_{\mathrm{eff}}(\mathbf y;\eta)=0,
\qquad r_{\mathrm{eff}}\in\mathbb R^n.
$$

这里 $r_{\mathrm{eff}}$ 表示与离散试验函数相容的**对偶残差向量**，其分量已经包含相应离散弱式检验与求积。对于同时求解多个物理场的问题，允许部分辅助场保留在高保真离散上进行求解。

例如，在主状态 $\mathbf y$ 与辅助场 $\mathbf z$ 满足

$$
F_h(\mathbf y,\mathbf z;\eta)=0,
\qquad H_h(\mathbf y,\mathbf z;\eta)=0
$$

且所追踪解分支上 $H_{\mathbf z}$ 可逆时，可局部表示 $\mathbf z=\mathcal S_\eta(\mathbf y)$，并定义

$$
r_{\mathrm{eff}}(\mathbf y;\eta)
=F_h\big(\mathbf y,\mathcal S_\eta(\mathbf y);\eta\big).
$$

相应的有效 Jacobian 为

$$
J_{\mathrm{eff}}=F_{\mathbf y}-F_{\mathbf z}H_{\mathbf z}^{-1}H_{\mathbf y}.
$$

这一写法允许同一空间增广机制作用于不同流热耦合结构，而不要求全部物理场采用完全相同的降阶表示。

### 4.2 Galerkin 方程与展开系数

给定参数后，最终状态写为

$$
\mathbf y_B(\eta)
=\boldsymbol\ell_h(\eta)+\bar{\mathbf u}
+B_\theta(\eta)\mathbf c(\eta),
$$

其中 $\mathbf c$ 包含基础空间与神经互补空间中的全部待求展开系数。对有效残差实施 Galerkin 投影：

$$
\boxed{
B_\theta(\eta)^{\mathsf T}
 r_{\mathrm{eff}}\left(
\boldsymbol\ell_h(\eta)+\bar{\mathbf u}
+B_\theta(\eta)\mathbf c(\eta);\eta
\right)=0.
}
$$

当控制方程为非线性系统时，在给定参数下将 $B_\theta$ 视为已构造的基矩阵，降阶 Jacobian 为

$$
J_B=B_\theta^{\mathsf T}J_{\mathrm{eff}}B_\theta.
$$

$M$ 用于试探空间正交化和误差评价。由于 $r_{\mathrm{eff}}$ 已是离散对偶残差，在上述 Galerkin 方程中不再额外乘以 $M$。线性问题可直接求解约化方程；非线性问题求解规定目标分支上的投影方程，并按一致的收敛标准判断求解是否成功。

### 4.3 网络表示与物理求解的分工

网络当前读出系数 $w_\theta(\eta)$ **不作为最终 Galerkin 系数**。在线阶段重新求解 $\mathbf c(\eta)$，基础空间与神经互补空间的系数均由控制方程决定。

由此，$\widehat{\mathbf u}_\theta$ 的直接预测误差不构成最终求解误差的必然下界。即使网络当前组合预测不够准确，完整增广空间仍可能包含更合适的方向组合；只要投影方程在该空间中的求解质量适当，就可能获得明显更高的实际精度。

这里的“物理闭合”具体指**满足原离散控制方程在增广试探空间上的 Galerkin 投影条件**，而不是声称全阶残差在每个自由度上都为零。

---

## 五、精度与表示机制：余量捕获、额外方向及闭合放大

### 5.1 余量捕获与最佳逼近的精确关系

给定参数，基础空间的未表示余量为

$$
\boldsymbol\rho_W
=(I-P_W)(\mathbf u^\star-\bar{\mathbf u}).
$$

令 $P_{Z_\theta}=Z_\theta Z_\theta^{\mathsf T}M$。在 $\|\boldsymbol\rho_W\|_M>0$ 时，定义神经互补空间的余量捕获率：

$$
\boxed{
\Gamma_\theta(\eta)
=\frac{\|P_{Z_\theta}\boldsymbol\rho_W\|_M^2}
{\|\boldsymbol\rho_W\|_M^2},
\qquad 0\le\Gamma_\theta\le1.
}
$$

由 $W\perp_M\mathcal Z_\theta$，有 $P_{B_\theta}=P_W+P_{Z_\theta}$，因此

$$
\begin{aligned}
\|(I-P_{B_\theta})(\mathbf u^\star-\bar{\mathbf u})\|_M^2
={}&\|\boldsymbol\rho_W\|_M^2
-\|P_{Z_\theta}\boldsymbol\rho_W\|_M^2.
\end{aligned}
$$

设 $e_{\mathrm{proj},W}$ 和 $e_{\mathrm{proj},B}$ 是同一目标状态、同一 $M$ 范数、同一仿射原点以及同一正归一化尺度下的最佳逼近误差，则

$$
\boxed{
e_{\mathrm{proj},B}
=e_{\mathrm{proj},W}\sqrt{1-\Gamma_\theta}.
}
$$

这是空间正交投影几何给出的**恒等式**。它表明，最佳逼近收益完全取决于新增空间对基础余量的捕获，而不取决于候选方向的原始数量。$\Gamma_\theta=0$ 时最佳逼近误差不变；$\Gamma_\theta=1$ 时该主状态在增广仿射空间内可被精确表示。

### 5.2 当前读出关联方向与额外互补空间

完整神经候选空间包含网络当前组合预测以外的方向。为了识别这些方向对余量捕获的贡献，在已保留的互补空间 $\mathcal Z_\theta=\operatorname{span}Z_\theta$ 内定义当前读出关联向量

$$
\mathbf v_{\mathrm{out}}
=P_{Z_\theta}\widehat{\mathbf u}_\theta,
$$

并据此定义

$$
\boxed{
\mathcal Z_{\mathrm{out}}
=\operatorname{span}(\mathbf v_{\mathrm{out}}),
\qquad
\mathcal Z_{\mathrm{extra}}
=\mathcal Z_\theta\ominus_M\mathcal Z_{\mathrm{out}}.
}
$$

若 $\mathbf v_{\mathrm{out}}=0$，取 $\mathcal Z_{\mathrm{out}}=\{0\}$、$\mathcal Z_{\mathrm{extra}}=\mathcal Z_\theta$。该定义是在互补空间内对**当前网络读出关联方向**进行分解，不是将原始读出系数的零空间直接当作额外空间。

这里使用的是 $P_{Z_\theta}\widehat{\mathbf u}_\theta$，**不是** $P_{Z_\theta}(\widehat{\mathbf u}_\theta-\bar{\mathbf u})$。两者一般不相同。当前定义针对神经读出组合本身；若要检验“基础空间加网络直接预测向量”这一不同命题，应单独考察仿射空间

$$
\bar{\mathbf u}
+\operatorname{span}\big(W,\widehat{\mathbf u}_\theta-\bar{\mathbf u}\big)
$$

的最佳逼近误差。该空间与当前 $W\oplus\mathcal Z_{\mathrm{out}}$ 并不必然一致，现有读出关联方向的结论不能未经检验地替代这一比较。

定义两部分余量捕获率

$$
\Gamma_{\mathrm{out}}
=\frac{\|P_{\mathcal Z_{\mathrm{out}}}\boldsymbol\rho_W\|_M^2}
{\|\boldsymbol\rho_W\|_M^2},
\qquad
\Gamma_{\mathrm{extra}}
=\frac{\|P_{\mathcal Z_{\mathrm{extra}}}\boldsymbol\rho_W\|_M^2}
{\|\boldsymbol\rho_W\|_M^2}.
$$

由于两部分在 $M$ 内积下正交，

$$
\boxed{
\Gamma_\theta=\Gamma_{\mathrm{out}}+\Gamma_{\mathrm{extra}}.
}
$$

仅由基础空间和当前读出关联方向构成的受限空间，其最佳逼近误差满足

$$
\boxed{
e_{\mathrm{proj},W+\mathrm{out}}
=e_{\mathrm{proj},W}\sqrt{1-\Gamma_{\mathrm{out}}}.
}
$$

若完整 NE-PODG 的实际误差满足

$$
\boxed{
e_B<e_{\mathrm{proj},W+\mathrm{out}},}
$$

则在相同参考状态和误差范数下，**仅使用 $W\oplus\mathcal Z_{\mathrm{out}}$ 不可能达到完整模型已获得的精度**。这为额外互补自由度具有实际求解价值提供了可检验的受限空间判据；它并不声称每一个额外候选方向都分别不可替代。

### 5.3 Galerkin 闭合与实际误差

对任一仿射试探空间 $\mathcal T_X^{(u)}=\bar{\mathbf u}+\operatorname{span}X$，设其实际 Galerkin 容许状态为 $\mathbf u_X$。定义

$$
e_X=\frac{\|\mathbf u_X-\mathbf u^\star\|_M}{s(\eta)},
\qquad
e_{\mathrm{proj},X}
=\frac{\|(I-P_X)(\mathbf u^\star-\bar{\mathbf u})\|_M}
{s(\eta)},
$$

其中 $s(\eta)>0$ 为各比较空间共用的归一化尺度；实际算例通常取 $s(\eta)=\|\mathbf u^\star(\eta)\|_M$。因 $\mathbf u_X$ 属于该仿射空间，必有 $e_X\ge e_{\mathrm{proj},X}$。当 $e_{\mathrm{proj},X}>0$ 时，定义闭合放大系数

$$
\boxed{\chi_X=\frac{e_X}{e_{\mathrm{proj},X}}\ge1.}
$$

也可由正交分解写出

$$
e_X^2=e_{\mathrm{proj},X}^2+e_{\mathrm{real},X}^2,
$$

其中 $e_{\mathrm{real},X}$ 表示实际 Galerkin 解偏离该空间最佳逼近解的归一化误差。

对于基础空间 $W$ 和增广空间 $B$，在所涉分母非零时，得到

$$
\boxed{
\frac{e_B}{e_W}
=\underbrace{\sqrt{1-\Gamma_\theta}}_{\text{空间逼近因子}}
\underbrace{\frac{\chi_B}{\chi_W}}_{\text{Galerkin 闭合因子}}.
}
$$

因此，增广空间的实际误差低于基础空间的充要条件为

$$
\boxed{
\sqrt{1-\Gamma_\theta}\frac{\chi_B}{\chi_W}<1.
}
$$

若最佳逼近误差为零，相应 $\chi$ 不再定义，应直接报告最佳逼近与实际误差，而不使用上述比值形式。

这一关系将 NE-PODG 的精度机制分为两部分：

- **空间表示收益：** $\Gamma_\theta$ 衡量神经互补空间对基础余量的捕获，并严格决定最佳逼近误差的下降幅度。
- **物理闭合效果：** $\chi_B/\chi_W$ 表征实际 Galerkin 解相对于各自最佳逼近解的误差放大差异，决定空间表示收益在实际误差中体现的程度。

因此，空间增广保证最佳逼近误差不增加，但一般非对称或非线性系统中的实际 Galerkin 误差不随空间包含关系自动单调降低。NE-PODG 的实际精度改善应由上述两项因素共同解释。

### 5.4 主状态误差与传热物理输出

上述 $\Gamma_\theta$、$e_{\mathrm{proj}}$、$\chi$ 与误差恒等式针对**被降阶主状态**建立。不同问题中，主状态可以是温度，也可以是经约束处理后的流动变量；某些辅助场仍在全阶离散上求解。

温度、速度、平均或局部 Nusselt 数、流固界面热流、再附着长度等物理输出的精度需要结合各自定义单独评价。主状态最佳逼近关系并不直接给出这些派生输出的误差界。因而，传热输出验证承担独立的数值证据作用，而不是作为主状态误差公式的直接推论。

---

## 六、科学贡献、证据层次与适用范围

### 6.1 方法定位与已有研究的关系

已有神经 ROM 研究已经将神经网络用于解流形表示、降阶模型增强与物理投影求解。特别是 Peng、Wang 和 Li 的学习型投影降阶方法，已经建立了神经基函数学习与在线投影求解的结合；因此，NE-PODG 不将这一组合本身表述为首次提出。

NE-PODG 的具体贡献在于：**在已有快照—灵敏度基础空间上，利用场值监督形成的神经候选空间进行非冗余互补增广，并以余量捕获、当前读出之外的额外空间贡献及 Galerkin 闭合放大关系，定量解释新增自由度的实际求解价值。**

这一定位强调方法构造与科学解释之间的联系，而非把快照、灵敏度、神经网络、正交化和 Galerkin 分别列为独立创新点。

相关技术先例：Peng Z, Wang M, Li F. *A learning-based projection method for model order reduction of transport problems*. Journal of Computational and Applied Mathematics, 2023, 418: 114560. DOI: 10.1016/j.cam.2022.114560。其他神经增强投影 ROM 的具体文献与引用由正式 Introduction 承担。

### 6.2 已建立的数学性质

在共同容许状态坐标、正定度量和一致仿射原点下，NE-PODG 具有以下可直接从定义推出的性质：

1. **网络空间定义：** 对指定仿射读出结构，候选空间等于网络输出相对于最后一层读出偏置的变化方向所张成的空间。
2. **基础空间保留：** $\operatorname{span}W\subseteq\operatorname{span}B_\theta$，因此增广空间的最佳逼近误差不高于基础空间。
3. **余量捕获恒等式：** $e_{\mathrm{proj},B}=e_{\mathrm{proj},W}\sqrt{1-\Gamma_\theta}$。
4. **读出贡献分解：** $\Gamma_\theta=\Gamma_{\mathrm{out}}+\Gamma_{\mathrm{extra}}$；受限空间的最佳逼近下界可检验额外自由度的必要性。
5. **实际误差恒等式：** $e_B/e_W=\sqrt{1-\Gamma_\theta}\,\chi_B/\chi_W$，前提是相关比值有定义。

这些是空间结构及误差定义的数学结论，不依赖神经网络训练必然成功的假设。

### 6.3 当前数值证据支持的认识

**场值监督学习提高了互补空间的余量捕获能力。** 在当前自然对流主算例中，相同候选方向数 $m=256$、各三组随机种子的对照显示：插值评估时，训练网络与未训练网络的完整互补空间余量捕获率合并中位数分别约为 $0.99967$ 和 $0.68714$；外推评估时分别约为 $0.99515$ 和 $0.49643$。这一对照支持训练所形成的空间表示资源具有实质性的有效性，而不仅是增加候选维数。

**当前读出之外的方向承担了主要的余量捕获。** 在上述训练网络中，额外空间的余量捕获率合并中位数在插值与外推评估下分别约为 $0.99538$ 与 $0.99155$。对当前定义的读出关联受限空间，已有配对测试进一步满足 $e_B<e_{\mathrm{proj},W+\mathrm{out}}$，直接表明该受限空间不足以达到完整 NE-PODG 的求解精度。未训练网络中余量捕获同样主要分布于额外空间，因此训练效应应从**捕获能力的绝对增强**判断，而不是仅根据额外空间所占比例判断。

**表示收益与求解收益具有可分辨的关系。** 已有多类参数化稳态流动与传热算例显示，神经互补增广能够降低主状态的最佳逼近误差，并在所研究的已收敛求解中体现为实际误差下降。不同控制算子与耦合形式下，闭合放大系数之比并不完全相同；因此跨算例分析应同时报告 $\Gamma_\theta$ 和 $\chi_B/\chi_W$，而非仅以误差下降倍数替代机制解释。

**传热输出改善需要独立验证。** 现有算例已经提供温度场、速度场、壁面换热、界面热流等证据。正文应选择能直接支持核心主张的代表性结果，而不按实验完成顺序罗列所有探索内容。

上述数字对应现阶段自然对流主算例的既有统计口径；原始数据和完整参数条件由算例记录承载。后续若调整误差定义、读出关联方向或数据筛选口径，应同步更新本节结论，不能沿用旧数值。

### 6.4 成立条件与实际计算范围

方法使用以下条件与约定：

- **共同容许坐标：** 快照、灵敏度与所有候选方向处于同一离散容许状态空间，并采用一致的边界提升、参考表示和误差度量。
- **离散残差相容：** 投影残差与生成训练快照及灵敏度的高保真离散相容；有效残差采用对偶形式。若保留全阶辅助场，则其局部消元条件和求解一致性必须成立。
- **解分支与灵敏度：** 目标稳态解分支应有明确选择，训练点附近所需参数导数存在；在采用隐式灵敏度求解时，相应有效 Jacobian 可逆。
- **数值秩筛选：** 实际 $Z_\theta$ 是数值可辨识的互补空间，维数及稳定性受精度和秩阈值影响；空间包含与误差恒等式针对实际保留的 $Z_\theta$ 成立。
- **求解精度与成本：** Galerkin 解必须满足规定的收敛标准；网络生成、正交化和投影求解均具有在线计算代价。当前研究的主要证据是空间表示和数值精度收益，不将普遍在线加速作为方法的核心主张。

场值监督本身不保证额外候选方向对每个参数均有效，空间包含关系也不保证一般非对称或非线性系统的实际误差必然下降。方法的数学保证与数值观察应按照各自的证据性质陈述。

### 6.5 统一科学论证主线

NE-PODG 的核心不是用网络替换高保真方程，也不是将网络当前预测作为最终降阶解，而是利用训练网络生成的完整候选空间，为已经具有快照和灵敏度信息的基础 ROM 提供额外表示自由度。

这一科学论证由三个连续问题组织：**神经候选空间能够提供什么额外方向；这些方向捕获了多少基础空间遗漏的真实解分量；这种最佳逼近收益在 Galerkin 闭合后形成了怎样的实际求解效果。**

其统一定量表达为

$$
\boxed{
\begin{gathered}
\mathcal E_\theta(\eta)=\operatorname{span}\Phi_\theta(\eta),
\qquad
\mathcal Z_\theta(\eta)\subseteq(I-P_W)\mathcal E_\theta(\eta),
\\[5pt]
e_{\mathrm{proj},B}
=e_{\mathrm{proj},W}\sqrt{1-\Gamma_\theta},
\\[5pt]
\frac{e_B}{e_W}
=\sqrt{1-\Gamma_\theta}\frac{\chi_B}{\chi_W}.
\end{gathered}
}
$$

因此，**神经候选空间的价值由其对既有基础空间剩余解分量的捕获能力体现；该表示收益能否转化为实际求解精度，由 Galerkin 闭合效果共同决定。** 这构成 NE-PODG 的方法定义、机制分析、数值证据筛选以及论文各章节论证的一致基准。
