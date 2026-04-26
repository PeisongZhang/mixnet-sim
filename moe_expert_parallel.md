# MoE Expert Parallelism：在 Transformer 结构中的位置与通信模式

> 配图见 `moe.dio` (drawio)，共 3 页：
> 1. `1-Transformer-Structure` — EP 在 Transformer block 中的位置
> 2. `2-MoE-Layer-Dataflow` — MoE 层内部的 dataflow（含 2 次 AllToAll）
> 3. `3-Traffic-Matrix` — 均匀 vs 倾斜 gating 的通信量矩阵对比

---

## 1. 结构定位：EP 只存在于 MoE 层（替换了 FFN）

标准 Transformer block 有两个子层：

1. **多头自注意力 (MHA / GQA)**
2. **前馈网络 (FFN)**，标准的 2-层 MLP（现代 LLM 常用 SwiGLU 三矩阵变体）

**MoE 模型**（Switch Transformer, GShard, GLaM, Mixtral, DeepSeek-MoE, ...）只对 **FFN 子层** 做替换：

- **Router (gate)** $W_g \in \mathbb{R}^{H \times E}$：线性层，对每个 token 给出 $E$ 个 expert 的打分，选 top-$k$
- **$E$ 个 Expert FFN**：每个就是一个普通 FFN（参数量约 $3 H H_{ff}$，SwiGLU 的话）

**Expert Parallelism (EP)** 就是把这 $E$ 个 expert 切分到 EP 个 rank 上，每 rank 承载 $E/\text{EP}$ 个 expert。Attention 子层**完全不涉及 EP**，它用 DP / TP / SP / PP。

| 子层 | 适用并行 | 有 EP？ |
|---|---|---|
| Attention (QKV, Proj, MHA) | DP / TP / SP / PP | ❌ |
| Router (gate) | 通常在每 rank 复制（参数小） | ❌ |
| Expert FFN | DP / TP / PP / **EP** | ✅ |

**MoE 插入密度**（不同论文不同）：

- Switch Transformer / Mixtral / DeepSeek-MoE：每层都是 MoE
- GShard：**奇偶交替**，偶数层 MoE，奇数层 dense FFN
- GLaM：类似 GShard 的交替

> 详见 `moe.dio` 第 1 页。

---

## 2. MoE 层内部的 dataflow

### 记号

| 符号 | 含义 |
|---|---|
| $B$ | batch size（per DP rank） |
| $S$ | sequence length |
| $N = B \cdot S$ | 每 rank 的 token 数 |
| $H$ | hidden size |
| $H_{ff}$ | FFN 中间层宽度 |
| $E$ | 专家总数 |
| $k$ | top-k（每 token 激活的 expert 数） |
| $P = \mathrm{EP}$ | EP group 大小 |
| $b$ | 每元素字节数（bf16: $b=2$） |

### 前向流程（per rank 视角）

```
x  [N, H]
   │
   ▼
Router: logits = x · W_g,   top-k indices + weights
   │
   ▼
Permute + bucket      (按目标 expert 排序)
   │
   ▼
AllToAll  #1  DISPATCH   ← 跨 EP group 重分布 token
   │
   ▼
Local experts FFN        (每 rank 有 E/P 个 expert)
   │
   ▼
AllToAll  #2  COMBINE    ← 结果回发源 rank
   │
   ▼
Unpermute + weighted sum:  y = Σ_{j=1..k} g_j · FFN(x)_j
   │
   ▼
y  [N, H]
```

反向过程镜像：combine 的梯度经 AllToAll 发回 expert 侧，experts 做 backward，再经 AllToAll 把 input 梯度发回源 rank。

**每 MoE 层每次 iteration = 4 次 AllToAll**（2 前向 + 2 反向）。$L$ 个 MoE 层 → 每 iter $4L$ 次 AllToAll。对比 DP 的 AllReduce 每 iter 只有 1 次，MoE 的通信压力主要来自这 $4L$ 次 AllToAll。

> 详见 `moe.dio` 第 2 页。

---

## 3. 通信模式的数学刻画

### 3.1 均匀 gating 假设下（理想情况）

定义通信量矩阵 $M \in \mathbb{R}^{P \times P}$，$M[i,j]$ = rank $i$ 发到 rank $j$ 的字节数。

在完美负载均衡下（每个 token 被均匀分到所有 expert），每个 rank 向每个对端发送的 token 数：

$$
n_{i \to j} = \frac{k \cdot N}{P}
$$

对应字节数：

$$
M[i, j] = \frac{k \cdot N \cdot H \cdot b}{P} \quad (i \ne j)
$$

每 rank 每次 AllToAll 的 egress 总量：

$$
R_i \;=\; \sum_{j \ne i} M[i,j] \;=\; \frac{P-1}{P} \cdot k \cdot N \cdot H \cdot b \;\approx\; k \cdot N \cdot H \cdot b
$$

每 MoE 层每 iteration 每 rank 的总 EP 通信量（4 次 AllToAll）：

$$
T_\text{MoE-layer} \;\approx\; 4 \cdot k \cdot N \cdot H \cdot b
$$

### 3.2 真实 gating → AllToAllv

真实训练中，gating 输出非均匀分布。令 $T_{i \to e}$ = rank $i$ 本地 token 中被路由到 expert $e$ 的数量，则：

$$
M[i, j] \;=\; H \cdot b \cdot \sum_{e \in \mathrm{experts}(j)} T_{i \to e}
$$

即 rank $i$ 发到 rank $j$ 的字节数，等于"rank $i$ 的 token 中目标 expert 位于 rank $j$"的 token 数 × $H \cdot b$。

### 3.3 性能下界（α-β / Hockney 模型）

设链路带宽 $B_\text{net}$，则 AllToAllv 的完成时间下界：

$$
T \;\ge\; \frac{1}{B_\text{net}} \cdot \max\!\left(\, \max_i R_i, \;\max_j C_j \,\right)
$$

其中

- $R_i = \sum_j M[i,j]$：rank $i$ 的 egress 总量
- $C_j = \sum_i M[i,j]$：rank $j$ 的 ingress 总量

**最慢的 sender 或 receiver 决定墙钟** —— 这是 expert overflow 和 straggler 的根本原因。

定义负载不均衡度：

$$
\mathrm{LIF} \;=\; \frac{\max_i R_i}{\mathrm{avg}_i\, R_i}, \qquad \mathrm{LIF}_\text{ingress} \;=\; \frac{\max_j C_j}{\mathrm{avg}_j\, C_j}
$$

均匀 gating 下 $\mathrm{LIF} = 1$，倾斜越重 $\mathrm{LIF}$ 越大。

### 3.4 统计特征一览

| 指标 | 含义 | 对网络设计的意义 |
|---|---|---|
| 行和 $R_i$、列和 $C_j$ | rank 的 egress / ingress | 决定完成时间下界 |
| 熵 $H(M) = -\sum p_{ij} \log p_{ij}$ | 通信分布的均匀度 | 均匀时 $\to 2\log P$，倾斜时显著偏离 |
| 基尼 / Zipf 指数 $\alpha$ | 重尾程度 | MoE 实测 $\alpha \in [0.8, 1.5]$ |
| 切割带宽 $\mathrm{Cut}(S) = \sum_{i\in S, j\notin S} M[i,j]$ | 2-partition 上的跨分区流量 | 光路重配（MixNet/TopoOpt）的优化目标 |
| 时序自相关 $\mathrm{corr}(M_t, M_{t+1})$ | gating 慢变性 | 使"上一步预测下一步"成立（Tutel/MixNet） |
| 稀疏率 $\lVert M \rVert_0 / P^2$ | 非零占比 | top-$k$ 小时可有显著零元素 |

> 详见 `moe.dio` 第 3 页（均匀 vs 倾斜的通信量矩阵对比）。

---

## 4. 与其他并行维度的通信对比

| 来源 | 类型 | 每次通信量（per rank egress） | 每 iter 次数 |
|---|---|---|---|
| DP（权重梯度同步） | AllReduce | $\sim 2 \cdot \text{params} / \text{DP}$ | 1（iter 尾） |
| TP（Megatron-style） | AllReduce | $\sim B S H b$ per sub-layer | $\mathcal{O}(L)$ |
| PP | P2P SendRecv | $B S H b$ per stage boundary | $\mathcal{O}(\text{PP})$ |
| **EP（MoE）** | **AllToAll / AllToAllv** | $\sim k \cdot N \cdot H \cdot b$ | **$4 L$** |

MoE 的 AllToAll 往往是**主导瓶颈**，原因：

1. **次数多**：每 MoE 层 4 次，L 层模型 $4L$ 次
2. **全连接通信**：所有 EP rank 两两通信，难以用局部性优化
3. **gating 倾斜**：AllToAllv 形态下 ingress/egress 不均衡，拖慢整体

---

## 5. 关键观察

1. **EP 的通信全部集中在替代 FFN 的 MoE 层**。Attention 子层跟 EP 解耦。
2. **每个 MoE 层每 iter = 4 次 AllToAll**，远超 DP 的 1 次 AllReduce，通常是最大通信热点。
3. **真实 gating 让 AllToAll 变成 AllToAllv**，完成时间受 $\max(\max_i R_i, \max_j C_j)$ 限制，即最忙的 sender 或 receiver。
4. **ASTRA-sim / STG 假设 gating 完美均匀**（见 `symbolic_tensor_graph/models/stage1/moe_model.py:50` 的 `experts_each_group = experts / ep`），Chakra ET 协议里通信是单标量 `comm_size`，不建模倾斜。
5. **mixnet-sim 专门建模倾斜** —— `htsim_tcp_mixnet` 的 `-weightmatrix <path>` 参数（仓库附的 `test/num_global_tokens_per_expert.txt`）就是实测的 token-per-expert 分布，驱动 pod 级光路重配。
6. **token imbalance 是网络优化的杠杆**：capacity factor、expert choice routing、FlexMoE/SmartMoE 的 expert placement 再平衡、MixNet / TopoOpt 的光路重配，共同动机都是"把倾斜的 $M$ 变平"或"把热对 (i,j) 映射到高带宽链路"。

---

## 附录 A：Mixtral-8x22B 示例参数

- $H = 6144$，$H_{ff} = 16384$，$E = 8$，$k = 2$
- 典型并行配置：DP=2, TP=8, PP=8, **EP=8**
- bf16：$b = 2$
- 假设 $B = 128$，$S = 8192$，DP=2 后每 rank $N = 128 \cdot 8192 / 2 = 524288$ tokens

每 rank 每层每次 AllToAll 的 egress：

$$
k \cdot N \cdot H \cdot b \;=\; 2 \cdot 524288 \cdot 6144 \cdot 2 \;\approx\; 12.9 \text{ GB}
$$

每 MoE 层每 iter 共 4 次 AllToAll，即 **~51.5 GB / 层 / iter / rank**。若所有 $L = 56$ 层都是 MoE（Mixtral 风格），单 iter 仅 EP 通信量就达到 **~2.9 TB / rank / iter**。这就是为什么 MoE 训练对网络如此敏感。

## 附录 B：为什么 k 是乘数而不是 1

top-$k$ gating 里，每个 token 会 **复制 $k$ 份** 各发一个 expert。dispatch 阶段发出的 token 数是 $k \cdot N$，combine 阶段收回的也是 $k \cdot N$ 个结果（随后在 unpermute 里做 $k$-路加权求和降维回 $N$ 个）。所以通信量里的 $k$ 因子来自 top-$k$ 选择的天然冗余。

Switch Transformer 选 $k=1$ 就是为了压这个因子到最小；Mixtral、GShard 用 $k=2$；某些方案用 $k=4$ 或更多。
