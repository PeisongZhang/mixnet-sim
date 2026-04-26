# MixNet 复现计划（30 GB RAM、无 GPU 主机）

本计划针对本机环境（30 GiB RAM + 8 GiB swap、20 vCPU、无 NVIDIA GPU、无 CUDA、磁盘可用 1.5 TB），目标是在 SIGCOMM 2025 论文 *MixNet* 的开源实现 (`mixnet-flexflow` + `mixnet-htsim`) 上跑通 Mixtral-8x22B 的 mixnet vs. fattree 对比实验。

---

## 0. 关键判断

1. **不在本机构建 FlexFlow**：`config/config.linux` 默认 `FF_GPU_BACKEND=cuda`，`INSTALL.md` 明确写无 GPU 也必须给 `FF_CUDA_ARCH`，而本机连 `nvcc` 都没有；硬走 CPU-only Legion 路径成本远高于直接下载作者预生成的任务图。
2. **使用 README 提供的预生成 `.fbuf`**（[Google Drive 链接](https://drive.google.com/drive/folders/1hChT-tVYJwBSCAC_hTm3x99JLcnl3vRk?usp=sharing)）替代 Step 1。
3. **htsim 必须 v24.3.25 头文件兼容的 flatbuffers**：`mixnet-flexflow/include/flexflow/taskgraph_generated.h` 内含
   ```cpp
   static_assert(FLATBUFFERS_VERSION_MAJOR == 24 &&
                 FLATBUFFERS_VERSION_MINOR == 3 &&
                 FLATBUFFERS_VERSION_REVISION == 25, ...);
   ```
   Ubuntu 22.04 自带 `libflatbuffers-dev` 版本太老，不能用；需要从 GitHub 拉 v24.3.25 的 header-only。
4. **htsim 的 Makefile 使用 `-O0 -g`（debug）**，单文件编译峰值 ~1–2 GB；`make -j` 全开 20 核可能瞬时占 20–40 GB → 必须 `-j4` 限并发。
5. **运行时主要内存来源**：
   - `FFApplication::load_taskgraph_flatbuf` 一次性 `new` 出所有 task 对象（百万量级）；
   - 1024 节点 fat-tree 拓扑（k=8 → 80 个交换机）+ TCP 队列 + 在飞包；
   - 经验上 1024-node + Mixtral-8x22B 全量 fbuf 的 mixnet sim 峰值 RSS 接近甚至超过 24 GB，需要 cgroup + swap 兜底。
6. **作者发布的 onestage 脚本默认 512 节点 + pp=1**，是更安全的冒烟入口；但 Drive 上是否有对应的 onestage fbuf 需要下载后确认（脚本期望 `mixtral8x22B_onestage_dp2_tp8_pp1_ep8_8.fbuf`）。

---

## 1. 总体步骤

```
Step A. 下载预生成任务图   (~ 数百 MB / 个)
Step B. 准备 flatbuffers v24.3.25 头文件
Step C. 编译 htsim (限 -j4)
Step D. 改写运行脚本中的硬编码路径，去掉后台 &
Step E. 冒烟：先跑 onestage(512N) 或最小 microbatch
Step F. 1024N 全量复现：cgroup 限内存 + 串行 + RSS 监控
Step G. 收集 logs/ 下结果，与论文图对照
```

---

## 2. 详细操作清单

### Step A — 下载预生成 `.fbuf`

- 目标目录：`/home/ps/sow/part2/mixnet-sim/mixnet-flexflow/results/`（与脚本默认 `new_fbuf_dir` 命名一致，能少改一行）。
- 优先下 `mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf`（mb=8 主算例）。
- 工具建议：`gdown`（已存在的 venv 可装）：
  ```bash
  /home/ps/sow/part2/astra-sim/.venv/bin/pip install gdown
  /home/ps/sow/part2/astra-sim/.venv/bin/gdown --folder \
      'https://drive.google.com/drive/folders/1hChT-tVYJwBSCAC_hTm3x99JLcnl3vRk' \
      -O /home/ps/sow/part2/mixnet-sim/mixnet-flexflow/results
  ```
- 校验：下载完后 `ls -lh results/*.fbuf`，记录文件大小（评估 task graph 规模）。

### Step B — 准备 flatbuffers 头

```bash
cd /tmp && rm -rf flatbuffers
git clone --depth 1 -b v24.3.25 https://github.com/google/flatbuffers.git
mkdir -p /home/ps/sow/part2/mixnet-sim/mixnet-flexflow/fbuf/include
cp -r /tmp/flatbuffers/include/flatbuffers \
      /home/ps/sow/part2/mixnet-sim/mixnet-flexflow/fbuf/include/
cp /home/ps/sow/part2/mixnet-sim/mixnet-flexflow/include/flexflow/taskgraph_generated.h \
   /home/ps/sow/part2/mixnet-sim/mixnet-flexflow/fbuf/include/
```

校验：
```bash
ls /home/ps/sow/part2/mixnet-sim/mixnet-flexflow/fbuf/include/flatbuffers/flatbuffers.h \
   /home/ps/sow/part2/mixnet-sim/mixnet-flexflow/fbuf/include/taskgraph_generated.h
```

### Step C — 编译 htsim

```bash
cd /home/ps/sow/part2/mixnet-sim/mixnet-htsim
( cd src/clos       && make clean && make -j4 )
( cd src/clos/datacenter && make clean && make -j4 )
```

预期产物：
- `src/clos/libhtsim.a`
- `src/clos/datacenter/htsim_tcp_mixnet`
- `src/clos/datacenter/htsim_tcp_fattree`
- `src/clos/datacenter/htsim_tcp_os_fattree`

如果 `-j4` 仍出现内存压力，降到 `-j2` 或 `make` 单线程。

### Step D — 改写运行脚本

`mixnet_scripts/*.sh` 里两处硬编码绝对路径，需要改成：
```bash
dir="/home/ps/sow/part2/mixnet-sim/mixnet-htsim/src/clos/datacenter"
new_fbuf_dir="/home/ps/sow/part2/mixnet-sim/mixnet-flexflow/results"
```
建议复制一份到本仓库根 `run/` 目录，避免污染上游 submodule。

另外把脚本最后一行命令末尾的 `&` 删掉（默认是后台并行多 sweep，本机内存撑不起多进程）。

### Step E — 冒烟测试（择一）

**E1. onestage（首选，若 Drive 有对应 fbuf）**：
- 脚本：`onestage_mixtral_8x22B_mixnet.sh`
- 规模：512 节点、pp=1，预计 RSS < 12 GB
- 如果没有 onestage 版本的 fbuf，跳到 E2。

**E2. 1024 节点最小 microbatch**：
- 用 `mixtral_8x22B_mixnet.sh`，但只跑一组配置（mb=8、bw=100、rdelay=25、num_global_tokens_per_expert）
- 加 `/usr/bin/time -v` 收集峰值 RSS

执行模板：
```bash
mkdir -p logs/smoke
/usr/bin/time -v ./htsim_tcp_mixnet \
    -simtime 3600.1 \
    -flowfile $new_fbuf_dir/mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf \
    -speed 100000 \
    -ocs_file nwsim_ocs_100.txt \
    -ecs_file nwsim_ecs_100.txt \
    -nodes 1024 -ssthresh 10000 -rtt 1000 -q 10000 \
    -dp_degree 2 -tp_degree 8 -pp_degree 8 -ep_degree 8 \
    -rdelay 25 \
    -weightmatrix ../../../test/num_global_tokens_per_expert.txt \
    -logdir ./logs/smoke/ \
    > ./logs/smoke/output.log 2>&1
```

**通过判据**：
- `output.log` 出现 `load_taskgraph_flatbuf: start load tasks` 后正常结束；
- `logs/smoke/` 内有 `nwsim_ocs_100.txt` / `nwsim_ecs_100.txt` 输出；
- `time -v` 报告的 `Maximum resident set size` < 24 GB；
- 进程不被 OOM Killer 杀（`dmesg | tail` 无 `Out of memory`）。

### Step F — 1024 节点全量复现

只有冒烟通过且 RSS 留有余量时再做。

1. **临时增大 swap**（防 OOM，但仅作保险）：
   ```bash
   sudo fallocate -l 16G /swap_extra
   sudo chmod 600 /swap_extra && sudo mkswap /swap_extra && sudo swapon /swap_extra
   ```
   实验后 `sudo swapoff /swap_extra && sudo rm /swap_extra`。

2. **cgroup 限内存**（推荐）：
   ```bash
   systemd-run --scope --user \
       -p MemoryMax=24G -p MemorySwapMax=16G \
       ./htsim_tcp_mixnet ... > out.log 2>&1
   ```
   触上限会被杀但不会拖死整机。

3. **逐配置串行跑**：bw、microbatch、rdelay、workload sweep 一组一组跑，**绝不并行**。

4. **Mixnet vs FatTree 对比**：跑完 mixnet 再跑 fattree baseline（`mixtral_8x22B_fattree.sh`），同样去掉 `&`、串行执行。

### Step G — 结果收集

每个 `logs/<config>/` 目录预期产物：
- `output.log`：标准输出
- `nwsim_ocs_<bw>.txt`：OCS 链路 FCT/util
- `nwsim_ecs_<bw>.txt`：ECS（电交换）链路 FCT/util

后处理脚本作者未提供，可先 grep 关键指标（finished iteration time、average link util）做横向对比，再视需要写小 Python 脚本聚合到 CSV / 画图。

---

## 3. 风险与回退

| 风险 | 触发信号 | 应对 |
|---|---|---|
| flatbuffers 版本不匹配 | 编译报 `static_assert` 失败 | 重新核对拉的 v24.3.25 tag |
| 编译期 OOM | gcc 被 SIGKILL / 系统卡死 | `make -j2` 或单线程 |
| 加载 taskgraph 阶段 OOM | `load_taskgraph_flatbuf: start load tasks` 后被杀 | 换更小 microbatch fbuf；或先跑 onestage |
| 仿真期间 OOM | `time -v` 显示 RSS 持续增长 → OOM Killer | 加 swap / cgroup；或缩 simtime 看是否能跑出至少 1 个 iteration |
| Drive 无 onestage fbuf | gdown 列表里看不到 onestage 文件 | 直接跳到 E2，并接受 1024N 风险 |
| pre-generated fbuf 缺某 microbatch | 只有 mb=8 | 仅复现该一组结果，论文其它点放弃 |

---

## 4. 工时估算

| 阶段 | 主要耗时 | 预估 |
|---|---|---|
| A. 下 fbuf | 网络 + 几百 MB | 10–30 min |
| B. flatbuffers headers | git clone | 2 min |
| C. htsim 编译 | -O0 全量 | 10–20 min |
| D. 脚本改写 | 手改 | 5 min |
| E. 冒烟一组 | 仿真本身 | 30 min – 数小时（视 task 规模） |
| F. 全量 sweep | 多组串行 | 数小时 – 1 天 |
| G. 结果整理 | 自写小工具 | 1–2 h |

---

## 5. 不在本计划范围

- 重新生成 `.fbuf`（需要 GPU + FlexFlow 完整构建，非 30 GB / 无 GPU 主机能本地完成）。
- TopoOpt / Opera baseline 之外的其他论文图（除非 Drive 上提供了对应 fbuf）。
- 跨多机分布式仿真（htsim 单进程，不涉及）。

---

## 6. 下一步动作

按顺序：
1. 用户确认本计划。
2. 执行 Step A → B → C，先把构建跑通，再回报 `htsim_tcp_mixnet` 是否可执行。
3. 决定 E1 / E2 入口，开始冒烟。
