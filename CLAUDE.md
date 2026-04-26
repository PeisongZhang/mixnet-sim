# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository identity

This repo is the public simulation bundle for **MixNet** (ACM SIGCOMM 2025), a runtime-reconfigurable optical-electrical fabric for distributed MoE training. It is a *meta-repo* — the actual simulator code lives in two git submodules that point at **PeisongZhang forks** (not the upstream `mixnet-project` repos; see commit `4508ed9`):

- `mixnet-flexflow/` — fork of FlexFlow, used **only** to emit a hybrid-parallel task graph as a FlatBuffer (`*.fbuf`). GPU-dependent (needs CUDA + `nvcc` + `FF_CUDA_ARCH`).
- `mixnet-htsim/` — fork of the htsim packet-level simulator (descended from TopoOpt NSDI'23 and Opera NSDI'20). Consumes the `.fbuf` and evaluates MixNet's reconfiguration logic. CPU-only, but build and run are memory-heavy.

Before doing anything, fetch submodules: `git submodule update --init --recursive`.

The two pre-downloaded FlatBuffers at the repo root (`mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf` ~308 MB and `mixtral8x22B_onestage_dp2_tp8_pp1_ep8_8.fbuf` ~4.5 MB) are the author-provided pre-generated task graphs ([Google Drive](https://drive.google.com/drive/folders/1hChT-tVYJwBSCAC_hTm3x99JLcnl3vRk?usp=sharing)). They let you skip Step 1 entirely on machines without a GPU.

## The two-step pipeline

1. **Task graph generation** (`mixnet-flexflow`, GPU-only) — `./build/examples/cpp/mixture_of_experts/moe --taskgraph ...` emits `taskgraph.fbuf`. Parameters: `--train_dp/tp/pp`, `--expnum`, `--topk`, `--batchsize`, `--microbatchsize`, `--num-layers`, `--embedding-size`, `--expert-hidden-size`, `--hidden-size`, `--num-heads`, `--sequence-length`. See README.md for the canonical Mixtral-8x22B invocation. Total NPUs = `dp*tp*pp` (EP is a routing view over DP×TP, not multiplicative).
2. **Packet-level simulation** (`mixnet-htsim`) — `htsim_tcp_mixnet` / `htsim_tcp_fattree` / `htsim_tcp_os_fattree` consume the `.fbuf` plus a weight matrix (expert-token distribution). Output: `nwsim_ocs_<bw>.txt`, `nwsim_ecs_<bw>.txt` per run inside `logs/<config>/`.

Ready-to-use sweep scripts live in `mixnet-htsim/mixnet_scripts/`:
- `mixtral_8x22B_{mixnet,fattree}.sh` — full 1024-node, pp=8 sweep.
- `onestage_mixtral_8x22B_{mixnet,fattree}.sh` — 512-node, pp=1 smoke test (needs the `onestage_*.fbuf`).

All four scripts have **hardcoded `/usr/wkspace/...` absolute paths** and end each command with `&` to launch background sweeps. Always rewrite the two path variables (`dir`, `new_fbuf_dir`) and usually strip the trailing `&` before running on this machine — a parallel sweep will OOM the host.

## Build

htsim is the only thing buildable on this host. Use the helper script (cleans clos → builds clos → builds datacenter):

```bash
cd mixnet-htsim && bash mixnet_scripts/compile.sh
```

Or manually, which is what you want when you need to cap parallelism:

```bash
cd mixnet-htsim/src/clos       && make -j4
cd mixnet-htsim/src/clos/datacenter && make -j4
```

Important build details:

- **Flatbuffers v24.3.25 header-only is required.** `mixnet-flexflow/include/flexflow/taskgraph_generated.h` has a `static_assert(FLATBUFFERS_VERSION_MAJOR==24 && _MINOR==3 && _REVISION==25, ...)`. Ubuntu 22.04 `libflatbuffers-dev` is too old. Both Makefiles (`src/clos/Makefile`, `src/clos/datacenter/Makefile`) read `FF_HOME?=$(PWD)/../../../FlexFlow` and expect headers at `$(FF_HOME)/fbuf/include/`. `REPRODUCE_PLAN.md` documents the workaround: clone v24.3.25 from GitHub into `mixnet-flexflow/fbuf/include/` and also drop a copy of `taskgraph_generated.h` there.
- Makefiles compile with `-O0 -g` (debug). Peak RSS per `g++` can reach 1–2 GB; `make -j` with all cores can momentarily balloon to 20–40 GB. **Cap with `-j4` or lower** on the 30 GiB host.
- Gurobi is gated behind `USE_GUROBI` (disabled by default) — leave it off unless you know what you're doing; the hardcoded `/usr/wkspace/3dparty/gurobi1000/...` path doesn't exist here.

Executables land in `mixnet-htsim/src/clos/datacenter/`:

| Binary | Topology |
| --- | --- |
| `htsim_tcp_fattree` | Fat-tree baseline |
| `htsim_tcp_os_fattree` | Oversubscribed fat-tree (ToR oversubscribed) |
| `htsim_tcp_mixnet` | MixNet reconfigurable optical-electrical fabric |

Others (`htsim_tcp_fc`, `htsim_tcp_dyn_flat`, `htsim_tcp_flat`, multijob variants) build but are from the ancestor TopoOpt/Opera codebase and aren't part of the paper's evaluation.

## Running a simulation

Required args to `htsim_tcp_mixnet` (see `src/clos/datacenter/main_tcp_mixnet.cpp` for the full parser):

- `-simtime <sec>` — wall-clock cap on simulated time.
- `-flowfile <path.fbuf>` — the FlexFlow task graph.
- `-speed <Mbps>` — link speed (e.g. `100000` for 100 Gbps).
- `-nodes <N>` — 1024 for full Mixtral-8x22B, 512 for onestage.
- `-dp_degree / -tp_degree / -pp_degree / -ep_degree` — must match the `.fbuf`'s parallelism dims.
- `-rdelay <us>` — optical reconfiguration delay in microseconds (paper sweeps 25/50/100).
- `-weightmatrix <path>` — expert-token-count matrix. Repo ships `mixnet-htsim/test/num_global_tokens_per_expert.txt`.
- `-ocs_file / -ecs_file` — output filenames for optical-circuit / electrical-packet link stats (written into the current working dir).
- `-logdir <dir>` — where `htsim_tcp_mixnet` writes run logs.
- `-ssthresh`, `-rtt`, `-q` — TCP knobs; scripts use `10000`, `1000`, `10000` respectively.

`htsim_tcp_fattree` accepts roughly the same args but ignores `-rdelay`, `-ocs_file`, `-ecs_file`.

## Code architecture (mixnet-htsim)

All sources are in `mixnet-htsim/src/clos/` (flat layout, no subdirs for the core library). The MixNet-specific extensions over stock htsim are:

- `ffapp.{h,cpp}` — parses the FlatBuffer task graph and injects flows into the event list. Entry point: `FFApplication::load_taskgraph_flatbuf`. This one call allocates millions of `FFTask` objects up front; it's the memory-pressure hotspot during `.fbuf` load.
- `taskgraph_generated.h` / `taskgraph.proto` / `taskgraph.pb.{cc,h}` — FlatBuffer and (legacy, unused) protobuf schemas for the task graph. Do not edit `taskgraph_generated.h` by hand; it's generated from the flexflow-side `.fbs` + pinned to flatbuffers v24.3.25.
- `dyn_net_sch.{h,cpp}` — dynamic network scheduler shared by reconfigurable topologies.
- `mixnet_topomanager.{h,cpp}` — **regional reconfiguration logic**: decides which pod pairs get optical bandwidth shifts in response to observed all-to-all demand. This is the paper's core algorithmic contribution on the htsim side.
- `datacenter/mixnet.{h,cpp}` — the hybrid optical-electrical topology model itself (circuit + packet plane, node↔pod mapping). Paired with `fat_tree_topology.{h,cpp}` which provides the packet plane.
- `datacenter/main_tcp_mixnet.{h,cpp}` — CLI entry point and topology wiring for `htsim_tcp_mixnet`.

Stock htsim pieces (kept from upstream): `eventlist`, `pipe`, `queue*`, `tcp*`, `ndp*`, `dctcp`, `compositequeue`, `ecnqueue`, `switch`, `route`, `logfile/loggers`, `network`. Don't reinvent flows/queues — extend these.

Each topology has its own `main_tcp_*.cpp` with its own `main()` — they all link the same `libhtsim.a` plus their topology files. When adding a new experiment, mirror `main_tcp_mixnet.cpp` rather than shoe-horning into an existing one.

## Running context on this specific host

This host has **30 GiB RAM, no GPU, no CUDA**. `REPRODUCE_PLAN.md` at the repo root is the live reproduce plan that accounts for these constraints — read it before running anything expensive. Key points it raises:

- FlexFlow cannot be built here (no `nvcc`, no GPU). Use the pre-generated `.fbuf` files; don't try to regenerate them.
- 1024-node Mixtral-8x22B mixnet sim peak RSS is close to the RAM limit. The plan recommends `systemd-run --scope --user -p MemoryMax=24G -p MemorySwapMax=16G ...` to avoid OOM-killing the box, plus running sweep configs strictly serially (strip the `&`). The `onestage` variant (512 nodes, pp=1) is the preferred smoke entry if a matching `.fbuf` is available.
- Use `/usr/bin/time -v` on runs to capture peak RSS — that's the signal for whether a configuration fits.

## Gotchas

- The submodules are pinned to **PeisongZhang forks**, not the `mixnet-project` org. Do not `git remote set-url` back to upstream without checking — the forks may carry fixes (e.g. the flatbuffers header vendoring) that upstream doesn't.
- `mixnet-flexflow/fbuf/include/` is not checked in; it's a build-time artifact you create from v24.3.25 of google/flatbuffers plus a copy of `taskgraph_generated.h`.
- The ns-3 / analytical ASTRA-sim workflow described in the parent `/home/ps/sow/part2/CLAUDE.md` is **a different project**. Don't cross the streams: ASTRA-sim consumes Chakra ET protobufs; htsim here consumes FlexFlow FlatBuffers. Nothing in `astra-sim/` feeds into `mixnet-sim/`.
- The sweep scripts write output files (`nwsim_ocs_<bw>.txt`, `nwsim_ecs_<bw>.txt`) to the simulator's current working directory, not to `-logdir`. Always `cd` to a per-run directory before invoking, or be prepared to find them next to the binary.
