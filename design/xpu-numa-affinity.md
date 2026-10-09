# verl on Intel XPU: NUMA affinity is silently a no-op

Proposal + feasibility evidence for closing the pynvml → XPU gap in
`set_numa_affinity()`. Branch `fix/xpu-numa-affinity` (not started — this is
the design only). 2026-09-23.

---

## The one-line version

`verl/utils/distributed.py::set_numa_affinity()` pins each worker to the CPU
cores local to its GPU using `pynvml.nvmlDeviceSetCpuAffinity()`. On XPU,
`pynvml` isn't installed, so the call raises `ImportError`, which is already
caught — the function prints a warning and returns. No crash, no error in the
logs anyone would notice. Every XPU training run today is simply never
NUMA-pinned.

This is the same *shape* of bug as the Ray device-pinning issue
([`fix/ray-device-pinning`](../verl-ray-device-wt/design/ray-device-pinning.md)):
nothing fails loudly, the job runs, and the only cost is silently worse
performance (here: cross-socket memory traffic between a worker and its GPU,
instead of same-socket).

---

## Where this actually gets called

```
verl/workers/engine_workers.py:92        set_numa_affinity()   # every training worker
verl/model_merger/megatron_model_merger.py:154   set_numa_affinity()   # checkpoint merge
```

Both call sites are unconditional — not gated behind a CUDA check — so this
runs on every XPU job today and silently no-ops both times.

There's a second, unrelated `pynvml` usage in
`verl/workers/rollout/trtllm_rollout/trtllm_rollout.py` (`nvmlDeviceGetUUID`
for GPU identification). That one doesn't matter for XPU: TRT-LLM rollout is
CUDA-only today (see the WW38 UT failure breakdown on the VeRL Confluence
page — 5 of the 19 real failures are exactly this path assuming `'cuda'`).
Out of scope here.

---

## Why "pynvml equivalent" is a real, separate, already-answered ask

Internal ticket [PYTORCHDGQ-5603](https://jira.devtools.intel.com/browse/PYTORCHDGQ-5603)
("need a library like pynvml for XPU") was filed by Matrix Yao and its
assignee, Lee Siaw Chen, commented directly that pynvml is a dependency for
verl enablement. The ticket's answer (Guangye Yu, 2026-08-13): PyTorch XPU
ships **`pyzes`** (Level Zero Sysman Python bindings) since **torch 2.12** —
confirmed by reading `torch/xpu/__init__.py`, which wraps `pyzes` for
`temperature()`, `clock_rate()`, `power_draw()`, `utilization()`,
`memory_usage()`, `device_memory_used()`.

That covers the *telemetry* half of pynvml. It does **not** cover
`nvmlDeviceSetCpuAffinity` — there is no CPU-affinity function anywhere in
`torch.xpu`. Ting Ye asked the team to confirm `pyzes` satisfies the need;
no confirmation was ever posted back to the ticket, and it's marked Closed
anyway. This doc is that confirmation, for the affinity half specifically:
partial. Telemetry, yes. Affinity, no — but buildable, per below.

### The gap in plain terms (for management / non-code review)

pynvml has always bundled two unrelated things under one library:

1. **Telemetry** — temperature, power, clock speed, utilization, memory used.
2. **System integration** — `nvmlDeviceSetCpuAffinity`: pin *this process* to
   the CPU cores physically closest to *this GPU*.

PyTorch's XPU team built (1). They never built (2). That's the entire gap —
not a bug, not regressed functionality, just a convenience wrapper that's
half-finished. Impact: not a correctness problem (training runs and produces
correct results either way) — it's a **silent performance tax**. Every XPU
worker should run on the CPU cores local to its own GPU; without (2), it
might run on cores attached to a *different* CPU socket than its GPU,
adding cross-socket memory latency to every data-loading / host↔device
transfer for the life of the job. CUDA deployments get this for free
today via pynvml; XPU quietly does not. No throughput number is measured
yet (see "Still open" below) — what's established is that the mechanism is
simply absent, confirmed on real hardware, not assumed.

### Is this a stale-pyzes / needs-a-version-bump problem? No.

Checked directly rather than assumed, on 2026-09-23:

- **pyzes is actively maintained**, not abandoned. It lives in
  `oneapi-src/level-zero` (`bindings/sysman/python/`, Intel's own repo, not a
  third-party wrapper). Commit history on that path: a real feature PR
  (`#462`, "Add more APIs to L0 Sysman python binding") merged **2026-06-11**,
  released as pyzes **0.1.2** the next day, and a fix/CI commit as recently
  as **2026-08-05** — about seven weeks before this doc, not years.
- **`zesDevicePciGetProperties` — the one function this whole design depends
  on — was itself only added in that same June 2026 PR.** Before that release,
  this approach wasn't buildable via pyzes at all. So if anything, the
  timeline is "just became possible," not "regressed."
- **Checked PyTorch's actual current `main` branch** (not just the older
  commit the Jira thread referenced) for `torch/xpu/__init__.py`: still zero
  affinity/NUMA functions today. So pulling a newer PyTorch would not hand you
  this feature — it genuinely doesn't exist upstream anywhere yet.
- **Checked pyzes's own source for any affinity-shaped API** (grepped the
  full `pyzes.py`): nothing. The only hit is an unrelated enum constant
  (`ZES_DEVICE_TYPE_CPU`, a device-type tag, not a function).
- **Why it's probably not coming for free**: `nvmlDeviceSetCpuAffinity` is a
  bit of an outlier even inside NVML — it's an OS-process-scheduling
  operation dressed up as a "device" API, not a GPU telemetry/management
  primitive. Level Zero Sysman's whole scope is device state (power,
  frequency, PCI, ECC, memory, engine utilization) — process CPU affinity
  doesn't obviously belong there, which is likely *why* nobody's added it
  yet, not because it's queued and waiting. Comparing to pynvml's own release
  cadence (mature, releases going back to 2019, latest 13.0.1 in 2025) vs.
  pyzes (pre-1.0, first release 2026-02, three releases total) — pyzes is
  young, but young ≠ stalled, and this specific gap looks architectural, not
  a version lag either side will close on its own. That's the actual reason
  this needs a plugin-side patch rather than "wait for the next release."

---

## Feasibility: can it be built from pyzes anyway?

Yes. `pyzes` wraps `zesDevicePciGetProperties(handle, &props)`, which fills a
`zes_pci_properties_t.address` struct (`domain`, `bus`, `device`,
`function`) — the same PCI BDF that NVML's own `nvmlDeviceSetCpuAffinity`
uses internally on Linux (PCI bus id → sysfs `numa_node` → sysfs `cpulist` →
`sched_setaffinity`). No missing primitive; it's a mechanical port of what
NVML already does.

### Evidence: verified on real hardware, 2 nodes × 2 GPUs

Ran a standalone probe (`hsdp-avg-probe/pyzes_numa_probe.py`) via
`devctl test --replicas=2 --gpu=2 --gpu-model=B60`, which is Kueue-admitted as
a JobSet — genuinely two separate physical nodes, not two pods on one box.
Image `intel/deep-learning-essentials:2026.1.0-devel-ubuntu24.04` +
`pip install pyzes` (that image ships no `pip` by default; needs
`apt-get install python3-pip` first).

```
rank 0 — gnr17908.jf.intel.com (256 logical CPUs)
  gpu[0] bdf=0000:5f:00.0  numa_node=1  cpulist=32-63,160-191
  gpu[1] bdf=0000:70:00.0  numa_node=1  cpulist=32-63,160-191
  sched_setaffinity: OK

rank 1 — gnr818729.jf.intel.com (512 logical CPUs)
  gpu[0] bdf=0000:3d:00.0  numa_node=0  cpulist=0-127,256-383
  gpu[1] bdf=0000:ba:00.0  numa_node=1  cpulist=128-255,384-511
  sched_setaffinity: OK
```

Every step worked on both nodes: `zesInit` → `zesDriverGet` → `zesDeviceGet`
enumerated exactly the 2 GPUs the cgroup allocated (not the host's full GPU
count), `zesDevicePciGetProperties` returned distinct real BDFs,
`/sys/bus/pci/devices/<bdf>/numa_node` and
`/sys/devices/system/node/nodeN/cpulist` resolved to sane values, and
`os.sched_setaffinity` succeeded with no container permission issues.

**Rank 1 is the important case**: its two GPUs sit on *two different NUMA
nodes within the same pod* (gpu0 → numa0, gpu1 → numa1). A naive
implementation that pins "the pod" to "a" local NUMA node would be wrong on
this exact box. The affinity call has to be keyed per device index — which
is what `nvmlDeviceGetHandleByIndex(local_rank)` already does today, and
what the design below preserves.

---

## Proposed design (same two-layer split as the Ray-pinning fix)

Per the maintainer's ruling on plugin issue #26 (default to plugin-side
override, core PR only for genuinely hardware-agnostic blockers):

### Layer 1 — core (hardware-agnostic gap)

`verl/utils/distributed.py::set_numa_affinity()` hardcodes pynvml with no
platform hook — same shape of gap as `worker.py`'s hardcoded device pinning
was before the Ray-pinning fix. Proposed change:

```python
def set_numa_affinity():
    if is_npu_available:
        return
    local_rank = _resolve_local_rank()
    get_platform().set_numa_affinity(local_rank)
```

Default/base implementation keeps today's pynvml code path verbatim, so CUDA
behavior is unchanged by construction. Non-CUDA, non-XPU platforms without an
override just inherit the existing no-op-with-warning behavior — no
regression for anyone.

### Layer 2 — plugin (`platform_xpu.py`)

```python
def set_numa_affinity(self, local_rank: int) -> None:
    handle = _zes_device_handle(local_rank)   # zesInit/zesDriverGet/zesDeviceGet
    props = zes_pci_properties_t()
    zesDevicePciGetProperties(handle, byref(props))
    bdf = _format_bdf(props.address)          # domain:bus:device.function

    numa_node = _read_int(f"/sys/bus/pci/devices/{bdf}/numa_node")
    if numa_node is None or numa_node < 0:
        return  # no NUMA topology on this box; nothing to pin to

    cpulist = _read_text(f"/sys/devices/system/node/node{numa_node}/cpulist")
    os.sched_setaffinity(0, _parse_cpulist(cpulist))
```

All four steps are exactly what the probe above already exercised on real
hardware, on two different node topologies.

---

## Still open

- [ ] Write the actual `_zes_device_handle` / `_format_bdf` / `_parse_cpulist`
      helpers as real plugin code (probe above is throwaway, not shippable).
- [ ] Decide whether `zesInit`/`zesDriverGet`/`zesDeviceGet` should be called
      fresh each time or cached — NVML keeps a long-lived handle;
      `torch.xpu`'s own zes wrapper caches per-device info
      (`_cached_zes_device_infos`) for the same reason. Reuse that pattern.
    - Confirm `ZES_ENABLE_SYSMAN=1` is actually required on the cluster's
      driver version, or if it's a no-op safety net (probe set it
      defensively; never hit a failure either way).
- [ ] Unit test coverage mirroring `tests/plugin/test_platform_abstraction.py`
      from the Ray-pinning fix (unmasked, no-NUMA box, multi-NUMA-per-pod
      case — rank 1 above is a real fixture for that last one).
- [ ] Post the pyzes-covers-telemetry-not-affinity finding back onto
      PYTORCHDGQ-5603 — it's marked Closed with an unanswered
      "please verify" from Ting Ye.
- [ ] Re-run the probe with `--gpu=4` / more replicas to confirm the
      per-device NUMA split generalizes past 2 GPUs/pod (untested).
- [ ] Check whether `engine_workers.py:92` and
      `megatron_model_merger.py:154` need the call to happen before or after
      `set_device()` — NVML's affinity call is independent of which device is
      "current", but worth confirming the XPU zes calls are too.

## Not this bug

`nvmlDeviceGetUUID` in `trtllm_rollout.py` is a separate, unrelated pynvml
usage gated to the CUDA-only TRT-LLM rollout path. Not touched by this
proposal.

## References

- Feasibility probe: `hsdp-avg-probe/pyzes_numa_probe.py`,
  `hsdp-avg-probe/run_pyzes_probe.sh`
- Internal ticket: PYTORCHDGQ-5603 ("need a library like pynvml for XPU")
- Prior art / same pattern: `fork/verl-ray-device-wt/design/ray-device-pinning.md`
- `torch/xpu/__init__.py` (pyzes-backed telemetry functions, `_zes_ensure_device_infos` caching pattern)
- `pyzes` 0.1.2 on PyPI — Level Zero Sysman Python bindings
