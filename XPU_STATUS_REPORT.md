# verl XPU Enablement — Status Update
2026-07-28 · Intel B70 (2× GPU)

---

## Context 
verl is an open-source LLM post-training framework (RLHF, GRPO, PPO).
We are enabling it to run on Intel XPU hardware and contributing fixes back upstream.

---

## What We Accomplished

All 4 major training algorithms now pass end-to-end on 2× Intel GPU:

- ✅ GRPO — pass
- ✅ PPO — pass
- ✅ SFT — pass
- ✅ GRPO colocated reward model — pass

**Key technical wins:**

- Fixed device isolation crash between Ray and vLLM at rollout startup
- Implemented graceful shared-memory fallback when GPU IPC is unavailable
- Resolved false out-of-memory error triggered by colocated training actors
- 2 verl upstream PRs submitted (#7098, ray#64440)
- 2 verl upstream PRs merged (#7128, #7184)
---

## Blockers

**1. PyTorch 2.12 workarounds**
5 temporary patches in verl core to compensate for PT 2.12 API gaps due to waiting VLLM official update.
→ All drop automatically once we upgrade to **PyTorch 2.13 + oneCCL 2022.0**.

**2. SGLang rollout — not yet colocated**
SGLang cannot run alongside FSDP training actors yet.
→ Need to add shared-memory fallback (same pattern vLLM already uses) or support IPC for SGLang rollout.

---

## Next Steps

| Priority | Action |
|---|---|
| 🔴 This week | Get PR #7184, ray#64440 merged upstream |
| 🟡 Next step | Upgrade to PyTorch 2.13 |
| 🟡 Next step | Remove workaround of PyTorch 2.12 |
| 🟡 Backlog | SGLang colocated support |
| 🟡 Backlog | torchtitan support |
| 🟡 Backlog | VeOmni support |
---