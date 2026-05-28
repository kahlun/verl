# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
XPU smoke test for verl library — comprehensive sanity checks.

Tests device availability, tensor ops, DataProto, advantage estimation,
policy gradient, and value function training on Intel XPU.
"""

import os
import subprocess
import tempfile
import unittest

import torch
import torch.nn as nn


class TestXPUSmoke(unittest.TestCase):
    """Comprehensive XPU smoke test for verl."""

    @classmethod
    def setUpClass(cls):
        """Disable torch.compile to avoid InductorError in CI (no Python.h)."""
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.disable = True

    def test_level1_device_sanity(self):
        """Level 1: Verify XPU is available and basic tensor ops work."""
        # Check XPU availability
        self.assertTrue(torch.xpu.is_available(), "XPU is not available")
        device_count = torch.xpu.device_count()
        self.assertGreater(device_count, 0, f"Expected >0 XPU devices, got {device_count}")

        # Test basic tensor operations on XPU
        device = torch.device("xpu:0")
        a = torch.randn(128, 256, device=device)
        b = torch.randn(256, 128, device=device)

        # Matmul
        c = torch.matmul(a, b)
        self.assertEqual(c.shape, (128, 128))
        self.assertFalse(torch.isnan(c).any(), "NaN detected in matmul output")
        self.assertFalse(torch.isinf(c).any(), "Inf detected in matmul output")

        # Softmax
        d = torch.softmax(a, dim=-1)
        self.assertAlmostEqual(d[0].sum().item(), 1.0, places=5, msg="Softmax doesn't sum to 1")
        self.assertFalse(torch.isnan(d).any(), "NaN in softmax")

        # Layer norm
        ln = nn.LayerNorm(256).to(device)
        e = ln(a)
        self.assertFalse(torch.isnan(e).any(), "NaN in LayerNorm")

    def test_level2_dataproto_construction(self):
        """Level 2: Construct DataProto and move to XPU."""
        try:
            from verl.protocol import DataProto
        except ImportError:
            self.skipTest("verl.protocol not available")

        device = torch.device("xpu:0")

        # Create synthetic tensors
        input_ids = torch.randint(0, 100, (4, 32), dtype=torch.long)
        attention_mask = torch.ones(4, 32, dtype=torch.long)

        # Construct DataProto
        from tensordict import TensorDict

        batch = TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=[4])

        data_proto = DataProto(batch=batch, non_tensor_batch={}, meta_info={})

        # Move to XPU
        batch_xpu = batch.to(device)
        self.assertEqual(batch_xpu["input_ids"].device.type, "xpu")
        self.assertEqual(batch_xpu["attention_mask"].device.type, "xpu")

        # Verify shape preservation
        self.assertEqual(batch_xpu["input_ids"].shape, (4, 32))

    def test_level3_model_forward(self):
        """Level 3: Build minimal model, move to XPU, run forward pass."""
        device = torch.device("xpu:0")

        # Build a minimal Transformer decoder layer (not a full LLM)
        d_model = 64
        nhead = 4
        num_layers = 2

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        model = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(device)

        # Synthetic input
        batch_size = 4
        seq_len = 16
        tgt = torch.randn(batch_size, seq_len, d_model, device=device)
        memory = torch.randn(batch_size, seq_len, d_model, device=device)

        # Forward pass
        output = model(tgt, memory)

        # Verify output
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertFalse(torch.isnan(output).any(), "NaN in model output")
        self.assertFalse(torch.isinf(output).any(), "Inf in model output")

    def test_level4_training_step(self):
        """Level 4: Complete training step with forward, loss, backward, optimizer step."""
        device = torch.device("xpu:0")

        # Build small model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic data
        inputs = torch.randn(8, 64, device=device)
        targets = torch.randint(0, 10, (8,), device=device)

        # Store initial parameters
        params_before = [p.detach().clone() for p in model.parameters()]

        # Forward
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)

        # Verify loss
        self.assertGreater(loss.item(), 0, "Loss should be positive")
        self.assertFalse(torch.isnan(loss), f"Loss is NaN: {loss.item()}")
        self.assertFalse(torch.isinf(loss), f"Loss is Inf: {loss.item()}")

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Verify gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient in {name}")

        # Optimizer step
        optimizer.step()

        # Verify parameters actually changed
        params_after = [p.detach().clone() for p in model.parameters()]
        for i, (p_before, p_after) in enumerate(zip(params_before, params_after)):
            self.assertFalse(torch.equal(p_before, p_after), f"Parameter {i} did not update after optimizer step")

    def test_level5_gae_advantage_computation(self):
        """Level 5 (RL): Test GAE advantage estimation on XPU."""
        try:
            from verl.trainer.ppo.core_algos import compute_gae_advantage_return
        except ImportError:
            self.skipTest("verl.trainer.ppo.core_algos not available")

        device = torch.device("xpu:0")

        # Synthetic RL data
        batch_size = 4
        seq_len = 16
        token_level_rewards = torch.randn(batch_size, seq_len, device=device) * 0.1
        values = torch.randn(batch_size, seq_len, device=device)
        response_mask = torch.ones(batch_size, seq_len, device=device)
        # Mask out last 4 tokens for batch 0 to test masking
        response_mask[0, -4:] = 0

        gamma = torch.tensor(0.99, device=device)
        lam = torch.tensor(0.95, device=device)

        # Compute GAE
        advantages, returns = compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam)

        # Assertions
        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))
        self.assertFalse(torch.isnan(advantages).any(), "NaN in advantages")
        self.assertFalse(torch.isnan(returns).any(), "NaN in returns")

        # Verify masked whitening: mean of UNMASKED advantages should be ~0
        # (Note: masked_whiten whitens only over valid positions)
        masked_adv_sum = (advantages * response_mask).sum()
        masked_adv_count = response_mask.sum()
        masked_adv_mean = masked_adv_sum / masked_adv_count
        self.assertAlmostEqual(masked_adv_mean.item(), 0.0, delta=0.1, msg="Advantages not whitened")

    def test_level5_grpo_advantage_computation(self):
        """Level 5 (RL): Test GRPO outcome advantage on XPU."""
        try:
            from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
        except ImportError:
            self.skipTest("verl.trainer.ppo.core_algos not available")

        import numpy as np

        device = torch.device("xpu:0")

        # Synthetic GRPO data: 4 samples, 2 groups
        batch_size = 4
        seq_len = 8
        token_level_rewards = torch.randn(batch_size, seq_len, device=device) * 0.1
        response_mask = torch.ones(batch_size, seq_len, device=device)

        # Group index: samples 0,1 in group A; samples 2,3 in group B
        index = np.array([0, 0, 1, 1], dtype=np.int64)

        # Compute GRPO advantage
        advantages, returns = compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)

        # Assertions
        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))
        self.assertFalse(torch.isnan(advantages).any(), "NaN in GRPO advantages")
        self.assertFalse(torch.isnan(returns).any(), "NaN in GRPO returns")

    def test_level5_policy_gradient_with_correct_old_log_probs(self):
        """Level 5 (RL): Test PPO policy gradient with proper old_log_probs (avoid zero gradient pitfall)."""
        device = torch.device("xpu:0")

        # Build tiny actor model (vocabulary-output model)
        vocab_size = 128
        d_model = 64
        seq_len = 8
        batch_size = 4

        model = nn.Sequential(nn.Linear(d_model, vocab_size)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate input embeddings and synthetic actions
        input_embeds = torch.randn(batch_size, seq_len, d_model, device=device)
        actions = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Compute old_log_probs CORRECTLY (from model's own output, NOT random noise)
        with torch.no_grad():
            old_logits = model(input_embeds)
            old_log_probs_full = old_logits.log_softmax(-1)
            old_log_probs = old_log_probs_full.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Add small noise to old_log_probs to simulate policy divergence
        old_log_probs = old_log_probs.detach() + torch.randn_like(old_log_probs) * 0.01

        # Compute new log_probs from current policy
        new_logits = model(input_embeds)
        new_log_probs = new_logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Synthetic advantages
        advantages = torch.randn(batch_size, seq_len, device=device)
        response_mask = torch.ones(batch_size, seq_len, device=device)

        # PPO clip loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        epsilon = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        loss1 = -advantages * ratio
        loss2 = -advantages * clipped_ratio
        policy_loss = torch.max(loss1, loss2)
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()

        # Verify loss is finite and non-zero
        self.assertFalse(torch.isnan(policy_loss), f"Policy loss is NaN: {policy_loss.item()}")
        self.assertFalse(torch.isinf(policy_loss), f"Policy loss is Inf: {policy_loss.item()}")
        self.assertNotAlmostEqual(policy_loss.item(), 0.0, places=6, msg="Policy loss is zero — no gradient flow")

        # Backward + verify gradients
        optimizer.zero_grad()
        policy_loss.backward()

        has_nonzero_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
                if grad_norm > 1e-8:
                    has_nonzero_grad = True

        self.assertTrue(has_nonzero_grad, "All gradients are near-zero — policy gradient failed")

    def test_level5_value_function_training(self):
        """Level 5 (RL): Test value function / critic loss separately."""
        device = torch.device("xpu:0")

        # Build value network
        d_model = 64
        seq_len = 8
        batch_size = 4

        value_net = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
        optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

        # Synthetic state embeddings
        states = torch.randn(batch_size, seq_len, d_model, device=device)

        # Compute predicted values
        pred_values = value_net(states).squeeze(-1)  # (batch_size, seq_len)

        # Synthetic target returns
        target_returns = torch.randn(batch_size, seq_len, device=device)
        response_mask = torch.ones(batch_size, seq_len, device=device)

        # MSE value loss
        value_loss = ((pred_values - target_returns) ** 2 * response_mask).sum() / response_mask.sum()

        # Verify loss
        self.assertGreater(value_loss.item(), 0, "Value loss should be positive")
        self.assertFalse(torch.isnan(value_loss), f"Value loss is NaN: {value_loss.item()}")

        # Backward
        optimizer.zero_grad()
        value_loss.backward()

        # Verify gradients
        for name, param in value_net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name} in value network")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_level5_mini_rl_rollout_loop(self):
        """Level 5 (RL): Mini rollout → advantage → actor update → critic update loop."""
        device = torch.device("xpu:0")

        # Small actor-critic
        d_model = 32
        vocab_size = 64
        seq_len = 8
        batch_size = 2

        actor = nn.Sequential(nn.Linear(d_model, vocab_size)).to(device)
        critic = nn.Sequential(nn.Linear(d_model, 1)).to(device)

        actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

        try:
            from verl.trainer.ppo.core_algos import compute_gae_advantage_return
        except ImportError:
            self.skipTest("verl.trainer.ppo.core_algos not available for mini rollout test")

        # Mini rollout loop (3 steps)
        for step in range(3):
            # Generate synthetic rollout
            input_embeds = torch.randn(batch_size, seq_len, d_model, device=device)
            actions = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            # Critic forward
            with torch.no_grad():
                values = critic(input_embeds).squeeze(-1)
                rewards = torch.randn(batch_size, seq_len, device=device) * 0.1
                response_mask = torch.ones(batch_size, seq_len, device=device)

                gamma = torch.tensor(0.99, device=device)
                lam = torch.tensor(0.95, device=device)

                advantages, returns = compute_gae_advantage_return(rewards, values, response_mask, gamma, lam)

            # Actor forward (with old_log_probs from previous policy)
            with torch.no_grad():
                old_logits = actor(input_embeds)
                old_log_probs = old_logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            # Actor update
            new_logits = actor(input_embeds)
            new_log_probs = new_logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            epsilon = 0.2
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            loss1 = -advantages.detach() * ratio
            loss2 = -advantages.detach() * clipped_ratio
            actor_loss = torch.max(loss1, loss2)
            actor_loss = (actor_loss * response_mask).sum() / response_mask.sum()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Critic update
            pred_values = critic(input_embeds).squeeze(-1)
            critic_loss = ((pred_values - returns.detach()) ** 2 * response_mask).sum() / response_mask.sum()

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # Verify both losses are finite
            self.assertFalse(torch.isnan(actor_loss), f"Actor loss NaN at step {step}")
            self.assertFalse(torch.isnan(critic_loss), f"Critic loss NaN at step {step}")

    def test_level6_distributed_xccl(self):
        """Level 6 (Distributed): Launch 2-GPU subprocess with xccl backend, test all_reduce."""
        device_count = torch.xpu.device_count()
        if device_count < 2:
            self.skipTest("Distributed test requires at least 2 XPU devices")

        # Write subprocess script
        script_content = """
import os
import sys
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize xccl backend
    dist.init_process_group(backend='xccl', init_method='env://')
    torch.xpu.set_device(local_rank)

    device = torch.device(f'xpu:{local_rank}')
    tensor = torch.ones(4, device=device) * (rank + 1)

    # All-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Expected: sum of (1, 2) = 3 for each element
    expected = torch.tensor([3.0] * 4, device=device)
    assert torch.allclose(tensor, expected, atol=1e-5), f"Rank {rank}: expected {expected}, got {tensor}"

    dist.destroy_process_group()
    print(f"Rank {rank}: all_reduce passed")

if __name__ == '__main__':
    main()
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Launch with torchrun
            result = subprocess.run(
                [
                    "torchrun",
                    "--nproc_per_node=2",
                    "--master_port=29501",
                    script_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Check result
            if result.returncode != 0:
                self.fail(f"Distributed test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            # Verify both ranks printed success
            self.assertIn("Rank 0: all_reduce passed", result.stdout)
            self.assertIn("Rank 1: all_reduce passed", result.stdout)

        finally:
            os.unlink(script_path)

    def test_level7_checkpoint_save_load(self):
        """Level 7 (Checkpoint): Save and load model state."""
        device = torch.device("xpu:0")

        # Build model
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ).to(device)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pt")

            # Save
            state_dict = model.state_dict()
            torch.save(state_dict, ckpt_path)

            # Create new model and load
            model2 = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            ).to(device)

            loaded_state = torch.load(ckpt_path, map_location=device)
            model2.load_state_dict(loaded_state)

            # Verify parameters match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.equal(p1, p2), f"Parameter {n1} mismatch after load")


if __name__ == "__main__":
    unittest.main()
