"""Wrapper: apply XPU patches, then run verl.trainer.sft_trainer."""
import tests.xpu_torchtitan_patch  # noqa: F401 – must import before anything else

from verl.trainer.sft_trainer import main

if __name__ == "__main__":
    main()
