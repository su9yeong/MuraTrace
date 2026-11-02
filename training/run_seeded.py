# -*- coding: utf-8 -*-
"""Seed-fixed launcher for ANY training entry point (reproducible runs).

Sets every RNG seed (default 7 = project generation seed; override with env SEED)
*before* the target module's main() runs, so model init, batch shuffle, dropout,
and augmentation are deterministic. Originals are untouched; this is the one new
file to manage for reproducibility.

Usage (same args as the original script, just prepend the module name):
  python run_seeded.py train_rca_advanced --data_root ... --out ... [--mode ...]
  python run_seeded.py train_segmentation_advanced --data_root ... --out ...
  SEED=7 python run_seeded.py train_rca_advanced ...        # explicit seed
"""
import importlib
import os
import random
import sys

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python run_seeded.py <module_name> [script args...]")
    seed = int(os.environ.get("SEED", "7"))
    set_all_seeds(seed)
    module_name = sys.argv[1]
    # hand the remaining args to the target script's own argparse
    sys.argv = [module_name] + sys.argv[2:]
    print(f"[run_seeded] SEED={seed} -> launching {module_name}.main()", flush=True)
    mod = importlib.import_module(module_name)
    mod.main()


if __name__ == "__main__":
    main()
