#!/usr/bin/env python3
"""
Verify the ZE_AFFINITY_MASK fix for ONEAPI_DEVICE_SELECTOR bug.

Tests that:
1. get_visible_devices_keyword() returns ZE_AFFINITY_MASK on XPU
2. sanitize_xpu_device_selector() fixes bare-ID ONEAPI_DEVICE_SELECTOR
3. No SYCL crash when env vars are set the way Ray + verl sets them

Run: python tests/special_xpu/test_device_selector_fix.py
"""

import os
import sys
import subprocess


def test_keyword():
    """get_visible_devices_keyword should return ZE_AFFINITY_MASK on XPU."""
    from verl.utils.device import get_visible_devices_keyword, is_xpu_available
    if not is_xpu_available:
        print("SKIP: no XPU")
        return True
    kw = get_visible_devices_keyword()
    ok = kw == "ZE_AFFINITY_MASK"
    print(f"  get_visible_devices_keyword() = {kw!r} ... {'PASS' if ok else 'FAIL (expected ZE_AFFINITY_MASK)'}")
    return ok


def test_sanitize_bare_id():
    """sanitize_xpu_device_selector should fix bare IDs."""
    from verl.utils.device import sanitize_xpu_device_selector, is_xpu_available
    if not is_xpu_available:
        print("SKIP: no XPU")
        return True

    # Simulate Ray setting bare ID
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "0"
    os.environ.pop("ZE_AFFINITY_MASK", None)
    sanitize_xpu_device_selector()

    sel = os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
    mask = os.environ.get("ZE_AFFINITY_MASK", "")
    ok = sel == "level_zero:0" and mask == "0"
    print(f"  After sanitize: ONEAPI_DEVICE_SELECTOR={sel!r}, ZE_AFFINITY_MASK={mask!r} ... {'PASS' if ok else 'FAIL'}")

    # Cleanup
    os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)
    os.environ.pop("ZE_AFFINITY_MASK", None)
    return ok


def test_sanitize_already_correct():
    """sanitize should not touch correctly-formatted values."""
    from verl.utils.device import sanitize_xpu_device_selector, is_xpu_available
    if not is_xpu_available:
        print("SKIP: no XPU")
        return True

    os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
    sanitize_xpu_device_selector()
    sel = os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
    ok = sel == "level_zero:0"
    print(f"  Already correct: ONEAPI_DEVICE_SELECTOR={sel!r} ... {'PASS' if ok else 'FAIL'}")
    os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)
    return ok


def test_sanitize_multi_gpu():
    """sanitize should handle multi-GPU bare IDs like '0,1'."""
    from verl.utils.device import sanitize_xpu_device_selector, is_xpu_available
    if not is_xpu_available:
        print("SKIP: no XPU")
        return True

    os.environ["ONEAPI_DEVICE_SELECTOR"] = "0,1"
    os.environ.pop("ZE_AFFINITY_MASK", None)
    sanitize_xpu_device_selector()
    sel = os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
    mask = os.environ.get("ZE_AFFINITY_MASK", "")
    ok = sel == "level_zero:0,1" and mask == "0,1"
    print(f"  Multi-GPU: ONEAPI_DEVICE_SELECTOR={sel!r}, ZE_AFFINITY_MASK={mask!r} ... {'PASS' if ok else 'FAIL'}")
    os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)
    os.environ.pop("ZE_AFFINITY_MASK", None)
    return ok


def test_subprocess_no_crash():
    """Spawn a subprocess with ZE_AFFINITY_MASK=0 — should not SIGABRT."""
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("SKIP: no XPU")
        return True

    env = os.environ.copy()
    env["ZE_AFFINITY_MASK"] = "0"
    env.pop("ONEAPI_DEVICE_SELECTOR", None)

    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(f'XPU devices: {torch.xpu.device_count()}')"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    ok = result.returncode == 0 and "XPU devices:" in result.stdout
    print(f"  Subprocess with ZE_AFFINITY_MASK=0: rc={result.returncode} {result.stdout.strip()} ... {'PASS' if ok else 'FAIL'}")
    if not ok and result.stderr:
        print(f"    stderr: {result.stderr[:200]}")
    return ok


def test_subprocess_bare_selector_crashes():
    """Confirm that bare ONEAPI_DEVICE_SELECTOR=0 still crashes (proves bug exists)."""
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("SKIP: no XPU")
        return True

    env = os.environ.copy()
    env["ONEAPI_DEVICE_SELECTOR"] = "0"
    env.pop("ZE_AFFINITY_MASK", None)

    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(f'XPU devices: {torch.xpu.device_count()}')"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    ok = result.returncode != 0  # Should crash
    status = "PASS (confirmed crash)" if ok else "FAIL (did not crash?!)"
    print(f"  Bare ONEAPI_DEVICE_SELECTOR=0 subprocess: rc={result.returncode} ... {status}")
    return ok


def test_subprocess_sanitized_selector():
    """Confirm that level_zero:0 format works in subprocess."""
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("SKIP: no XPU")
        return True

    env = os.environ.copy()
    env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
    env.pop("ZE_AFFINITY_MASK", None)

    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(f'XPU devices: {torch.xpu.device_count()}')"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    ok = result.returncode == 0 and "XPU devices:" in result.stdout
    print(f"  ONEAPI_DEVICE_SELECTOR=level_zero:0: rc={result.returncode} {result.stdout.strip()} ... {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("ONEAPI_DEVICE_SELECTOR Fix Verification")
    print("=" * 60)

    tests = [
        ("1. get_visible_devices_keyword", test_keyword),
        ("2. sanitize bare ID", test_sanitize_bare_id),
        ("3. sanitize already correct", test_sanitize_already_correct),
        ("4. sanitize multi-GPU", test_sanitize_multi_gpu),
        ("5. subprocess ZE_AFFINITY_MASK=0", test_subprocess_no_crash),
        ("6. bare selector crashes (bug exists)", test_subprocess_bare_selector_crashes),
        ("7. sanitized selector works", test_subprocess_sanitized_selector),
    ]

    results = []
    for name, fn in tests:
        print(f"\n{name}:")
        try:
            results.append(fn())
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if all(results):
        print("ALL PASS — fix is working correctly")
    else:
        print("FAILURES — check above")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
