# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""A resource pool can be pinned to nodes that declare a Ray custom resource.

Pinning lets trainer and rollout pools sit on different hardware in one Ray cluster. The tests run without a Ray
cluster: placement-group creation and the cluster resource check are stubbed out.
"""

import pytest

from verl.single_controller.ray import base
from verl.single_controller.ray.base import ResourcePoolManager
from verl.workers.config.rollout import RolloutConfig


class _FakePlatform:
    device_name = "cuda"

    def ray_resource_name(self):
        return "GPU"

    def supports_colocated_worker_groups(self):
        return True


@pytest.fixture
def stub_ray(monkeypatch):
    """Stub placement-group creation so RayResourcePool records its bundles without a cluster."""

    def _fake_placement_group(bundles, **kwargs):
        return type("PG", (), {"bundle_specs": bundles, "ready": lambda self: None})()

    monkeypatch.setattr(base, "get_platform", lambda: _FakePlatform())
    monkeypatch.setattr(base, "placement_group", _fake_placement_group)
    monkeypatch.setattr(base.ray, "get", lambda refs, **kw: None)
    monkeypatch.setattr(base, "sort_placement_group_by_node_ip", lambda pgs: pgs)
    monkeypatch.setattr(ResourcePoolManager, "_check_resource_available", lambda self: None)


def _all_bundles(manager, pool):
    return [bundle for pg in manager.resource_pool_dict[pool].get_placement_groups() for bundle in pg.bundle_specs]


def test_pool_without_key_keeps_default_bundles(stub_ray):
    manager = ResourcePoolManager(resource_pool_spec={"pool": [2]}, mapping={})
    manager.create_resource_pool()
    assert manager.resource_pool_dict["pool"].accelerator_type is None
    for bundle in _all_bundles(manager, "pool"):
        assert set(bundle) == {"CPU", "GPU"}


def test_pool_key_is_added_to_every_bundle(stub_ray):
    key = "accelerator_type:Intel-GPU-B60"
    manager = ResourcePoolManager(
        resource_pool_spec={"trainer": [2], "rollout": [2]},
        mapping={},
        pool_accelerator_resource_key={"rollout": key},
    )
    manager.create_resource_pool()

    assert manager.resource_pool_dict["rollout"].accelerator_type == key
    assert manager.resource_pool_dict["trainer"].accelerator_type is None
    assert all(key in bundle for bundle in _all_bundles(manager, "rollout"))
    assert all(key not in bundle for bundle in _all_bundles(manager, "trainer"))


def test_null_key_means_unpinned(stub_ray):
    manager = ResourcePoolManager(
        resource_pool_spec={"pool": [1]}, mapping={}, pool_accelerator_resource_key={"pool": None}
    )
    manager.create_resource_pool()
    assert manager.resource_pool_dict["pool"].accelerator_type is None


def test_key_for_unknown_pool_is_rejected():
    with pytest.raises(ValueError, match="unknown resource pools"):
        ResourcePoolManager(
            resource_pool_spec={"pool": [1]}, mapping={}, pool_accelerator_resource_key={"missing": "k"}
        )


@pytest.mark.parametrize("bad_key", ["", "   ", 7])
def test_blank_or_non_string_key_is_rejected(bad_key):
    with pytest.raises(ValueError, match="invalid accelerator resource key"):
        ResourcePoolManager(
            resource_pool_spec={"pool": [1]}, mapping={}, pool_accelerator_resource_key={"pool": bad_key}
        )


def test_rollout_config_exposes_key():
    assert RolloutConfig.__dataclass_fields__["accelerator_resource_key"].default is None
