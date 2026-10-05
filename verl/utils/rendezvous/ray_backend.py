# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import socket
import time

import ray
import torch.distributed as dist
from ray.util import list_named_actors

from verl.plugin.platform import get_platform


@ray.remote
class RendezvousHandleStore:
    """Ray actor publishing a rendezvous handle (an opaque value agreed on out-of-band)
    from rank 0 to every other rank. Named NCCLIDStore in older callers; kept as an
    alias below since the handle used to always be a cupy/NCCL unique-id blob."""

    def __init__(self, handle):
        self._handle = handle

    def get(self):
        return self._handle


# Deprecated alias: this actor class used to be NCCL-specific. The handle it carries
# is now either a cupy/NCCL unique-id blob or a (host, port) pair, depending on which
# path create_collective_communicator_in_ray() takes.
NCCLIDStore = RendezvousHandleStore


def get_rendezvous_handle_store_by_name(name):
    all_actors = list_named_actors(all_namespaces=True)
    matched_actors = [actor for actor in all_actors if actor.get("name", None) == name]
    if len(matched_actors) == 1:
        actor = matched_actors[0]
        return ray.get_actor(**actor)
    elif len(matched_actors) > 1:
        logging.warning("multiple actors with same name found: %s", matched_actors)
    elif len(matched_actors) == 0:
        logging.info("failed to get any actor named %s", name)
    return None


# Deprecated alias, same rename reason as RendezvousHandleStore above.
get_nccl_id_store_by_name = get_rendezvous_handle_store_by_name


def _create_cupy_communicator_in_ray(
    collective, rank: int, world_size: int, group_name: str, max_retries: int, interval_s: int
):
    """cupy.cuda.nccl-shaped path: collective.get_unique_id() / collective.NcclCommunicator().
    Used when get_collective_module() is non-None (currently only PlatformCUDA)."""
    NcclCommunicator = collective.NcclCommunicator
    get_unique_id = collective.get_unique_id

    if rank == 0:
        nccl_id = get_unique_id()
        handle_store = RendezvousHandleStore.options(name=group_name).remote(nccl_id)

        assert ray.get(handle_store.get.remote()) == nccl_id
        return NcclCommunicator(ndev=world_size, commId=nccl_id, rank=0)
    else:
        for i in range(max_retries):
            handle_store = get_rendezvous_handle_store_by_name(group_name)
            if handle_store is not None:
                logging.info("nccl_id_store %s got", group_name)
                nccl_id = ray.get(handle_store.get.remote())
                logging.info("nccl id for %s got: %s", group_name, nccl_id)
                return NcclCommunicator(ndev=world_size, commId=nccl_id, rank=rank)
            logging.info("failed to get nccl_id for %d time, sleep for %d seconds", i + 1, interval_s)
            time.sleep(interval_s)
        raise RuntimeError(f"timed out waiting for rendezvous handle '{group_name}'")


class _TorchDistributedCommunicator:
    """rank_id()-compatible wrapper around the process group init_process_group()
    installs as the default group, for platforms with no cupy/NCCL-shaped module."""

    def __init__(self, rank: int):
        self._rank = rank

    def rank_id(self) -> int:
        return self._rank

    @property
    def _pg(self):
        return dist.group.WORLD


def _create_torch_distributed_communicator_in_ray(
    rank: int, world_size: int, group_name: str, max_retries: int, interval_s: int
):
    """Fallback path for platforms with no cupy/NCCL-shaped collective module (e.g.
    PlatformXPU). Bootstraps a torch.distributed process group instead of constructing
    a device-only process group directly: rank 0 publishes a plain (host, port) pair
    through the same named-actor rendezvous, then every rank calls
    init_process_group() with a hybrid "cpu:gloo,<device>:<backend>" backend string.

    The hybrid backend is required, not cosmetic -- a pure XCCL process group hits a
    known Level-Zero event-completion bug on 2-card Battlemage (PTF1-99); pairing it
    with a gloo CPU group for bootstrap/control-plane traffic avoids that path.
    """
    platform = get_platform()
    device = platform.device_name
    backend = f"cpu:gloo,{device}:{platform.communication_backend_name()}"

    if rank == 0:
        host = ray.util.get_node_ip_address()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
        handle = (host, port)
        handle_store = RendezvousHandleStore.options(name=group_name).remote(handle)
        assert ray.get(handle_store.get.remote()) == handle
    else:
        handle_store = None
        for i in range(max_retries):
            handle_store = get_rendezvous_handle_store_by_name(group_name)
            if handle_store is not None:
                logging.info("rendezvous handle for %s got", group_name)
                break
            logging.info("failed to get rendezvous handle for %d time, sleep for %d seconds", i + 1, interval_s)
            time.sleep(interval_s)
        else:
            raise RuntimeError(f"timed out waiting for rendezvous handle '{group_name}'")
        handle = ray.get(handle_store.get.remote())

    host, port = handle
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{host}:{port}",
        rank=rank,
        world_size=world_size,
    )
    return _TorchDistributedCommunicator(rank)


def create_collective_communicator_in_ray(
    rank: int, world_size: int, group_name: str, max_retries: int = 100, interval_s: int = 5
):
    """Build a cross-actor collective communicator via Ray rendezvous.

    Uses the platform's cupy/NCCL-shaped collective module when get_collective_module()
    returns one (currently only PlatformCUDA). Otherwise falls back to a plain
    torch.distributed process group -- get_collective_module() returning None is a
    valid, supported platform response, not a gap every platform must fill.
    """
    collective = get_platform().get_collective_module()
    if collective is not None:
        return _create_cupy_communicator_in_ray(collective, rank, world_size, group_name, max_retries, interval_s)
    return _create_torch_distributed_communicator_in_ray(rank, world_size, group_name, max_retries, interval_s)


def create_nccl_communicator_in_ray(
    rank: int, world_size: int, group_name: str, max_retries: int = 100, interval_s: int = 5
):
    """Deprecated alias for create_collective_communicator_in_ray(). The old name
    implied a cupy/NCCL-only contract; kept for existing callers."""
    return create_collective_communicator_in_ray(rank, world_size, group_name, max_retries, interval_s)
