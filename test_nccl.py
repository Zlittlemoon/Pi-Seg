import os
import time
import socket
import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(
        f"[init] host={socket.gethostname()} "
        f"rank={rank}/{world_size} local_rank={local_rank} "
        f"cuda_device={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )

    dist.init_process_group(backend="nccl", init_method="env://")

    # GPU compute test
    x = torch.randn(2048, 2048, device=device)
    y = x @ x
    torch.cuda.synchronize()

    print(
        f"[compute ok] rank={rank} y_mean={y.mean().item():.6f}",
        flush=True,
    )

    # all_reduce test
    t = torch.ones(1, device=device) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = world_size * (world_size + 1) / 2
    print(
        f"[all_reduce] rank={rank} value={t.item()} expected={expected}",
        flush=True,
    )

    assert abs(t.item() - expected) < 1e-4, (
        f"rank={rank} all_reduce failed: got {t.item()}, expected {expected}"
    )

    # barrier test
    print(f"[barrier before] rank={rank}", flush=True)
    dist.barrier()
    print(f"[barrier after] rank={rank}", flush=True)

    dist.destroy_process_group()
    print(f"[done] rank={rank}", flush=True)


if __name__ == "__main__":
    main()