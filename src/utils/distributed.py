import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    return 1 if not is_dist_avail_and_initialized() else dist.get_world_size()

def get_rank():
    return 0 if not is_dist_avail_and_initialized() else dist.get_rank()

def is_main_process():
    return get_rank() == 0

