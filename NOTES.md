Train: data epoch: [0] Total time: 14:55:08 (0.1547 s / it)
2024-02-22 11:51:39,335 [INFO] Averaged stats: lr: 0.0001  loss: 1.8361
2024-02-22 11:51:39,338 [INFO] No validation splits found.
2024-02-22 11:51:39,357 [INFO] Saving checkpoint at epoch 0 to /extra/workspaces/ml/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20240221205/checkpoint_0.pth.
Traceback (most recent call last):
  File "/extra/workspaces/ml/LAVIS/train.py", line 103, in <module>
    main()
  File "/extra/workspaces/ml/LAVIS/train.py", line 99, in main
    runner.train()
  File "/extra/workspaces/ml/LAVIS/lavis/runners/runner_base.py", line 403, in train
    dist.barrier()
  File "/extra/workspaces/ml/LAVIS/venv/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
    return func(*args, **kwargs)
  File "/extra/workspaces/ml/LAVIS/venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 3685, in barrier
    opts.device = _get_pg_default_device(group)
  File "/extra/workspaces/ml/LAVIS/venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 593, in _get_pg_default_device
    group = group or _get_default_group()
  File "/extra/workspaces/ml/LAVIS/venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 940, in _get_default_group
    raise RuntimeError(
RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.