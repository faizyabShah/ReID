/home/sidna/ReID_Project/ReID/processor/part_attention_vit_processor.py:52: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = amp.GradScaler(init_scale=512)
/home/sidna/ReID_Project/ReID/processor/part_attention_vit_processor.py:99: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with amp.autocast(enabled=True):
/home/sidna/ReID_Project/ReID/loss/myloss.py:29: UserWarning: This overload of addmm_ is deprecated:
	addmm_(Number beta, Number alpha, Tensor mat1, Tensor mat2)
Consider using one of the following signatures instead:
	addmm_(Tensor mat1, Tensor mat2, *, Number beta = 1, Number alpha = 1) (Triggered internally at /pytorch/torch/csrc/utils/python_arg_parser.cpp:1661.)
  dist_map.addmm_(1, -2, part_feat, part_centers.t())
slurmstepd-neumann: error: *** JOB 145487 ON neumann CANCELLED AT 2026-04-30T08:52:01 DUE TO TIME LIMIT ***
