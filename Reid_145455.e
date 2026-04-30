/home/sidna/ReID_Project/ReID/processor/part_attention_vit_processor.py:52: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = amp.GradScaler(init_scale=512)
/home/sidna/ReID_Project/ReID/processor/part_attention_vit_processor.py:99: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with amp.autocast(enabled=True):
/home/sidna/ReID_Project/ReID/loss/myloss.py:29: UserWarning: This overload of addmm_ is deprecated:
	addmm_(Number beta, Number alpha, Tensor mat1, Tensor mat2)
Consider using one of the following signatures instead:
	addmm_(Tensor mat1, Tensor mat2, *, Number beta = 1, Number alpha = 1) (Triggered internally at /pytorch/torch/csrc/utils/python_arg_parser.cpp:1661.)
  dist_map.addmm_(1, -2, part_feat, part_centers.t())
Traceback (most recent call last):
  File "/home/sidna/ReID_Project/ReID/train.py", line 107, in <module>
    do_train_dict[model_name](
  File "/home/sidna/ReID_Project/ReID/processor/part_attention_vit_processor.py", line 100, in part_attention_vit_do_train_with_amp
    score, layerwise_global_feat, layerwise_feat_list = model(img)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/sidna/ReID_Project/ReID/model/make_model.py", line 291, in forward
    layerwise_tokens = self.base(x) # B, N, C
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/sidna/ReID_Project/ReID/model/backbones/vit_pytorch.py", line 685, in forward
    x = self.forward_features(x)
  File "/home/sidna/ReID_Project/ReID/model/backbones/vit_pytorch.py", line 679, in forward_features
    x = blk(x, mask)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/sidna/ReID_Project/ReID/model/backbones/vit_pytorch.py", line 272, in forward
    x = x + self.drop_path(self.mlp(self.norm2(x)))
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/sidna/ReID_Project/ReID/model/backbones/vit_pytorch.py", line 168, in forward
    x = self.fc2(x)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/sidna/miniconda3/envs/pat/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.00 MiB. GPU 0 has a total capacity of 15.77 GiB of which 12.19 MiB is free. Including non-PyTorch memory, this process has 15.75 GiB memory in use. Of the allocated memory 14.89 GiB is allocated by PyTorch, and 497.13 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
