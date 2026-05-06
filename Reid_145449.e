Traceback (most recent call last):
  File "/home/sidna/ReID_Project/ReID/train.py", line 74, in <module>
    model = make_model(cfg, modelname=model_name, num_class=num_classes, camera_num=None, view_num=None)
  File "/home/sidna/ReID_Project/ReID/model/make_model.py", line 368, in make_model
    model = build_part_attention_vit(num_class, cfg, __factory_LAT_type)
  File "/home/sidna/ReID_Project/ReID/model/make_model.py", line 278, in __init__
    self.base.load_param(self.model_path)
  File "/home/sidna/ReID_Project/ReID/model/backbones/vit_pytorch.py", line 721, in load_param
    self.state_dict()[k].copy_(v)
KeyError: 'base.cls_token'
