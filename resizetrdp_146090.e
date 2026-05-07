Traceback (most recent call last):
  File "/home/sidna/ReID_Project/ReID_vitlarge64/train.py", line 71, in <module>
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
  File "/home/sidna/ReID_Project/ReID_vitlarge64/data/build_DG_dataloader.py", line 100, in build_reid_test_loader
    dataset.show_test()
  File "/home/sidna/ReID_Project/ReID_vitlarge64/data/datasets/bases.py", line 169, in show_test
    num_query_pids, num_query_cams = self.parse_data(self.query)
  File "/home/sidna/ReID_Project/ReID_vitlarge64/data/datasets/bases.py", line 74, in parse_data
    for _, pid, camid, _ in data:
ValueError: too many values to unpack (expected 4)
