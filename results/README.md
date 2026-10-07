# Results

`submissions/` holds Kaggle submission CSVs (`imageName, Corresponding Indexes`: top-100 gallery
indices per query) produced while developing the method. File names describe the setting, e.g.
`vitlarge25nrmyn_clsfusion6` = ViT-L, epoch 25, CLS fusion K = 6; `_qmv` = Query Majority Voting;
`_camfeat` = camera feature normalisation; `_prototype` / `_shape…` = traffic-sign filtering.

The leaderboard scores of individual files were not recorded in the repository; the paper's Table 1
reports the scores of the configurations in `configs/paper/`.
