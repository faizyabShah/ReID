# ReID

## Setup Instructions

Clone this repo.
```bash
git clone [https://github.com/faizyabShah/ReID.git](https://github.com/faizyabShah/ReID.git)
cd ReID
```

Run these commands:
```bash
conda create -n pat python==3.10
conda activate pat
bash enviroments.sh
```

Then download the model file from this link:
[jx_vit_base_p16_224-80ecf9dd.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

Then change the dataset directory and model directory in `config/UrbanElementsReID_test.yml` and `config/UrbanElementsReID_train.yml`.

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
