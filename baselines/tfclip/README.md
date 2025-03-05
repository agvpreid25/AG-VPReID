# TFCLIP - Baseline

This repository contains code for training and evaluation on the AG-VPReID dataset.

## Environment Setup

```bash
conda create -n tfclip python=3.8
conda activate tfclip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install yacs timm scikit-image tqdm ftfy regex
```

## Training

To train the model:

```bash
python train_main.py --output_dir "logs/all"
```

## Evaluation

The repository supports two cross-view matching scenarios:

1. Aerial-to-Ground matching:
```bash
python eval_main.py --custom_output_dir "results/case1_aerial_to_ground" --output_dir "logs/all"
```

2. Ground-to-Aerial matching:
```bash
python eval_main.py --custom_output_dir "results/case2_ground_to_aerial" --output_dir "logs/all"
```

Note: For case 2, you need to modify the dataset path in `datasets/set/agreidvid.py` to point to `case2_ground_to_aerial` for query and gallery.

## Submission File

- After completing each case, the `tracklet_rankings.csv` file can be found in the corresponding custom output directory. For example: `logs/all/results/case1_aerial_to_ground_mat/tracklet_rankings.csv`
- To create the final CSV submission file, combine the two files with case 1 results first followed by case 2 results as specified in the sample_submission.

## Hardware Requirements

This code has been tested on NVIDIA A100 GPUs.

## Acknowledgment
This baseline is based on the work of [TF-CLIP](https://github.com/Syliz517/CLIP-REID). We appreciate the authors for their excellent contribution.