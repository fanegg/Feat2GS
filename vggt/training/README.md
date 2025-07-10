# Training

This is a re-implementation of our framework for training VGGT. This document shows how to set up the environment and run VGGT training. I have aimed to faithfully reproduce the original training framework, but please open an issue if anything looks off.


## 1. Prerequisites

Before you begin, ensure you have completed the following steps:

1.  **Install VGGT as a package:**
    ```bash
    pip install -e .
    ```

2.  **Prepare the dataset and annotations:**
    -   Download the Co3D dataset from the [official repository](https://github.com/facebookresearch/co3d).
    -   Download the required annotation files from [Hugging Face](https://huggingface.co/datasets/JianyuanWang/co3d_anno/tree/main).

## 2. Configuration

After downloading the dataset and annotations, you need to configure the paths in `training/config/default.yaml`.

1.  Open `training/config/default.yaml`.
2.  In this file, modify the `CO3D_DIR` and `CO3D_ANNOTATION_DIR` to the absolute paths where you have stored the dataset and annotations, respectively.

```yaml
data:
  train:
    dataset:
      dataset_configs:
        - _target_: data.datasets.co3d.Co3dDataset
          split: train
          CO3D_DIR: /YOUR/PATH/TO/CO3D
          CO3D_ANNOTATION_DIR: /YOUR/PATH/TO/CO3D_ANNOTATION
# ... same for val ...
```

## 3. Fine-tuning on Co3D

To fine-tune the provided pre-trained model on the Co3D dataset, run the following command. This example uses 4 GPUs with PyTorch Distributed Data Parallel (DDP).


```bash
torchrun --nproc_per_node=4 launch.py
```

The default configuration in `training/config/default.yaml` is set up for fine-tuning. It automatically resumes from a checkpoint and freezes the model's `aggregator` module during training.

## 4. Memory Management

If you encounter out-of-memory (OOM) errors on your GPU, consider adjusting the following parameters in `training/config/default.yaml`:

-   `max_img_per_gpu`: Reduce this value to decrease the batch size per GPU.
-   `accum_steps`: This sets the number of gradient accumulation steps (default is 2). This feature splits batches into smaller chunks to save memory, though it may slightly increase training time. Note that gradient accumulation was not used for the original VGGT model.



