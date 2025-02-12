
<h2 align="center"> <a href="https://arxiv.org/abs/2412.09606">Feat2GS: Probing Visual Foundation Models with Gaussian Splatting</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2412.09606-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09606) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://fanegg.github.io/Feat2GS/)  [![youtube](https://img.shields.io/badge/Video-E33122?logo=Youtube)](https://youtu.be/4fT5lzcAJqo?si=_fCSIuXNBSmov2VA)  [![X](https://img.shields.io/badge/@Yue%20Chen-black?logo=X)](https://twitter.com/faneggchen)  [![Bluesky](https://img.shields.io/badge/@Yue%20Chen-white?logo=Bluesky)](https://bsky.app/profile/fanegg.bsky.social) 
<!-- [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kairunwen/InstantSplat)  -->

[Yue Chen](https://fanegg.github.io/),
[Xingyu Chen](https://rover-xingyu.github.io/),
[Anpei Chen](https://apchenstu.github.io/),
[Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/),
[Yuliang Xiu](https://xiuyuliang.cn/)
</h5>

<div align="center">
This repository is the official implementation of Feat2GS, a unified framework to probe “texture and geometry awareness” of visual foundation models. Novel view synthesis serves as an effective proxy for 3D evaluation.
</div>
<br>

https://github.com/user-attachments/assets/07ebb8e1-6001-47bf-bf74-984b0032cc17

## Get Started

### Installation
1. Clone Feat2GS and download pre-trained model from [DUSt3R](https://github.com/naver/dust3r)/[MASt3R](https://github.com/naver/mast3r).
```bash
git clone https://github.com/fanegg/Feat2GS.git
cd Feat2GS/submodules/mast3r/
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
cd ../../
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n feat2gs python=3.11 cmake=3.14.0
conda activate feat2gs
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
cd Feat2GS/
pip install -r requirements.txt
pip install submodules/simple-knn
```

3. Optional but highly suggested, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd submodules/mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../
```

4. (Optional) follow [this instruction](https://github.com/cvlab-columbia/zero123?tab=readme-ov-file#novel-view-synthesis-1) to install requirements for probing [Zero123](https://github.com/cvlab-columbia/zero123).

### Usage
1. Data preparation (We provide our evaluation and inference datasets: [link](https://drive.google.com/file/d/1PLTFcvJfiPucrB-pIwfp5QG-AIHcJdjN/view?usp=drive_link)
```bash
  cd <data_root>/Feat2GS/
```

If you want to build custom datasets, please follow and edit:
```
build_dataset/0_create_json.py ## create dataset_split.json to split train/test set
build_dataset/1_create_feat2gs_dataset.py ## use dataset_split.json to create dataset
```


2. Evaluate Visual Foundation Models:

  | Step | Description (link to command) |
  |------|-------------|
  | (1)  | [DUSt3R initialization & Feature extraction](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L245-L250) |
  | (2)  | [Readout 3DGS from features & Jointly optimize pose](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L253-L262) |
  | (3)  | [Test pose initialization](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L265-L270) |
  | (4)  | [Render test view for evaluation](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L273-L282) |
  | (5)  | [Metric](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L298-L301) |
  | (Optional)  | [Render video with generated trajectory](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L304-L315) |

```bash
  # Run evaluation for all dataset, all VFM features, all probing mode
  bash scripts/run_feat2gs_eval_parallel.sh

  # Run evaluation for a single scene, DINO feature, Geometry mode
  bash scripts/run_feat2gs_eval.sh
```
> [!NOTE]
> To run experiments in parallel, we added a **GPU lock** feature to ensure only one evaluation experiment runs per GPU. Once an experiment finishes, the GPU is automatically unlocked. **If interrupted by `Ctrl+C`, the GPU won’t be unlocked automatically.** To fix this, manually delete the `.lock` file in the `LOCK_DIR`. To disable this feature, comment out these lines in the script:
    [L4-L5](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L4-L5),
    [L9-L22](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L9-L22),
    [L223-L233](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L223-L233),
    [L330-L331](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L330-L331),

  | Config | Operation |
  |--------|-----------------|
  | GPU | Edit in [`<AVAILABLE_GPUS>`](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L7) |
  | Dataset | Edit in [`<SCENES[$Dataset]>`](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L105-L111) |
  | Scene | Edit in [`<SCENES_$Dataset>`](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L31-L99) |
  | Visual Foundation Model | Edit in [`<FEATURES>`](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L120-L162) |
  | Probing Mode | Edit in [`<MODELS>`](https://github.com/fanegg/Feat2GS/blob/b8eadaa54549d34420eba61b388548b8ec8e7325/scripts/run_feat2gs_eval_parallel.sh#L181-L188) |
  | Inference-only Mode | Comment out STEP (3)(4)(5) in [`execute_command`](https://github.com/fanegg/Feat2GS/blob/main/scripts/run_feat2gs_eval_parallel.sh#L325-L327) |

```bash
  # Evaluate Visual Foundation Models on DTU dataset
  bash scripts/run_feat2gs_eval_dtu_parallel.sh

  # Run InstantSplat for evaluation
  bash scripts/run_instantsplat_eval_parallel.sh
```


3. After training, render RGB/depth/normal video with generated trajectory.
```bash
  # If render depth/normal, set RENDER_DEPTH_NORMAL=true
  # Set type of generated trjectory by editing <TRAJ_SCENES>
  bash bash scripts/run_video_render.sh

  # Render video on DTU dataset
  bash scripts/run_video_render_dtu.sh
```
### 🎮 Interactive demo

#### 🚀 Quickstart
1. **Input Images**
* Upload 2 or more images of the same scene from different views
* For best results, ensure images have good overlap

2. **Step 1: DUSt3R Initialization & Feature Extraction**
* Click "RUN Step 1" to process your images
* This step estimates initial DUSt3R point cloud and camera poses, and extracts DUSt3R features for each pixel

3. **Step 2: Readout 3DGS from Features**
* Set the number of training iterations, larger number leads to better quality but longer time (default: 2000, max: 8000) 
* Click "RUN Step 2" to optimize the 3D model

4. **Step 3: Video Rendering**
* Choose a camera trajectory
* Click "RUN Step 3" to generate a video of your 3D model
  
```bash
gradio demo.py
```

#### 💡 Tips
* Processing time depends on image resolution and quantity
* For optimal performance, test on high-end GPUs (A100/4090)
* Use the mouse to interact with 3D models:
  - Left button: Rotate
  - Scroll wheel: Zoom
  - Right button: Pan


## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [DUSt3R](https://github.com/naver/dust3r) and [MASt3R](https://github.com/naver/mast3r)
- [InstantSplat](https://github.com/NVlabs/InstantSplat)
- [Probe3D](https://github.com/mbanani/probe3d)
- [FeatUp](https://github.com/mhamilton723/FeatUp)
- [Shape of Motion](https://github.com/vye16/shape-of-motion/)
- [Splatt3R](https://github.com/btsmart/splatt3r)

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@article{chen2024feat2gs,
  title={Feat2GS: Probing Visual Foundation Models with Gaussian Splatting},
  author={Chen, Yue and Chen, Xingyu and Chen, Anpei and Pons-Moll, Gerard and Xiu, Yuliang},
  journal={arXiv preprint arXiv:2412.09606},
  year={2024}
}
```

## Contact

For feedback, questions, or press inquiries please contact [Yue Chen](faneggchen@gmail.com) and [Xingyu Chen](roverxingyu@gmail.com)
