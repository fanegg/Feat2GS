conda activate feat2gs
cd Feat2GS/

bash scripts/run_feat2gs_eval_parallel.sh
bash scripts/run_feat2gs_eval.sh
bash scripts/run_instantsplat_eval_parallel.sh
bash scripts/run_feat2gs_eval_dtu_parallel.sh

python video/generate_video.py

bash scripts/run_all_trajectories.sh
bash scripts/run_video_render.sh
bash scripts/run_video_render_instantsplat.sh
bash scripts/run_video_render_dtu.sh

tensorboard --logdir=/home/chenyue/output/Feat2gs/output/eval/ --port=7001

cd /home/chenyue/output/Feat2gs/output/eval/Tanks/Train/6_views/feat2gs-G/dust3r/
tensorboard --logdir_spec \
radio:radio,\
dust3r:dust3r,\
dino_b16:dino_b16,\
mast3r:mast3r,\
dift:dift,\
dinov2:dinov2_b14,\
clip:clip_b16,\
mae:mae_b16,\
midas:midas_l16,\
sam:sam_base,\
iuvrgb:iuvrgb \
--port 7002

CUDA_VISIBLE_DEVICES=7 gradio demo.py