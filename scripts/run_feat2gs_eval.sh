#! /bin/bash

GPU_ID=7
OUTPUT_ROOT="/home/chenyue/output/Feat2gs/"
DATA_ROOT_DIR="/home/chenyue/dataset/Feat2GS_Dataset_TEST"
DATASET_SPLIT_JSON="${DATA_ROOT_DIR}/dataset_split.json"

declare -A SCENES

# Scenes
declare -a SCENES_Casual=(
    # erhai
    # paper2
    plushies
    # stuff
    # xbox
)

# declare -a SCENES_DL3DV=(
#     Center
#     Electrical
#     Museum
#     Supermarket2
#     Temple
# )

# declare -a SCENES_LLFF=(
#     fortress
#     horns
#     orchids
#     room
#     trex
# )

# declare -a SCENES_MVimgNet=(
#     bench
#     bicycle
#     car
#     ladder
#     suv
# )

# declare -a SCENES_MipNeRF360=(
#     bicycle
#     # garden
#     # kitchen
#     # room
#     # stump
# )

# declare -a SCENES_Tanks=(
#     Auditorium
#     Caterpillar
#     Family
#     Ignatius
#     Train
# )

# declare -a SCENES_Infer=(
#     erhai
#     plushies
#     bread
#     brunch
#     cy
#     cy_crop
#     paper
#     paper3
#     paper4
#     stuff
#     xbox
#     desk
#     house
#     castle
#     coffee
#     cy_crop1
#     hogwarts
#     home
#     plant
# )

# Choose trajectory in [arc/spiral/lemniscate/wander/ellipse/interpolated]
RENDER_TRAJ=${RENDER_TRAJ:-interpolated} 

## Dataset
SCENES[Casual]="${SCENES_Casual[*]}"
# SCENES[DL3DV]="${SCENES_DL3DV[*]}"
# SCENES[LLFF]="${SCENES_LLFF[*]}"
# SCENES[MVimgNet]="${SCENES_MVimgNet[*]}"
# SCENES[MipNeRF360]="${SCENES_MipNeRF360[*]}"
# SCENES[Tanks]="${SCENES_Tanks[*]}"
# # SCENES[Infer]="${SCENES_Infer[*]}" # Only for inference, no test views, pls COMMENDED OUT STEP(3)(4)(5)!

# DUSt3R/MASt3R initialization
POINTMAPS=(
    dust3r
    # mast3r
    )

# Visual Foundation Models
FEATURES=(
    # radio
    # dust3r
    dino_b16
    # mast3r
    # dift
    # dinov2_b14
    # clip_b16
    # mae_b16
    # midas_l16
    # sam_base
    # iuvrgb
    )

get_train_view_count() {
    local dataset=$1
    local scene=$2
    jq -r ".[\"$dataset\"][\"$scene\"].train | length" "$DATASET_SPLIT_JSON"
}

declare -A N_VIEWS
for DATASET in "${!SCENES[@]}"; do
    read -a SCENE_ARRAY <<< "${SCENES[$DATASET]}"
    for SCENE in "${SCENE_ARRAY[@]}"; do
        N_VIEWS["${DATASET}_${SCENE}"]=$(get_train_view_count "$DATASET" "$SCENE")
    done
done

METHOD=feat2gs

# Probing modes: G:Geometry, T:Texture, A:All, ft:finetune
MODELS=(
    G
    # T
    # A
    # # Gft
    # # Tft
    # # Aft
    )

PCA_DIM=256
VIS_FEAT=True

gs_train_iter=8000


execute_command() {
    local step_name=$1
    local command=$2
    local model_path=$3

    echo "${step_name}"
    eval "$command"
}

run_process() {
    local MODEL=$1 DATASET=$2 SCENE=$3 N_VIEW=$4 POINTMAP=$5 FEATURE=$6 GPU_ID=$7

    # BASE_FOLDER must be Absolute path
    BASE_FOLDER=${DATA_ROOT_DIR}/${DATASET}/${SCENE}
    SOURCE_PATH=${BASE_FOLDER}/${N_VIEW}_views
    MODEL_PATH=${OUTPUT_ROOT}/output/eval_rerun/${DATASET}/${SCENE}/${N_VIEW}_views/${METHOD}-${MODEL}/${POINTMAP}/${FEATURE}/

    local CMD_PREFIX="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore "

    local FEATURE_ARG="${FEATURE//-/ }"
    local CMDS=(
        # ----- (1) DUSt3R initialization & Feature extraction -----
        "${CMD_PREFIX} ./coarse_init_eval.py \
            --img_base_path ${BASE_FOLDER} \
            --n_views ${N_VIEW} \
            --focal_avg \
            --method ${POINTMAP} \
            --feat_type ${FEATURE_ARG}"

        # ----- (2) Readout 3DGS from features & Jointly optimize pose -----
        "${CMD_PREFIX} ./train_feat2gs.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --n_views ${N_VIEW} \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --optim_pose \
            --method ${POINTMAP} \
            --feat_type ${FEATURE_ARG} \
            --model ${MODEL}"

        # ----- (3) Test pose initialization -----
        "${CMD_PREFIX} ./init_test_pose.py \
            --img_base_path ${BASE_FOLDER} \
            --n_views ${N_VIEW} \
            --focal_avg \
            --method ${POINTMAP} \
            --feat_type ${FEATURE_ARG}"

        # ----- (4) Render test view for evaluation -----
        "${CMD_PREFIX} ./render.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --n_views ${N_VIEW} \
            --scene ${SCENE} \
            --optim_test_pose_iter 500 \
            --iter ${gs_train_iter} \
            --eval \
            --method ${POINTMAP} \
            --feat_type ${FEATURE_ARG}"

        # ----- (5) Metrics -----
        "${CMD_PREFIX} ./metrics.py \
            -m ${MODEL_PATH} \
            --iter ${gs_train_iter} \
            --n_views ${N_VIEW}"

        # ----- (Optional) Render video with generated trajectory -----
        "${CMD_PREFIX} ./run_video.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --n_views ${N_VIEW} \
            --dataset ${DATASET} \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --eval \
            --get_video \
            --method ${POINTMAP} \
            --feat_type ${FEATURE_ARG} \
            --cam_traj ${RENDER_TRAJ}"
    )

    [ "$PCA_DIM" != "None" ] && CMDS[0]+=" --feat_dim ${PCA_DIM}" && CMDS[1]+=" --feat_dim ${PCA_DIM}"
    [ "$VIS_FEAT" = True ] && CMDS[0]+=" --vis_feat"

    execute_command "${SCENE}: STEP(1) DUSt3R initialization & Feature extraction" "${CMDS[0]}" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(2) Readout 3DGS from features & Jointly optimize pose" "${CMDS[1]}" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(3) Test pose initialization" "${CMDS[2]}" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(4) Render test view for evaluation" "${CMDS[3]}" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(5) Metric" "${CMDS[4]}" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(6) Render video with generated trajectory" "${CMDS[5]}" "$MODEL_PATH"
}

for MODEL in "${MODELS[@]}"; do
    # echo "Model: $MODEL"
    for FEATURE in ${FEATURES[@]}; do
        # echo "Feature: $FEATURE"
        for DATASET in "${!SCENES[@]}"; do
            # echo "Current dataset: $DATASET"
            read -a SCENE_ARRAY <<< "${SCENES[$DATASET]}"
            echo "Scenes to process: ${SCENE_ARRAY[@]}"
            for SCENE in "${SCENE_ARRAY[@]}"; do
                # echo "Processing scene: $SCENE"
                N_VIEW=${N_VIEWS["${DATASET}_${SCENE}"]}
                # echo "Number of views: $N_VIEW"
                for POINTMAP in "${POINTMAPS[@]}"; do
                    # echo "Pointmap: $POINTMAP"
                    # Print a summary for the current run experiment
                    info=("GPU $GPU_ID is free, starting run:" "MODEL-${MODEL}" "DATASET-${DATASET}" "SCENE-${SCENE}" "N_VIEW-${N_VIEW}" "POINTMAP-${POINTMAP}" "FEATURE-${FEATURE}")
                    max_length=$(printf "%s\n" "${info[@]}" | wc -L)
                    max_length=$((max_length + 4))
                    border=$(printf "+%${max_length}s+" | tr " " "-")
                    echo "$border"
                    for line in "${info[@]}"; do
                        printf "| %-$((max_length-2))s |\n" "$line"
                    done
                    echo "$border"

                    run_process $MODEL $DATASET $SCENE $N_VIEW $POINTMAP $FEATURE $GPU_ID
                 
                done
            done
        done
    done
done
