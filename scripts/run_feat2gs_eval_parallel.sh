#! /bin/bash

ERROR_DIR="/home/chenyue/tmp/error"
LOCK_DIR="/home/chenyue/tmp/gpu_locks"
mkdir -p $LOCK_DIR

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

# Function to check if a GPU is free
is_gpu_free() {
    local GPU_ID=$1
    local LOCK_FILE="$LOCK_DIR/gpu_$GPU_ID.lock"
    local STOP_FILE="$LOCK_DIR/gpu_$GPU_ID.stop"

    if [ -f $STOP_FILE ]; then
        return 1
    elif [ -f $LOCK_FILE ]; then
        return 1  # GPU is locked
    else
        return 0  # GPU is free
    fi
}

OUTPUT_ROOT="/home/chenyue/output/Feat2gs"
DATA_ROOT_DIR="/home/chenyue/dataset/Feat2GS_Dataset"
DATASET_SPLIT_JSON="${DATA_ROOT_DIR}/dataset_split.json"

declare -A SCENES

# Scenes
declare -a SCENES_Casual=(
    erhai
    paper2
    plushies
    stuff
    xbox
)

declare -a SCENES_DL3DV=(
    Center
    Electrical
    Museum
    Supermarket2
    Temple
)

declare -a SCENES_LLFF=(
    fortress
    horns
    orchids
    room
    trex
)

declare -a SCENES_MVimgNet=(
    bench
    bicycle
    car
    ladder
    suv
)

declare -a SCENES_MipNeRF360=(
    bicycle
    garden
    kitchen
    room
    stump
)

declare -a SCENES_Tanks=(
    Auditorium
    Caterpillar
    Family
    Ignatius
    Train
)

declare -a SCENES_Infer=(
    erhai
    plushies
    bread
    brunch
    cy
    cy_crop
    paper
    paper3
    paper4
    stuff
    xbox
    desk
    house
    castle
    coffee
    cy_crop1
    hogwarts
    home
    plant
)

# Choose trajectory in [arc/spiral/lemniscate/wander/ellipse/interpolated]
RENDER_TRAJ=${RENDER_TRAJ:-interpolated} 

# Dataset
SCENES[Casual]="${SCENES_Casual[*]}"
SCENES[DL3DV]="${SCENES_DL3DV[*]}"
SCENES[LLFF]="${SCENES_LLFF[*]}"
SCENES[MVimgNet]="${SCENES_MVimgNet[*]}"
SCENES[MipNeRF360]="${SCENES_MipNeRF360[*]}"
SCENES[Tanks]="${SCENES_Tanks[*]}"
# SCENES[Infer]="${SCENES_Infer[*]}" # Only for inference, no test views, pls COMMENDED OUT STEP(3)(4)(5)!

# DUSt3R/MASt3R initialization
POINTMAPS=(
    dust3r
    # mast3r
    )

# Visual Foundation Models
FEATURES=(
    vggt_e
    vggt_d
    radio
    dust3r
    dino_b16
    mast3r
    dift
    dinov2_b14
    clip_b16
    mae_b16
    midas_l16
    sam_base
    iuvrgb
    # zero123

    #### Open it when doing Feature Concatenation
    # mast3r-radio
    # dust3r-mast3r-radio
    # dust3r-mast3r-dino_b16-radio
    # radio-mast3r-dust3r-dino_b16-mae_b16
    # radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16
    # radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16
    # radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14
    # radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base
    # radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base-dift
    # radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base-dift-iuvrgb

    # iuvrgb-dift
    # iuvrgb-dift-sam_base
    # iuvrgb-dift-sam_base-dinov2_b14
    # iuvrgb-dift-sam_base-dinov2_b14-clip_b16
    # iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16
    # iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16
    # iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16
    # iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r
    # iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r-mast3r
    # # iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r-mast3r-radio

    # dinov2_b14-clip_b16-sam_base

    # dino16_featup
    # dinov2_featup
    # clip_featup
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
    T
    A
    # Gft
    # Tft
    # Aft
    )

PCA_DIM=256
VIS_FEAT=True

gs_train_iter=8000


execute_command() {
    local step_name=$1
    local command=$2
    local error_log=$3
    local model_path=$4

    echo "${step_name}"

    if ! output=$(eval "$command" 2>&1); then
        {
            echo "========= ${step_name} ========="
            echo "Model Path: ${model_path}"
            echo "Execution ended at: $(date)"
            echo "Output:"
            echo "$output"
            echo "----------------------------------------"
        } >> "$error_log"
        echo "Error: ${step_name} failed. Error message has been logged to $error_log"
    else
        echo "$output"
    fi
}

run_process() {
    local MODEL=$1 DATASET=$2 SCENE=$3 N_VIEW=$4 POINTMAP=$5 FEATURE=$6 GPU_ID=$7 FIRST_RUN=$8

    local ERROR_LOG="${ERROR_DIR}/${MODEL}-${DATASET}-${SCENE}-${N_VIEW}-${POINTMAP}-${FEATURE}.log"
    local LOCK_FILE="$LOCK_DIR/gpu_$GPU_ID.lock"
    touch $LOCK_FILE  # Create a lock file to reserve the GPU
    {
        echo "Running:"
        echo "MODEL-${MODEL}"
        echo "DATASET-${DATASET}"
        echo "SCENE-${SCENE}"
        echo "N_VIEW-${N_VIEW}"
        echo "POINTMAP-${POINTMAP}"
        echo "FEATURE-${FEATURE}"
    } > $LOCK_FILE

    # BASE_FOLDER must be Absolute path
    BASE_FOLDER=${DATA_ROOT_DIR}/${DATASET}/${SCENE}
    SOURCE_PATH=${BASE_FOLDER}/${N_VIEW}_views
    MODEL_PATH=${OUTPUT_ROOT}/output/eval/${DATASET}/${SCENE}/${N_VIEW}_views/${METHOD}-${MODEL}/${POINTMAP}/${FEATURE}/

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

        # # ----- (4) Render Depth & Normal -----
        # "${CMD_PREFIX} ./render_dep_nom.py \
        #     -s ${SOURCE_PATH} \
        #     -m ${MODEL_PATH} \
        #     --n_views ${N_VIEW} \
        #     --scene ${SCENE} \
        #     --optim_test_pose_iter 500 \
        #     --iter ${gs_train_iter} \
        #     --eval \
        #     --method ${POINTMAP} \
        #     --feat_type ${FEATURE_ARG} \
        # #     --render_depth_normal"

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

    if $FIRST_RUN; then
      execute_command "${SCENE}: STEP(1) DUSt3R initialization & Feature extraction" "${CMDS[0]}" "$ERROR_LOG" "$MODEL_PATH"
    fi
    execute_command "${SCENE}: STEP(2) Readout 3DGS from features & Jointly optimize pose" "${CMDS[1]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(3) Test pose initialization" "${CMDS[2]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(4) Render test view for evaluation" "${CMDS[3]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(5) Metric" "${CMDS[4]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(6) Render video with generated trajectory" "${CMDS[5]}" "$ERROR_LOG" "$MODEL_PATH"

    # Clean up lock file after completion
    rm -f $LOCK_FILE
}

INIT_RUN=true
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
                        GPU_FOUND=0
                        while [ $GPU_FOUND -eq 0 ]; do
                            for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
                                if is_gpu_free $GPU_ID; then
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

                                    run_process $MODEL $DATASET $SCENE $N_VIEW $POINTMAP $FEATURE $GPU_ID $INIT_RUN &
                                    sleep 3

                                    GPU_FOUND=1
                                    break # Break out of the outer loop and continue assigning the next task
                                fi
                            done
                            if [ $GPU_FOUND -eq 0 ]; then
                                sleep 3  # Wait for 3 seconds before retrying if no GPUs are free
                            fi
                        done
                    
                done
            done
        done
    done

    INIT_RUN=false
done
# Wait for all background tasks to complete
wait