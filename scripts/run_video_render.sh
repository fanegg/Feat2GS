#! /bin/bash

ERROR_DIR="/home/chenyue/tmp/error"
LOCK_DIR="/home/chenyue/tmp/gpu_locks"
mkdir -p $LOCK_DIR
mkdir -p $ERROR_DIR

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

RENDER_DEPTH_NORMAL=false  # true

declare -A TRAJ_SCENES=(
    ["interpolated"]="DL3DV/Center DL3DV/Electrical DL3DV/Museum DL3DV/Supermarket2 DL3DV/Temple Infer/erhai Infer/paper4 MVimgNet/bench MVimgNet/suv MVimgNet/car Tanks/Train"
    
    ["lemniscate"]="Infer/cy Infer/bread Infer/brunch Infer/paper4 Infer/plushies LLFF/fortress LLFF/horns LLFF/orchids LLFF/trex LLFF/room MipNeRF360/room MipNeRF360/garden MVimgNet/bench Tanks/Family Tanks/Auditorium Tanks/Ignatius"
    
    ["spiral"]="Infer/cy LLFF/orchids LLFF/trex LLFF/fortress DL3DV/Center Infer/castle"
    
    ["ellipse"]="Infer/bread Infer/brunch MipNeRF360/kitchen"
    
    ["arc"]="Infer/paper LLFF/horns Tanks/Auditorium Tanks/Ignatius Tanks/Caterpillar"

    ["wander"]="Infer/erhai"
)

POINTMAPS=(
    dust3r
    )

FEATURES=(
    radio
    dust3r
    dino_b16
    mast3r
    dift
    dinov2_b14
    clip_b16
    midas_l16
    mae_b16
    sam_base
    iuvrgb
    # dust3r-mast3r-dift-dino_b16-dinov2_b14-radio-clip_b16-mae_b16-midas_l16-sam_base-iuvrgb
    )


METHOD=feat2gs
MODELS=(
    G
    T
    A
    # Gft
    # Tft
    # Aft
    )


get_train_view_count() {
    local dataset=$1
    local scene=$2
    jq -r ".[\"$dataset\"][\"$scene\"].train | length" "$DATASET_SPLIT_JSON"
}

declare -A N_VIEWS
for TRAJ in "${!TRAJ_SCENES[@]}"; do
    read -ra SCENES_FOR_TRAJ <<< "${TRAJ_SCENES[$TRAJ]}"
    for SCENE_PATH in "${SCENES_FOR_TRAJ[@]}"; do
        DATASET=${SCENE_PATH%%/*}
        SCENE=${SCENE_PATH#*/}
        N_VIEWS["${DATASET}_${SCENE}"]=$(get_train_view_count "$DATASET" "$SCENE")
    done
done

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
    local MODEL=$1 DATASET=$2 SCENE=$3 N_VIEW=$4 POINTMAP=$5 FEATURE=$6 GPU_ID=$7 RENDER_TRAJ=$8
    
    local ERROR_LOG="${ERROR_DIR}/${MODEL}-${DATASET}-${SCENE}-${N_VIEW}-${POINTMAP}-${FEATURE}-${RENDER_TRAJ}.log"
    local LOCK_FILE="$LOCK_DIR/gpu_$GPU_ID.lock"
    touch $LOCK_FILE
    {
        echo "Running:"
        echo "MODEL-${MODEL}"
        echo "DATASET-${DATASET}"
        echo "SCENE-${SCENE}"
        echo "N_VIEW-${N_VIEW}"
        echo "POINTMAP-${POINTMAP}"
        echo "FEATURE-${FEATURE}"
        echo "TRAJ-${RENDER_TRAJ}"
    } > $LOCK_FILE

    # BASE_FOLDER must be Absolute path
    BASE_FOLDER=${DATA_ROOT_DIR}/${DATASET}/${SCENE}
    SOURCE_PATH=${BASE_FOLDER}/${N_VIEW}_views
    MODEL_PATH=${OUTPUT_ROOT}/output/eval/${DATASET}/${SCENE}/${N_VIEW}_views/${METHOD}-${MODEL}/${POINTMAP}/${FEATURE}/

    local CMD_PREFIX="CUDA_VISIBLE_DEVICES=${GPU_ID} python"

    local FEATURE_ARG="${FEATURE//-/ }"
    local CMD="${CMD_PREFIX} -W ignore"
    if [ "$RENDER_DEPTH_NORMAL" = true ]; then
        CMD+=" ./run_video_dep_nom.py"
    else
        CMD+=" ./run_video.py"
    fi
    CMD+=" -s ${SOURCE_PATH} \
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


    execute_command "${SCENE}: Render ${RENDER_TRAJ} trajectory" "$CMD" "$ERROR_LOG" "$MODEL_PATH"

    # Clean up lock file after completion
    rm -f $LOCK_FILE
}

for TRAJ in "${!TRAJ_SCENES[@]}"; do
    read -ra SCENES_FOR_TRAJ <<< "${TRAJ_SCENES[$TRAJ]}"
    
    for SCENE_PATH in "${SCENES_FOR_TRAJ[@]}"; do
        DATASET=${SCENE_PATH%%/*}
        SCENE=${SCENE_PATH#*/}
        N_VIEW=${N_VIEWS["${DATASET}_${SCENE}"]}
        
        for MODEL in "${MODELS[@]}"; do
            for FEATURE in "${FEATURES[@]}"; do
                for POINTMAP in "${POINTMAPS[@]}"; do
                    GPU_FOUND=0
                    while [ $GPU_FOUND -eq 0 ]; do
                        for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
                            if is_gpu_free $GPU_ID; then
                                info=("GPU $GPU_ID is free, starting run:" "MODEL-${MODEL}" "DATASET-${DATASET}" "SCENE-${SCENE}" "N_VIEW-${N_VIEW}" "POINTMAP-${POINTMAP}" "FEATURE-${FEATURE}" "TRAJ-${TRAJ}")
                                max_length=$(printf "%s\n" "${info[@]}" | wc -L)
                                max_length=$((max_length + 4))
                                border=$(printf "+%${max_length}s+" | tr " " "-")
                                echo "$border"
                                for line in "${info[@]}"; do
                                    printf "| %-$((max_length-2))s |\n" "$line"
                                done
                                echo "$border"

                                run_process $MODEL $DATASET $SCENE $N_VIEW $POINTMAP $FEATURE $GPU_ID $TRAJ &
                                sleep 3

                                GPU_FOUND=1
                                break
                            fi
                        done
                        if [ $GPU_FOUND -eq 0 ]; then
                            sleep 3
                        fi
                    done
                done
            done
        done
    done
done

wait