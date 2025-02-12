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

OUTPUT_ROOT="/home/chenyue/output/Feat2gs/"
DATA_ROOT_DIR="/home/chenyue/dataset/Feat2GS_Dataset"
DATASET_SPLIT_JSON="${DATA_ROOT_DIR}/dataset_split.json"

declare -A SCENES

declare -a SCENES_Casual=(
    erhai
    paper2
    plushies
    stuff
    xbox
    # bread
    # brunch
    # cy2
    # cy3
    # desk
    # house
)

declare -a SCENES_DL3DV=(
    Center
    Electrical
    Museum
    Supermarket2
    Temple
    # Supermarket1
    # Furniture
    # Gallery
    # Garden
)

declare -a SCENES_LLFF=(
    fortress
    horns
    orchids
    room
    trex
    # fern
    # flower
)

declare -a SCENES_MVimgNet=(
    bench
    bicycle
    car
    ladder
    suv
    # chair
    # table
)

declare -a SCENES_MipNeRF360=(
    bicycle
    garden
    kitchen
    room
    stump
    # bonsai
    # counter
)

declare -a SCENES_Tanks=(
    Auditorium
    Caterpillar
    Family
    Ignatius
    Train
    # Barn
    # Church
    # Francis
    # Horse
    # Museum
    # Playground
    # Truck
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

RENDER_DEPTH_NORMAL=false

# Choose trajectory in [arc/spiral/lemniscate/wander/ellipse/interpolated]
RENDER_TRAJ=${RENDER_TRAJ:-interpolated}

SCENES[Casual]="${SCENES_Casual[*]}"
SCENES[DL3DV]="${SCENES_DL3DV[*]}"
SCENES[LLFF]="${SCENES_LLFF[*]}"
SCENES[MVimgNet]="${SCENES_MVimgNet[*]}"
SCENES[MipNeRF360]="${SCENES_MipNeRF360[*]}"
SCENES[Tanks]="${SCENES_Tanks[*]}"
# SCENES[Infer]="${SCENES_Infer[*]}" # Only for inference, no test views, pls COMMENDED OUT STEP(3)(4)(5)!

POINTMAPS=(
    dust3r
    # mast3r
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

METHOD=instantsplat

gs_train_iter=7000

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
    local DATASET=$1 SCENE=$2 N_VIEW=$3 POINTMAP=$4 GPU_ID=$5

    local ERROR_LOG="${ERROR_DIR}/${METHOD}-${DATASET}-${SCENE}-${N_VIEW}-${POINTMAP}.log"
    local LOCK_FILE="$LOCK_DIR/gpu_$GPU_ID.lock"
    touch $LOCK_FILE  # Create a lock file to reserve the GPU
    {
        echo "Running:"
        echo "MODEL-${METHOD}"
        echo "DATASET-${DATASET}"
        echo "SCENE-${SCENE}"
        echo "N_VIEW-${N_VIEW}"
        echo "POINTMAP-${POINTMAP}"
    } > $LOCK_FILE

    # BASE_FOLDER must be Absolute path
    BASE_FOLDER=${DATA_ROOT_DIR}/${DATASET}/${SCENE}
    SOURCE_PATH=${BASE_FOLDER}/${N_VIEW}_views
    MODEL_PATH=${OUTPUT_ROOT}/output/eval/${DATASET}/${SCENE}/${N_VIEW}_views/${METHOD}/${POINTMAP}/

    local CMD_PREFIX="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore "

    local CMDS=(
        # ----- (1) Dust3r_coarse_geometric_initialization -----
        "${CMD_PREFIX} ./coarse_init_eval.py \
            --img_base_path ${BASE_FOLDER} \
            --n_views ${N_VIEW} \
            --focal_avg \
            --method ${POINTMAP}"

        # ----- (2) Train: jointly optimize pose -----
        "${CMD_PREFIX} ./train_instantsplat.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --n_views ${N_VIEW} \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --optim_pose \
            --method ${POINTMAP}"

        # ----- (3) Test pose initialization -----
        "${CMD_PREFIX} ./init_test_pose.py \
            --img_base_path ${BASE_FOLDER} \
            --n_views ${N_VIEW} \
            --focal_avg \
            --method ${POINTMAP}"

        # ----- (4) Render test view for evaluation -----
        "${CMD_PREFIX} ./render.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --n_views ${N_VIEW} \
            --scene ${SCENE} \
            --optim_test_pose_iter 500 \
            --iter ${gs_train_iter} \
            --eval \
            --method ${POINTMAP}"

        # # ----- (4) Render Depth & Normal -----
        # "${CMD_PREFIX} ./render_dep_nom.py \
        #     -s ${SOURCE_PATH} \
        #     -m ${MODEL_PATH} \
        #     --n_views ${N_VIEW} \
        #     --scene ${SCENE} \
        #     --optim_test_pose_iter 500 \
        #     --iter ${gs_train_iter} \
        #     --eval \
        #     --method ${POINTMAP}"

        # ----- (5) Metrics -----
        "${CMD_PREFIX} ./metrics.py \
            -m ${MODEL_PATH} \
            --iter ${gs_train_iter} \
            --n_views ${N_VIEW}"
    )

    # ----- (Optional) Render video with generated trajectory -----
    if [ "$RENDER_DEPTH_NORMAL" = true ]; then
        CMDS+=("${CMD_PREFIX} ./run_video_dep_nom.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --n_views ${N_VIEW} \
            --dataset ${DATASET} \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --eval \
            --get_video \
            --method ${POINTMAP} \
            --cam_traj ${RENDER_TRAJ}")
    else
        CMDS+=("${CMD_PREFIX} ./run_video.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --n_views ${N_VIEW} \
            --dataset ${DATASET} \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --eval \
            --get_video \
            --method ${POINTMAP} \
            --cam_traj ${RENDER_TRAJ}")
    fi

    execute_command "${SCENE}: STEP(1) DUSt3R coarse geometric initialization" "${CMDS[0]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(2) Train: jointly optimize pose" "${CMDS[1]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(3) Test pose initialization" "${CMDS[2]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(4) Render test view for evaluation" "${CMDS[3]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(5) Metric" "${CMDS[4]}" "$ERROR_LOG" "$MODEL_PATH"
    execute_command "${SCENE}: STEP(6) Render video with generated trajectory" "${CMDS[5]}" "$ERROR_LOG" "$MODEL_PATH"

    # Clean up lock file after completion
    rm -f $LOCK_FILE
}

for DATASET in "${!SCENES[@]}"; do
    read -a SCENE_ARRAY <<< "${SCENES[$DATASET]}"
    echo "Scenes to process: ${SCENE_ARRAY[@]}"

    for SCENE in "${SCENE_ARRAY[@]}"; do
        N_VIEW=${N_VIEWS["${DATASET}_${SCENE}"]}
        for POINTMAP in "${POINTMAPS[@]}"; do
            GPU_FOUND=0
            while [ $GPU_FOUND -eq 0 ]; do
                for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
                    if is_gpu_free $GPU_ID; then
                        # Print a summary for the current run experiment
                        info=("GPU $GPU_ID is free, starting run:" "DATASET-${DATASET}" "SCENE-${SCENE}" "N_VIEW-${N_VIEW}" "POINTMAP-${POINTMAP}")
                        max_length=$(printf "%s\n" "${info[@]}" | wc -L)
                        max_length=$((max_length + 4))
                        border=$(printf "+%${max_length}s+" | tr " " "-")
                        echo "$border"
                        for line in "${info[@]}"; do
                            printf "| %-$((max_length-2))s |\n" "$line"
                        done
                        echo "$border"

                        run_process $DATASET $SCENE $N_VIEW $POINTMAP $GPU_ID &
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

# Wait for all background tasks to complete
wait