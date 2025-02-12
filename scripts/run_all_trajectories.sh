#! /bin/bash

TRAJECTORIES=(
    "ellipse"
    "interpolated"
    "wander"
    "arc"
    "spiral"
    "lemniscate"
)


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

# Function to check if any GPU is free
is_any_gpu_free() {
    for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
        if is_gpu_free $GPU_ID; then
            return 0
        fi
    done
    return 1 
}


for TRAJ in "${TRAJECTORIES[@]}"; do
    while ! is_any_gpu_free; do
        sleep 3
    done
    
    echo "======================================"
    echo "Starting trajectory: $TRAJ"
    echo "======================================"
    
    (
        export RENDER_TRAJ=$TRAJ
        # bash scripts/run_instantsplat_eval_parallel.sh
        bash scripts/run_feat2gs_eval_parallel.sh
    ) &
    
    sleep 5
done

wait

echo "Done!"