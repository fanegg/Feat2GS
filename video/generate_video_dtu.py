import av
import numpy as np
import json
import os
import cv2
from tqdm import tqdm
import datetime
import traceback
from pathlib import Path

def get_train_view_count(dataset_split_json, dataset, scene):
    with open(dataset_split_json, 'r') as f:
        data = json.load(f)
    return len(data[dataset][scene]["train"])

def create_video_pair(base_path, dataset, scene, method, model, pointmap, feature,
                     dataset_split_json, output_dir, trajectories="interpolated", 
                     video_types=["color", "point_cloud_acc"],
                     start_frame=None, end_frame=None):
    """
    Create a combined video with two videos aligned vertically, maintaining original dimensions
    Args:
        base_path: Base path
        dataset: Dataset name
        scene: Scene name
        method: Method name
        model: Model name
        pointmap: Point cloud type
        feature: Feature name
        dataset_split_json: Dataset split JSON file path
        output_dir: Output directory
        trajectories: Trajectory type
        video_types: List of video types to merge
        start_frame: Starting frame (optional)
        end_frame: Ending frame (optional)
    """
    print(f"Start processing video pair: {feature}")
    
    # Get number of views
    n_views = get_train_view_count(dataset_split_json, dataset, scene)
    print(f"Number of training views: {n_views}")

    # Build video path list
    video_paths = []
    for video_type in video_types:
        video_path = os.path.join(
            base_path, dataset, scene, 
            f"{n_views}_views",
            f"{method}-{model}_n10" if dataset == "DTU" else f"{method}-{model}",
            "gt" if dataset == "DTU" else pointmap,
            feature,
            "videos",
            f"{scene}_{n_views}_view_{trajectories}.mp4"
        )
        if video_type not in ["color"]:
            video_path = video_path.replace(".mp4", f"_{video_type}.mp4")
        video_paths.append(video_path)
    
    # Read all videos
    print(f"\nReading video files...")
    videos = []
    for i, path in enumerate(video_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file does not exist: {path}")
            
        container = av.open(path)
        frames = []
        frame_count = 0

        for frame in container.decode(video=0):
            if start_frame is not None and frame_count < start_frame:
                frame_count += 1
                continue
            if end_frame is not None and frame_count > end_frame:
                break

            # Use original frame directly, without scaling
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
            frame_count += 1
            
        videos.append(frames)
        container.close()

    # Ensure two videos have the same frame count
    min_frames = min(len(video) for video in videos)
    if start_frame is not None and end_frame is not None:
        min_frames = min(min_frames, end_frame - start_frame + 1)

    # Get video dimensions
    video_width = videos[0][0].shape[1]  # Original width
    single_height = videos[0][0].shape[0]  # Single video height
    
    # Set output path
    output_path = os.path.join(output_dir, f"{method}-{model}", "paired")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{feature}_{dataset}_{scene}_{trajectories}.mp4")
    
    # Create output video
    container = av.open(output_file, mode='w')
    stream = container.add_stream('h264', rate=30)
    stream.width = video_width  # Use original width
    stream.height = single_height * 2  # Sum of two videos' heights
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'crf': '23',
        'preset': 'slow',
        'profile': 'high',
        'level': '4.2'
    }

    # Process each frame
    print("\nStarting video synthesis...")
    for frame_idx in tqdm(range(min_frames), desc=f"Processing {feature} video pair"):
        # Stack vertically two videos' current frame
        combined_frame = np.vstack([
            videos[0][frame_idx],
            videos[1][frame_idx]
        ])
        
        # Encode and write
        av_frame = av.VideoFrame.from_ndarray(combined_frame, format='rgb24')
        packets = stream.encode(av_frame)
        for packet in packets:
            container.mux(packet)

    # Complete video writing
    packets = stream.encode(None)
    for packet in packets:
        container.mux(packet)
    container.close()

    print(f"Video saved to: {output_file}")

if __name__ == "__main__":
    ERROR_LOG_DIR = "/home/chenyue/tmp/error"
    os.makedirs(ERROR_LOG_DIR, exist_ok=True)
    
    BASE_PATH = "/home/chenyue/output/Feat2gs/output/eval"
    DATASET_SPLIT_JSON = "/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json"
    OUTPUT_DIR = "/home/chenyue/output/Feat2gs/video"

    pairs = {
        "spiral": {
            "DTU": ["scan1",],
        },
    }

    method = "feat2gs"
    model = "G"
    pointmap = "dust3r"
    
    features = [
        "radio", "mast3r", "dust3r", "midas_l16",      # First row
        "dino_b16", "dinov2_b14", "sam_base", "clip_b16",  # Second row
        "mae_b16", "dift", "iuvrgb"    # Third row
    ]
    
    video_types = ["point_cloud_acc", "color", ]
    
    for trajectories in pairs.keys():
        for dataset in pairs[trajectories].keys():
            for scene in pairs[trajectories][dataset]:
                for feature in features:
                    print(f"\n=== Start processing {feature} video pair ===")
                    try:
                        create_video_pair(
                            base_path=BASE_PATH,
                            dataset=dataset,
                            scene=scene,
                            method=method,
                            model=model,
                            pointmap=pointmap,
                            feature=feature,
                            dataset_split_json=DATASET_SPLIT_JSON,
                            output_dir=OUTPUT_DIR,
                            trajectories=trajectories,
                            video_types=video_types,
                            start_frame=0,
                            end_frame=None
                        )
                        print(f"=== Video pair processing completed for {feature} ===")
                    except Exception as e:
                        print(f"\nError processing {feature}: {str(e)}")
                        error_file = os.path.join(
                            ERROR_LOG_DIR, 
                            f"video_pair_{dataset}_{scene}_{method}-{model}_{feature}_{trajectories}.txt"
                        )
                        with open(error_file, 'w') as f:
                            f.write(f"Error time: {datetime.datetime.now()}\n")
                            f.write(f"Dataset: {dataset}\n")
                            f.write(f"Scene: {scene}\n")
                            f.write(f"Feature: {feature}\n")
                            f.write(f"Method: {method}-{model}\n")
                            f.write(f"Trajectory: {trajectories}\n")
                            f.write(f"Error message: {str(e)}\n")
                            f.write(f"Detailed traceback:\n{traceback.format_exc()}")
                        print(f"Error information saved to: {error_file}")
                        continue