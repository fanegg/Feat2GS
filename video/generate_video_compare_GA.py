import av
import numpy as np
import json
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import datetime
import traceback
from pathlib import Path

def get_train_view_count(dataset_split_json, dataset, scene):
    with open(dataset_split_json, 'r') as f:
        data = json.load(f)
    return len(data[dataset][scene]["train"])

def create_video_compare(base_path, dataset, scene, method, feature,
                     dataset_split_json, output_dir, trajectories="interpolated", 
                     video_type="color", start_frame=None, end_frame=None):
    """
    Create comparison video with model="G" video on top and model="A" video at bottom
    Args:
        base_path: Base path
        dataset: Dataset name
        scene: Scene name
        method: Method name
        feature: Feature name
        dataset_split_json: Dataset split JSON file path
        output_dir: Output directory
        trajectories: Trajectory type
        video_type: Video type
        start_frame: Start frame (optional)
        end_frame: End frame (optional)
    """
    print(f"Starting to process comparison video for feature {feature}...")
    
    # Get number of views
    n_views = get_train_view_count(dataset_split_json, dataset, scene)
    print(f"Number of training views: {n_views}")

    # Build video paths for two models
    video_paths = []
    for model in ["G", "A"]:
        video_path = os.path.join(
            base_path, dataset, scene, 
            f"{n_views}_views",
            f"{method}-{model}_n10" if dataset == "DTU" else f"{method}-{model}",
            "gt" if dataset == "DTU" else "dust3r",
            feature,
            "videos",
            f"{scene}_{n_views}_view_{trajectories}.mp4"
        )
        if video_type not in ["color"]:
            video_path = video_path.replace(".mp4", f"_{video_type}.mp4")
        video_paths.append(video_path)
    
    # Read videos
    print(f"\nReading video files...")
    videos = []
    for i, path in enumerate(video_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file does not exist: {path}")
            
        print(f"Reading video {i+1}/2: {os.path.basename(path)}")
        container = av.open(path)
        frames = []
        frame_count = 0

        for frame in container.decode(video=0):
            if start_frame is not None and frame_count < start_frame:
                frame_count += 1
                continue
            if end_frame is not None and frame_count > end_frame:
                break

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
    video_width = videos[0][0].shape[1]
    single_height = videos[0][0].shape[0]
    text_height = 100  # Increase text area height
    
    # Prepare font - use larger font size
    font_size = 64  # Increase font size
    try:
        # First try Comic Sans MS
        font = ImageFont.truetype("Comic Sans MS", font_size)
    except:
        try:
            # Try alternative path for Comic Sans MS
            font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf", font_size)
        except:
            try:
                # Windows path
                font = ImageFont.truetype("C:\\Windows\\Fonts\\Comic.ttf", font_size)
            except:
                try:
                    # Fallback to DejaVu Sans
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    try:
                        # Fallback to Arial
                        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", font_size)
                    except:
                        print("Warning: Required fonts not found, using default font")
                        font = ImageFont.load_default()
    
    # Set output path
    output_path = os.path.join(output_dir, f"{method}_compare", "paired")
    os.makedirs(output_path, exist_ok=True)
    
    # Build output filename
    base_filename = f"{feature}_{dataset}_{scene}_{trajectories}"
    if start_frame is not None and end_frame is not None:
        base_filename += f"_frame_{start_frame}_{end_frame}"
    output_file = os.path.join(output_path, f"{base_filename}.mp4")
    
    # Create output video
    container = av.open(output_file, mode='w')
    stream = container.add_stream('h264', rate=30)
    stream.width = video_width
    stream.height = (single_height + text_height) * 2  # Total height including text area
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'crf': '23',          # Quality control (0-51, smaller is better)
        'preset': 'slow',     # Encoding speed
        'profile': 'high',    # Encoding profile
        'level': '4.2',        # Compatibility level
        'x264-params': 'keyint=60:min-keyint=60'  # Keyframe interval
    }

    # Process each frame
    print("\nStarting video composition...")
    for frame_idx in tqdm(range(min_frames), desc="Processing frames"):
        # Create canvas with text area
        combined_frame = np.full(((single_height + text_height) * 2, video_width, 3), 255, dtype=np.uint8)
        
        # Add two video frames
        combined_frame[text_height:text_height+single_height] = videos[0][frame_idx]  # G model
        combined_frame[text_height+single_height+text_height:] = videos[1][frame_idx]  # A model
        
        # Convert to PIL image for adding text
        pil_image = Image.fromarray(combined_frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Add text labels
        labels = ["Geometry", "Geometry+Texture"]
        for i, label in enumerate(labels):
            # Calculate text position to center
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (video_width - text_width) // 2
            text_y = i * (single_height + text_height) + (text_height - (text_bbox[3] - text_bbox[1])) // 2  # Vertical center
            draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
        
        # Convert back to numpy array and encode
        combined_frame = np.array(pil_image)
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

    # Define dataset and scene pairs
    pairs = {
        "arc": {
            "Tanks": ["Caterpillar"]
        },
        "lemniscate": {
            "MVimgNet": ["bench"],
            "Tanks": ["Ignatius"]
        }
    }

    method = "feat2gs"
    
    # Define feature list
    features = [
        "radio", "mast3r", "dust3r", "midas_l16",      # First row
        "dino_b16", "dinov2_b14", "sam_base", "clip_b16",  # Second row
        "mae_b16", "dift", "iuvrgb"    # Third row
    ]
    
    # Video types
    video_types = ["color"]
    
    # Process each combination
    for trajectories in pairs.keys():
        for dataset in pairs[trajectories].keys():
            for scene in pairs[trajectories][dataset]:
                for video_type in video_types:
                    for feature in features:
                        print(f"\n=== Starting to process comparison video for {feature} ===")
                        print(f"Dataset: {dataset}")
                        print(f"Scene: {scene}")
                        print(f"Feature: {feature}")
                        print(f"Trajectory: {trajectories}")
                        print(f"Video type: {video_type}")
                        
                        try:
                            create_video_compare(
                                base_path=BASE_PATH,
                                dataset=dataset,
                                scene=scene,
                                method=method,
                                feature=feature,
                                dataset_split_json=DATASET_SPLIT_JSON,
                                output_dir=OUTPUT_DIR,
                                trajectories=trajectories,
                                video_type=video_type,
                                start_frame=None,
                                end_frame=None
                            )
                            print(f"=== Comparison video processing completed for {feature} ===")
                        except Exception as e:
                            print(f"\nError processing {feature}: {str(e)}")
                            error_file = os.path.join(
                                ERROR_LOG_DIR, 
                                f"video_compare_{dataset}_{scene}_{method}_{feature}_{trajectories}.txt"
                            )
                            with open(error_file, 'w') as f:
                                f.write(f"Error time: {datetime.datetime.now()}\n")
                                f.write(f"Dataset: {dataset}\n")
                                f.write(f"Scene: {scene}\n")
                                f.write(f"Feature: {feature}\n")
                                f.write(f"Method: {method}\n")
                                f.write(f"Trajectory: {trajectories}\n")
                                f.write(f"Video type: {video_type}\n")
                                f.write(f"Error message: {str(e)}\n")
                                f.write(f"Detailed traceback:\n{traceback.format_exc()}")
                            print(f"Error information saved to: {error_file}")
                            continue