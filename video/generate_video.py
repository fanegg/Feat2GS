import av
import imageio.v3 as iio
import numpy as np
import json
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
import datetime
import traceback

def get_train_view_count(dataset_split_json, dataset, scene):
    with open(dataset_split_json, 'r') as f:
        data = json.load(f)
    return len(data[dataset][scene]["train"])

def create_video_grid(base_path, dataset, scene, method, model, pointmap, features, display_names,
                     dataset_split_json, output_dir, trajectories="interpolated", video_type="color",
                     save_frames=True, save_frames_per_video=False, 
                     start_frame=None, end_frame=None):
    print("Starting to process video grid...")
    
    # Layout parameters
    margin = 10  # Pixel spacing between videos
    n_cols = 4   # Number of columns
    target_total_width = 1920  # Total width
    
    # Calculate target width for single video (considering margins)
    total_margin_width = margin * (n_cols - 1)  # Sum of all horizontal margins
    target_width = (target_total_width - total_margin_width) // n_cols  # Width of single video
    
    # Get number of views
    n_views = get_train_view_count(dataset_split_json, dataset, scene)
    print(f"Number of training views: {n_views}")

    # Build video path list
    video_paths = []
    frame_base_name = []
    for feature in features:
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

        if save_frames_per_video:
            frame_save_path = os.path.join(output_dir, f"{method}-{model}", video_type, "per_video",
                f"{dataset}_{scene}_{trajectories}_{video_type}")
            frame_base_name.append(feature)
            os.makedirs(frame_save_path, exist_ok=True)
            # print(f"Saving frames to: {frame_save_path}")
    
    # Read all videos
    print(f"\nReading {len(video_paths)} video files...")
    videos = []
    max_height = 0  # Track maximum height

    for i, path in enumerate(tqdm(video_paths, desc="Reading videos")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        print(f"\nReading video {i+1}/{len(video_paths)}: {os.path.basename(path)}")
        # video = iio.imread(path, plugin='pyav')
           
        # # Calculate scaling ratio, maintaining aspect ratio
        # current_height, current_width = video[0].shape[:2]
        # scale = target_width / current_width
        # target_height = int(current_height * scale)
        # max_height = max(max_height, target_height)  # Update maximum height
        
        # # Scale each frame
        # resized_video = []
        # for frame in tqdm(video, desc=f"Scaling video {i+1}"):
        #     resized_frame = cv2.resize(frame, (target_width, target_height))
        #     resized_video.append(resized_frame)
        
        # videos.append(resized_video)

        # Use PyAV to read video
        container = av.open(path)
        video_stream = container.streams.video[0]
    
        frames = []
        frame_count = 0

        for frame in container.decode(video=0):
            # If start frame is set, skip previous frames
            if start_frame is not None and frame_count < start_frame:
                frame_count += 1
                continue
            # If end frame is set, stop reading after reaching it
            if end_frame is not None and frame_count > end_frame:
                print(f"Reached frame {frame_count}, stopping reading")
                break

            # Convert to numpy array
            img = frame.to_ndarray(format='rgb24')
            
            if save_frames_per_video:
                if frame_count >= start_frame:
                    frame_file = os.path.join(frame_save_path, f"{frame_base_name[i]}_{frame_count:05d}.png")
                    Image.fromarray(img).save(frame_file)

            # Calculate scaling ratio
            current_height, current_width = img.shape[:2]
            scale = target_width / current_width
            target_height = int(current_height * scale)
            max_height = max(max_height, target_height)
            
            # Scale frame
            resized_frame = cv2.resize(img, (target_width, target_height))
            frame_count += 1
            frames.append(resized_frame)
        
        videos.append(frames)
        container.close()     
    
    # Ensure all videos have the same frame count
    min_frames = min(len(video) for video in videos)
    if start_frame is not None and end_frame is not None:
        min_frames = min(min_frames, end_frame - start_frame + 1)

    print(f"\nMinimum frame count across all videos: {min_frames}")
    print(f"Maximum video height: {max_height}")
    
    # Add space for text
    text_height = 50  # Space for text
    n_rows = (len(videos) + n_cols - 1) // n_cols  # Round up to get number of rows
    grid_height = (max_height + text_height + margin) * n_rows - margin  # Use maximum height
    
    # Prepare font - use larger font
    try:
        # Try system font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        try:
            # Alternative system font path
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 32)
        except:
            print("Warning: System font not found, using default font")
            font = ImageFont.load_default()

    # Set output path
    output_path = os.path.join(output_dir, f"{method}-{model}", video_type)
    
    # Base filename
    base_filename = f"{dataset}_{scene}_{trajectories}_{video_type}"
    
    # If frame range is specified, add frame range information to filename
    if start_frame is not None and end_frame is not None:
        base_filename += f"_frame_{start_frame}_{end_frame}"
    
    output_path = os.path.join(output_path, f"{base_filename}.mp4")

    # Create frame output directory (if needed)
    if save_frames:
        frame_output_dir = os.path.join(os.path.dirname(output_path), "frames", base_filename)
        os.makedirs(frame_output_dir, exist_ok=True)

    # Create video output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   
    
    # Configure output container
    container = av.open(output_path, mode='w')
    stream = container.add_stream('h264', rate=30)
    stream.width = target_total_width
    stream.height = grid_height
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'crf': '23',          # Quality control (0-51, smaller is better)
        'preset': 'slow',     # Encoding speed
        'profile': 'high',    # Encoding profile
        'level': '4.2',       # Compatibility level
        'x264-params': 'keyint=60:min-keyint=60'  # Keyframe interval
    }

    # Create output video frames
    print("\nStarting frame processing...")
    output_frames = []
    for frame_idx in tqdm(range(min_frames), desc="Processing frames"):
        # Get current frame
        current_frames = [video[frame_idx] for video in videos]
        
        # Create white grid
        grid = np.full((grid_height, target_total_width, 3), 255, dtype=np.uint8)
        
        # Fill grid (considering margins)
        for i, frame in enumerate(current_frames):
            row = i // n_cols
            col = i % n_cols
            
            # Calculate position (including margins)
            x_start = col * (target_width + margin)
            y_start = row * (max_height + text_height + margin)
            
            # Place current frame vertically centered in grid
            frame_height = frame.shape[0]
            y_offset = (max_height - frame_height) // 2
            y_pos = y_start + y_offset
            
            # Copy frame
            grid[y_pos:y_pos+frame_height, x_start:x_start+target_width] = frame
            
            # Add text
            text = display_names[i]
            text_img = Image.fromarray(grid)
            draw = ImageDraw.Draw(text_img)
            
            # Get text size and calculate center position
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x_start + (target_width - text_width) // 2
            text_y = y_start + max_height + 5  # Text position below video
            
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
            grid = np.array(text_img)
        
        # output_frames.append(grid)
        
        # Save current frame (if needed)
        if save_frames:
            frame_filename = os.path.join(frame_output_dir, f"frame_{frame_idx:05d}.png")
            Image.fromarray(grid).save(
                frame_filename,
                format='PNG',
                optimize=True
            )

        # Convert to PyAV frame and encode
        av_frame = av.VideoFrame.from_ndarray(grid, format='rgb24')
        packets = stream.encode(av_frame)
        for packet in packets:
            container.mux(packet)

    # Flush remaining frames
    packets = stream.encode(None)
    for packet in packets:
        container.mux(packet)

    # Close container
    container.close()

    # Save video
    print("\nSaving final video...")

    print(f"\n✓ Video successfully saved to: {output_path}")

# Usage example
if __name__ == "__main__":
    ERROR_LOG_DIR = "/home/chenyue/tmp/error"
    os.makedirs(ERROR_LOG_DIR, exist_ok=True)
    
    BASE_PATH = "/home/chenyue/output/Feat2gs/output/eval"
    DATASET_SPLIT_JSON = "/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json"
    OUTPUT_DIR = "/home/chenyue/output/Feat2gs/video"

    # pairs = {
    #     "spiral": {
    #         "DTU": ["scan1", "scan48"],
    #     },
    # }

    pairs = {
        "interpolated": {
            "DL3DV": ["Center", "Electrical", "Museum", "Supermarket2", "Temple"],
            "Infer": ["erhai", "paper4"],
            "MVimgNet": ["bench", "suv", "car"],
            "Tanks": ["Train"]
        },
        "lemniscate": {
            "Infer": ["cy", "bread", "brunch", "paper4", "plushies"],
            "LLFF": ["fortress", "horns", "orchids", "trex", "room"],
            "MipNeRF360": ["room", "garden"],
            "MVimgNet": ["bench"],
            "Tanks": ["Family", "Auditorium", "Ignatius"]
        },
        "spiral": {
            "Infer": ["cy"],
            "LLFF": ["orchids", "trex", "fortress"]
        },
        "ellipse": {
            "Infer": ["bread", "brunch"],
            "MipNeRF360": ["kitchen"]
        },
        "arc": {
            "Infer": ["paper"],
            "LLFF": ["horns"],
            "Tanks": ["Auditorium", "Ignatius", "Caterpillar"]
        },
        "wander": {
            "Infer": ["erhai"],
        }
    }

    method = "feat2gs"
    models = ["G", ]
    pointmap = "dust3r"
    
    # Folder name and display name mapping
    features = [
        "radio", "mast3r", "dust3r", "midas_l16",      # First row
        "dino_b16", "dinov2_b14", "sam_base", "clip_b16",  # Second row
        "mae_b16", "dift", "iuvrgb"    # Third row
    ]
    
    display_names = [
        "RADIO", "MASt3R", "DUSt3R", "MiDaS",      # First row
        "DINO", "DINOv2", "SAM", "CLIP",  # Second row
        "MAE", "SD", "IUVRGB"    # Third row
    ]
    
 
    # video_types = ["gt", "gt_comp", "noised_gt", "point_cloud", "point_cloud_acc", "point_cloud_dist", "color"]
    # video_types = ["color", "depth", "normal"]
    video_types = ["color",]
    for model in models:
        for trajectories in pairs.keys():
            for dataset in pairs[trajectories].keys():
                for scene in pairs[trajectories][dataset]:
                    for video_type in video_types:
                        print("\n=== Starting Video Grid Generation ===")
                        print(f"Dataset: {dataset}")
                        print(f"Scene: {scene}")
                        print(f"Method: {method}-{model}")
                        print(f"Number of features: {len(features)}")
                        print(f"Trajectory: {trajectories}")
                        print(f"Video type: {video_type}")
                        
                        try:
                            create_video_grid(
                                base_path=BASE_PATH,
                                dataset=dataset,
                                scene=scene,
                                method=method,
                                model=model,
                                pointmap=pointmap,
                                features=features,
                                display_names=display_names,
                                dataset_split_json=DATASET_SPLIT_JSON,
                                output_dir=OUTPUT_DIR,
                                trajectories=trajectories,
                                video_type=video_type,
                                # save_frames=False,  # Whether to save separate frames
                                # start_frame=0,      # Set start frame (optional)
                                # end_frame=0,       # Set end frame (optional)
                                # save_frames_per_video=False,
                            )
                            print("\n=== Video Grid Generation Complete ===")
                        except Exception as e:
                            print(f"\nError: {str(e)}")
                            # Modify error log filename format
                            error_file = os.path.join(
                                ERROR_LOG_DIR, 
                                f"video_{dataset}_{scene}_{method}-{model}_{trajectories}.txt"
                            )
                            with open(error_file, 'w') as f:
                                f.write(f"Error time: {datetime.datetime.now()}\n")
                                f.write(f"Dataset: {dataset}\n")
                                f.write(f"Scene: {scene}\n")
                                f.write(f"Method: {method}-{model}\n")
                                f.write(f"Trajectory: {trajectories}\n")
                                f.write(f"Error message: {str(e)}\n")
                                f.write(f"Video type: {video_type}\n")
                                f.write(f"Detailed traceback:\n{traceback.format_exc()}")
                            print(f"Error information saved to: {error_file}")
                            continue