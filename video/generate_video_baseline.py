import av
import numpy as np
import json
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import datetime
import traceback

def get_train_view_count(dataset_split_json, dataset, scene):
    with open(dataset_split_json, 'r') as f:
        data = json.load(f)
    return len(data[dataset][scene]["train"])

def create_video_grid(base_path, dataset, scene, trajectories, video_type,
                     dataset_split_json, output_dir, save_frames=False,
                     save_frames_per_video=False, start_frame=None, end_frame=None):
    print("Starting to process video grid...")
    
    # Layout parameters
    margin = 10
    n_cols = 4
    target_total_width = 1920
    
    # Calculate single video width
    total_margin_width = margin * (n_cols - 1)
    target_width = (target_total_width - total_margin_width) // n_cols
    
    # Get number of views
    n_views = get_train_view_count(dataset_split_json, dataset, scene)
    print(f"Number of training views: {n_views}")

    # Define 4 video relative paths
    relative_paths = [
        f"feat2gs-G/dust3r/radio/videos/{scene}_{n_views}_view_{trajectories}.mp4",
        f"feat2gs-G/dust3r/dust3r-mast3r-dift-dino_b16-dinov2_b14-radio-clip_b16-mae_b16-midas_l16-sam_base-iuvrgb/videos/{scene}_{n_views}_view_{trajectories}.mp4",
        f"feat2gs-Gft/dust3r/dust3r/videos/{scene}_{n_views}_view_{trajectories}.mp4",
        f"instantsplat/dust3r/videos/{scene}_{n_views}_view_{trajectories}.mp4"
    ]
    
    # Build full paths
    video_paths = []
    frame_base_name = []
    common_path = os.path.join(base_path, dataset, scene, f"{n_views}_views")
    
    for i, rel_path in enumerate(relative_paths):
        video_path = os.path.join(common_path, rel_path)
        if video_type not in ["color"]:
            video_path = video_path.replace(".mp4", f"_{video_type}.mp4")
        video_paths.append(video_path)
        
        if save_frames_per_video:
            frame_save_path = os.path.join(output_dir, video_type, "per_video",
                f"{dataset}_{scene}_{trajectories}_{video_type}")
            frame_base_name.append(f"video_{i}")
            os.makedirs(frame_save_path, exist_ok=True)

    # Display names
    display_names = ["Feat2GS w/ RADIO", "Feat2GS w/ concat all", "Feat2GS w/ DUSt3R*", "InstantSplat"]

    # Read all videos
    print(f"\nReading {len(video_paths)} video files...")
    videos = []
    max_height = 0

    for i, path in enumerate(tqdm(video_paths, desc="Reading videos")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file does not exist: {path}")
        print(f"\nReading video {i+1}/{len(video_paths)}: {os.path.basename(path)}")

        container = av.open(path)
        video_stream = container.streams.video[0]
        frames = []
        frame_count = 0

        for frame in container.decode(video=0):
            if start_frame is not None and frame_count < start_frame:
                frame_count += 1
                continue
            if end_frame is not None and frame_count > end_frame:
                break

            img = frame.to_ndarray(format='rgb24')
            
            if save_frames_per_video:
                if frame_count >= start_frame:
                    frame_file = os.path.join(frame_save_path, f"{frame_base_name[i]}_{frame_count:05d}.png")
                    Image.fromarray(img).save(frame_file)

            current_height, current_width = img.shape[:2]
            scale = target_width / current_width
            target_height = int(current_height * scale)
            max_height = max(max_height, target_height)
            
            resized_frame = cv2.resize(img, (target_width, target_height))
            frame_count += 1
            frames.append(resized_frame)
        
        videos.append(frames)
        container.close()

    # Ensure all video frames are the same
    min_frames = min(len(video) for video in videos)
    if start_frame is not None and end_frame is not None:
        min_frames = min(min_frames, end_frame - start_frame + 1)

    print(f"\nMinimum frames across all videos: {min_frames}")
    print(f"Maximum video height: {max_height}")
    
    # Add space for text
    text_height = 50
    grid_height = max_height + text_height + margin
    
    # Prepare font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 32)
        except:
            print("Warning: System fonts not found, using default font")
            font = ImageFont.load_default()

    # Set output path
    output_path = os.path.join(output_dir, video_type)
    base_filename = f"{dataset}_{scene}_{trajectories}_{video_type}"
    
    if start_frame is not None and end_frame is not None:
        base_filename += f"_frame_{start_frame}_{end_frame}"
    
    output_path = os.path.join(output_path, f"{base_filename}.mp4")

    if save_frames:
        frame_output_dir = os.path.join(os.path.dirname(output_path), "frames", base_filename)
        os.makedirs(frame_output_dir, exist_ok=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)   
    
    # Configure output container
    container = av.open(output_path, mode='w')
    stream = container.add_stream('h264', rate=30)
    stream.width = target_total_width
    stream.height = grid_height
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'crf': '23',
        'preset': 'slow',
        'profile': 'high',
        'level': '4.2',
        'x264-params': 'keyint=60:min-keyint=60'
    }

    print("\nStarting to process frames...")
    for frame_idx in tqdm(range(min_frames), desc="Processing frames"):
        current_frames = [video[frame_idx] for video in videos]
        grid = np.full((grid_height, target_total_width, 3), 255, dtype=np.uint8)
        
        for i, frame in enumerate(current_frames):
            x_start = i * (target_width + margin)
            
            frame_height = frame.shape[0]
            y_offset = (max_height - frame_height) // 2
            
            grid[y_offset:y_offset+frame_height, x_start:x_start+target_width] = frame
            
            text = display_names[i]
            text_img = Image.fromarray(grid)
            draw = ImageDraw.Draw(text_img)
            
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x_start + (target_width - text_width) // 2
            text_y = max_height + 5
            
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
            grid = np.array(text_img)
        
        if save_frames:
            frame_filename = os.path.join(frame_output_dir, f"frame_{frame_idx:05d}.png")
            Image.fromarray(grid).save(frame_filename, format='PNG', optimize=True)

        av_frame = av.VideoFrame.from_ndarray(grid, format='rgb24')
        packets = stream.encode(av_frame)
        for packet in packets:
            container.mux(packet)

    packets = stream.encode(None)
    for packet in packets:
        container.mux(packet)

    container.close()
    print(f"\n✓ Video successfully saved to: {output_path}")

if __name__ == "__main__":
    ERROR_LOG_DIR = "/home/chenyue/tmp/error"
    os.makedirs(ERROR_LOG_DIR, exist_ok=True)
    
    BASE_PATH = "/home/chenyue/output/Feat2gs/output/eval"
    DATASET_SPLIT_JSON = "/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json"
    OUTPUT_DIR = "/home/chenyue/output/Feat2gs/video_baseline"

    pairs = {
        "interpolated": {
            "DL3DV": ["Center", "Electrical", "Museum", "Supermarket2", "Temple"],
            "Infer": ["erhai", "paper4"],
            "MVimgNet": ["bench", "suv", "car", "plushies"],
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
            "Infer": ["paper", "plushies"],
            "LLFF": ["horns"],
            "Tanks": ["Auditorium", "Ignatius", "Caterpillar"]
        },
        "wander": {
            "Infer": ["erhai"],
        }
    }

    video_types = ["color", "depth", "normal"]

    for trajectories in pairs.keys():
        for dataset in pairs[trajectories].keys():
            for scene in pairs[trajectories][dataset]:
                for video_type in video_types:
                    print(f"\n=== Starting Video Grid Generation ===")
                    print(f"Dataset: {dataset}")
                    print(f"Scene: {scene}")
                    print(f"Trajectory: {trajectories}")
                    print(f"Video type: {video_type}")
                    
                    try:
                        create_video_grid(
                            base_path=BASE_PATH,
                            dataset=dataset,
                            scene=scene,
                            trajectories=trajectories,
                            video_type=video_type,
                            dataset_split_json=DATASET_SPLIT_JSON,
                            output_dir=OUTPUT_DIR,
                            # save_frames=True,
                            # start_frame=220,
                            # end_frame=230,
                            # save_frames_per_video=True,
                        )
                        print("\n=== Video Grid Generation Complete ===")
                    except Exception as e:
                        print(f"\nError: {str(e)}")
                        error_file = os.path.join(
                            ERROR_LOG_DIR, 
                            f"video_{dataset}_{scene}_{trajectories}.txt"
                        )
                        with open(error_file, 'w') as f:
                            f.write(f"Error Time: {datetime.datetime.now()}\n")
                            f.write(f"Dataset: {dataset}\n")
                            f.write(f"Scene: {scene}\n")
                            f.write(f"Trajectory: {trajectories}\n")
                            f.write(f"Error Message: {str(e)}\n")
                            f.write(f"Video Type: {video_type}\n")
                            f.write(f"Detailed Traceback:\n{traceback.format_exc()}")
                        print(f"Error information has been saved to: {error_file}")
                        continue