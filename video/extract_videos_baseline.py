import os
import datetime
import traceback
import cv2
import numpy as np
import json
import av
from PIL import Image, ImageDraw, ImageFont

def get_train_view_count(dataset_split_json, dataset, scene):
    with open(dataset_split_json, 'r') as f:
        data = json.load(f)
    return len(data[dataset][scene]["train"])

def add_text_with_background(image, text, position, is_right=False):
    """Add text label with semi-transparent background"""
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 32)
        except:
            print("Warning: System font not found, using default font")
            font = ImageFont.load_default()
    
    # Create PIL image to get text size
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate background rectangle position and size
    padding = 20
    rect_width = text_width + 2 * padding
    rect_height = text_height + 2 * padding
    
    if is_right:
        rect_x = position[0] - rect_width
        text_x = position[0] - text_width - padding
    else:
        rect_x = position[0]
        text_x = position[0] + padding
    rect_y = position[1]
    text_y = position[1] + padding

    # Create semi-transparent background (in OpenCV image)
    sub_img = image[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]
    rect_img = np.zeros(sub_img.shape, dtype=np.uint8)
    rect_img[:] = (64, 64, 64)  # Gray background
    alpha = 0.6  # Transparency

    # Mix background
    image[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width] = \
        cv2.addWeighted(sub_img, 1-alpha, rect_img, alpha, 0)
    
    # Convert updated OpenCV image to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Use PIL to draw text
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
    
    # Convert back to OpenCV format
    image[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def concat_videos_horizontally(video1_path, video2_path, output_path):
    """Horizontally concatenate two videos and add labels"""
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Get video information
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap1.get(cv2.CAP_PROP_FPS)))
    
    # Create output container
    container = av.open(output_path, mode='w')
    stream = container.add_stream('h264', rate=fps)
    stream.width = width1 + width2
    stream.height = max(height1, height2)
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'crf': '23',          # Control quality (lower value means higher quality, 23 is a good default)
        'preset': 'slow',     # Slower encoding speed but better compression effect
        'profile': 'high',    # Use high specification configuration
        'level': '4.2',       # Compatible level setting
        'x264-params': 'keyint=60:min-keyint=60'  # Keyframe interval setting
    }
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        # Horizontally concatenate frames
        combined_frame = np.hstack((frame1, frame2))
        
        # Add labels
        margin = 20
        add_text_with_background(combined_frame, "Our baseline", (margin, margin), False)
        add_text_with_background(combined_frame, "InstantSplat", (width1 + width2 - margin, margin), True)
        
        # Convert to av.VideoFrame and encode
        av_frame = av.VideoFrame.from_ndarray(combined_frame, format='bgr24')
        packets = stream.encode(av_frame)
        for packet in packets:
            container.mux(packet)
    
    # Flush remaining frames
    packets = stream.encode(None)
    for packet in packets:
        container.mux(packet)
    
    # Release resources
    cap1.release()
    cap2.release()
    container.close()

def pack_videos(base_path, dataset, scene, trajectories,
                dataset_split_json, output_dir):
    print(f"Start processing videos...")
    
    # Get number of views
    n_views = get_train_view_count(dataset_split_json, dataset, scene)
    print(f"Number of training views: {n_views}")

    # Define video relative paths
    video_configs = {
        "dust3r": f"feat2gs-Gft/dust3r/dust3r/videos/{scene}_{n_views}_view_{trajectories}.mp4",
        "instantsplat": f"instantsplat/dust3r/videos/{scene}_{n_views}_view_{trajectories}.mp4"
    }
    
    # Build full path
    common_path = os.path.join(base_path, dataset, scene, f"{n_views}_views")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video type
    for vtype in ["color", "depth", "normal"]:
        video_paths = []
        for config_name, rel_path in video_configs.items():
            video_path = os.path.join(common_path, rel_path)
            if vtype != "color":
                video_path = video_path.replace(".mp4", f"_{vtype}.mp4")
            
            if not os.path.exists(video_path):
                print(f"Warning: Video file does not exist: {video_path}")
                continue
            video_paths.append(video_path)
        
        if len(video_paths) != 2:
            print(f"Warning: Not enough videos found for concatenation: {vtype}")
            continue
            
        # Build output file name
        output_name = f"{dataset}_{scene}_{trajectories}"
        if vtype != "color":
            output_name += f"_{vtype}"
        output_name += ".mp4"
        output_path = os.path.join(output_dir, output_name)
        
        # Horizontally concatenate videos
        concat_videos_horizontally(video_paths[0], video_paths[1], output_path)
        print(f"✓ Generated concatenated video: {output_name}")

if __name__ == "__main__":
    ERROR_LOG_DIR = "/home/chenyue/tmp/error"
    os.makedirs(ERROR_LOG_DIR, exist_ok=True)
    
    BASE_PATH = "/home/chenyue/output/Feat2gs/output/eval"
    DATASET_SPLIT_JSON = "/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json"
    OUTPUT_DIR = "/home/chenyue/output/Feat2gs/video_baseline"

    pairs = {
        "lemniscate": {
            "Tanks": ["Family"]
        },
        "arc": {
            "Infer": ["plushies"],
        },
    }

    for trajectories in pairs.keys():
        for dataset in pairs[trajectories].keys():
            for scene in pairs[trajectories][dataset]:
                print(f"\n=== Start Packing Videos ===")
                print(f"Dataset: {dataset}")
                print(f"Scene: {scene}")
                print(f"Trajectory: {trajectories}")
                
                try:
                    pack_videos(
                        base_path=BASE_PATH,
                        dataset=dataset,
                        scene=scene,
                        trajectories=trajectories,
                        dataset_split_json=DATASET_SPLIT_JSON,
                        output_dir=OUTPUT_DIR
                    )
                    print("\n=== Video Processing Complete ===")
                except Exception as e:
                    print(f"\nError: {str(e)}")
                    error_file = os.path.join(
                        ERROR_LOG_DIR, 
                        f"pack_video_{dataset}_{scene}_{trajectories}.txt"
                    )
                    with open(error_file, 'w') as f:
                        f.write(f"Error Time: {datetime.datetime.now()}\n")
                        f.write(f"Dataset: {dataset}\n")
                        f.write(f"Scene: {scene}\n")
                        f.write(f"Trajectory: {trajectories}\n")
                        f.write(f"Error Message: {str(e)}\n")
                        f.write(f"Detailed Traceback:\n{traceback.format_exc()}")
                    print(f"Error information has been saved to: {error_file}")
                    continue