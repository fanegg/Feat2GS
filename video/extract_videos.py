import os
import json
import shutil
from tqdm import tqdm
import datetime
import traceback

def get_train_view_count(dataset_split_json, dataset, scene):
    with open(dataset_split_json, 'r') as f:
        data = json.load(f)
    return len(data[dataset][scene]["train"])

def extract_videos(base_path, dataset, scene, method, model, pointmap, features,
                  dataset_split_json, output_dir, trajectories="interpolated", video_type="color"):
    print("Starting video extraction...")
    
    # Get number of views
    n_views = get_train_view_count(dataset_split_json, dataset, scene)
    print(f"Number of training views: {n_views}")

    # Build output directory
    output_subdir = os.path.join(output_dir, f"{method}-{model}", video_type, "raw_videos",
                                f"{dataset}_{scene}_{trajectories}_{video_type}")
    os.makedirs(output_subdir, exist_ok=True)

    # Copy video files
    for feature in tqdm(features, desc="Copying videos"):
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
            
        if not os.path.exists(video_path):
            print(f"Warning: Video file does not exist: {video_path}")
            continue
            
        # Build target filename
        dest_filename = f"{feature}.mp4"
        if video_type not in ["color"]:
            dest_filename = f"{feature}_{video_type}.mp4"
            
        dest_path = os.path.join(output_subdir, dest_filename)
        shutil.copy2(video_path, dest_path)

    # Create zip file
    zip_name = f"{dataset}_{scene}_{trajectories}_{video_type}"
    shutil.make_archive(
        os.path.join(output_dir, f"{method}-{model}", video_type, zip_name),
        'zip',
        output_subdir
    )
    
    print(f"\n✓ Videos successfully extracted and packed to: {zip_name}.zip")

if __name__ == "__main__":
    ERROR_LOG_DIR = "/home/chenyue/tmp/error"
    os.makedirs(ERROR_LOG_DIR, exist_ok=True)
    
    BASE_PATH = "/home/chenyue/output/Feat2gs/output/eval"
    DATASET_SPLIT_JSON = "/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json"
    OUTPUT_DIR = "/home/chenyue/output/Feat2gs/video"

    pairs = {
        "interpolated": {
            # "DL3DV": ["Museum"],
            "MVimgNet": ["suv"],
            "Tanks": ["Train"]
        },
        "arc": {
            "Tanks": ["Caterpillar"]
        },
        "wander": {
            "Infer": ["erhai"],
        }
    }

    method = "feat2gs"
    models = ["G"]
    pointmap = "dust3r"
    
    features = [
        "radio", "mast3r", "dust3r", "midas_l16",
        "dino_b16", "dinov2_b14", "sam_base", "clip_b16",
        "mae_b16", "dift", "iuvrgb"
    ]
    
    video_types = ["color"]

    for model in models:
        for trajectories in pairs.keys():
            for dataset in pairs[trajectories].keys():
                for scene in pairs[trajectories][dataset]:
                    for video_type in video_types:
                        print(f"\n=== Starting Video Extraction ===")
                        print(f"Dataset: {dataset}")
                        print(f"Scene: {scene}")
                        print(f"Method: {method}-{model}")
                        print(f"Trajectory: {trajectories}")
                        print(f"Video Type: {video_type}")
                        
                        try:
                            extract_videos(
                                base_path=BASE_PATH,
                                dataset=dataset,
                                scene=scene,
                                method=method,
                                model=model,
                                pointmap=pointmap,
                                features=features,
                                dataset_split_json=DATASET_SPLIT_JSON,
                                output_dir=OUTPUT_DIR,
                                trajectories=trajectories,
                                video_type=video_type
                            )
                            print("\n=== Video Extraction Complete ===")
                        except Exception as e:
                            print(f"\nError: {str(e)}")
                            error_file = os.path.join(
                                ERROR_LOG_DIR, 
                                f"extract_{dataset}_{scene}_{method}-{model}_{trajectories}.txt"
                            )
                            with open(error_file, 'w') as f:
                                f.write(f"Error Time: {datetime.datetime.now()}\n")
                                f.write(f"Dataset: {dataset}\n")
                                f.write(f"Scene: {scene}\n")
                                f.write(f"Method: {method}-{model}\n")
                                f.write(f"Trajectory: {trajectories}\n")
                                f.write(f"Error Message: {str(e)}\n")
                                f.write(f"Video Type: {video_type}\n")
                                f.write(f"Detailed Traceback:\n{traceback.format_exc()}")
                            print(f"Error information saved to: {error_file}")
                            continue