import json
import shutil
from pathlib import Path
import glob
from preprocess_dtu import process_dtu
from PIL import Image

"""
root_dir built will follow the structure like:

Feat2GS_Dataset/
    dataset_split.json
    dataset_name/
        scene_name/
            3_views/
                images/
                    train_view1.png
                    train_view2.png
                    ...
            images/
                train_view1.png
                train_view2.png
                ...
                test_view1.png
                test_view2.png
                ...
            ## Only for DTU dataset, we need depths, cams and binary_masks to compute pointclouds GT
            binary_masks/
                train_view1.png
                train_view2.png
                ...
                test_view1.png
                test_view2.png
                ...
            cams/
                train_view1_cam.txt
                train_view2_cam.txt
                ...
                test_view1_cam.txt
                test_view2_cam.txt
                ...
            depths/
                train_view1.npy
                train_view2.npy
                ...
                test_view1.npy
                test_view2.npy
                ...
"""

root_dir = Path('/home/chenyue/dataset/Feat2GS_Dataset') # build dataset
base_path = Path('/home/chenyue/dataset/eval_tmp_data') # eval_tmp_data dir
DL3DV_source = Path('/home/chenyue/.cache/huggingface/hub/datasets--DL3DV--DL3DV-10K-Sample/snapshots/76acf288db94245ceead597dd89ebbdd5e11bc6c/')   # download DL3DV
DTU_source = Path('/home/chenyue/dataset/dtu_test_mvsnet_release/')   # download DL3DV

def get_image_format(directory):
    for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
        if list(Path(directory).glob(f'**/*{ext}')):
            return ext[1:]
    raise ValueError(f"No valid image format found in directory {directory}")

def create_dataset_structure():
    json_path = root_dir / 'dataset_split.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for dataset_name, scenes in data.items():
        
        if (root_dir / dataset_name).exists():
            print(f"Dataset {dataset_name} already exists, skipping processing")
            continue

        for scene_name, scene_data in scenes.items():
            scene_path = root_dir / dataset_name / scene_name
            images_path = scene_path / 'images'
            train_views_path = scene_path / f"{len(scene_data['train'])}_views" / 'images'
            images_path.mkdir(parents=True, exist_ok=True)
            train_views_path.mkdir(parents=True, exist_ok=True)

            if dataset_name in ['DTU']:
                (scene_path / 'depths').mkdir(parents=True, exist_ok=True)
                (scene_path / 'binary_masks').mkdir(parents=True, exist_ok=True)
                (scene_path / 'cams').mkdir(parents=True, exist_ok=True)

            src_dir = base_path / dataset_name / scene_name
            
            image_format = get_image_format(src_dir)
            all_images = glob.glob(str(src_dir / f'**/*.{image_format}'), recursive=True)

            for img_path in all_images:
                img_path = Path(img_path)
                view = img_path.stem
                if dataset_name in ['DL3DV']:
                    # For DL3DV, we preview 480p images in eval_tmp_data, 
                    # but build dataset from the 1920p images path
                    img_path = DL3DV_source / f'{scene_data.get("hash", scene_name)}/colmap/images_2/{view}.{image_format}'

                if dataset_name in ['DTU']:
                    # For DTU, we need depths, cams and binary_masks to compute pointclouds GT
                    dtu_base = DTU_source / scene_name
                    masked_img = Image.fromarray(process_dtu(img_path, dtu_base))
                    if view in set(scene_data['train'] + scene_data['test']):
                        masked_img.save(images_path / f"{view}.{image_format}")
                        if view in scene_data['train']:
                            masked_img.save(train_views_path / f"{view}.{image_format}")
            
                        src_depth = dtu_base / 'depths' / f'{view}.npy'
                        dst_depth = scene_path / 'depths' / f'{view}.npy'
                        if src_depth.exists():
                            shutil.copy2(src_depth, dst_depth)
                        
                        src_cam = dtu_base / 'cams' / f'{view}_cam.txt'
                        dst_cam = scene_path / 'cams' / f'{view}_cam.txt'
                        if src_cam.exists():
                            shutil.copy2(src_cam, dst_cam)
                        
                        src_mask = dtu_base / 'binary_masks' / f'{view}.png'
                        dst_mask = scene_path / 'binary_masks' / f'{view}.png'
                        if src_mask.exists():
                            shutil.copy2(src_mask, dst_mask)


                else:
                    if view in set(scene_data['train'] + scene_data['test']):
                        shutil.copy2(img_path, images_path / f"{view}.{image_format}")
                        if view in scene_data['train']:
                            shutil.copy2(img_path, train_views_path / f"{view}.{image_format}")

    print(f"Dataset structure has been created in the {root_dir} directory")

if __name__ == "__main__":
    create_dataset_structure()