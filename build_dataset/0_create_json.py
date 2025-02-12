import os
import json

"""
datasets_root follows the structure as follows:

eval_tmp_data/
    dataset_name/
        scene_name/
            train/
                train_view1.jpg
                train_view2.jpg
                ...
            test_view1.jpg
            test_view2.jpg
            ...
"""

datasets_root = "/home/chenyue/dataset/eval_tmp_data/"
json_file = "/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json"

DL3DV_HASH_MAPPING = {
    "Center": "cd9c981eeb4a9091547af19181b382698e9d9eee0a838c7c9783a8a268af6aee",
    "Electrical": "e8ce51b6abfe05bf8dca47e29c8be6c1e6de27a8c9fece7a121400b931b2ca0f",
    "Furniture": "5c3af581028068a3c402c7cbe16ecf9471ddf2897c34ab634b7b1b6cf81aba00",
    "Gallery": "cc08c0bdc34ddd2867705d0b17a86ec2a9d7c7926ce99070ed1fdc66a812de07",
    "Garden": "9c8c0e0fadd97abf60088f667eba058ef077a71a8c0f5a4eff2782aa97f1ceb8",
    "Museum": "dac9796dd69e1c25277e29d7e30af4f21e3b575d62a0a208c2b3d1211e2d5d77",
    "Supermarket1": "5f0041e53d59d67c3ca25db97b269db183a532c4566a6bc46ca0e69cfa4234ad",
    "Supermarket2": "cbd44beb04f9c98f2d2c5affff89a6e0a72c25f1aa0c6f660fbd9e4d26702f8b",
    "Temple": "ba55c875d20c34ee85ffc72264c4d77710852e5fb7d9ce4b9c26a8442850e98f"
}

def update_data_split(datasets_root, json_file):
    # Check if the directory for the JSON file exists, create it if not
    json_dir = os.path.dirname(json_file)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # Read existing JSON file, or create an empty dictionary if it doesn't exist
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    # Iterate through all datasets in the root directory
    for dataset_name in sorted(os.listdir(datasets_root)):
        dataset_path = os.path.join(datasets_root, dataset_name)
        if os.path.isdir(dataset_path):
            # Create a new dictionary for the dataset if it doesn't exist in JSON
            if dataset_name not in data:
                data[dataset_name] = {}

            # Iterate through scenes
            for scene in sorted(os.listdir(dataset_path)):
                scene_path = os.path.join(dataset_path, scene)
                if os.path.isdir(scene_path):
                    train_path = os.path.join(scene_path, 'train')
                    
                    train_views = []
                    test_views = []

                    # Get training views
                    if os.path.exists(train_path):
                        train_views = sorted([os.path.splitext(f)[0] for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))])

                    # Get test views
                    for f in sorted(os.listdir(scene_path)):
                        file_path = os.path.join(scene_path, f)
                        if os.path.isfile(file_path) and f != 'train':
                            test_views.append(os.path.splitext(f)[0])

                    # Update JSON data
                    data[dataset_name][scene] = {
                        "train": train_views,
                        "test": test_views
                    }

                    # Add hash value if it's a DL3DV dataset
                    if dataset_name == "DL3DV" and scene in DL3DV_HASH_MAPPING:
                        data[dataset_name][scene]["hash"] = DL3DV_HASH_MAPPING[scene]

    # Write updated data back to JSON file
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated {json_file} file with information for all datasets.")
    except PermissionError:
        print(f"Unable to write to file {json_file}. Please check file permissions.")


update_data_split(datasets_root, json_file)