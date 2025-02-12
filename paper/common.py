import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
from jaxtyping import Float
from torch import Tensor

# from flowmap.export.colmap import read_colmap_model
# from flowmap.misc.ate import compute_ate

METRICS_PATH = Path("/home/chenyue/output/Feat2gs/output/eval/")
DATA_JSON_PATH = Path("/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json")

@dataclass(frozen=True)
class Method:
    tag: str
    full_name: str
    parent: str
    init: str = "dust3r"
    feature: str | None = None
    color: str | None = None
    iter: int = 8000

exp_name="feat2gs-G"
METHOD_INSTANTSPLAT = Method("instantsplat", "InstantSplat", "instantsplat",iter=7000)
METHOD_FEAT2GS_DUST3R = Method("feat2gs_dust3r", "DUSt3R", exp_name, feature="dust3r")
METHOD_FEAT2GS_MAST3R = Method("feat2gs_mast3r", "MASt3R", exp_name, feature="mast3r")
METHOD_FEAT2GS_DUST3R_ft = Method("feat2gs_dust3r_ft", r"DUSt3R$^*$", 'feat2gs-Gft', feature="dust3r")
METHOD_FEAT2GS_MIDAS = Method("feat2gs_midas", "MiDaS", exp_name, feature="midas_l16")
METHOD_FEAT2GS_DINOV2 = Method("feat2gs_dinov2", "DINOv2", exp_name, feature="dinov2_b14")
METHOD_FEAT2GS_DINO16 = Method("feat2gs_dino16", "DINO", exp_name, feature="dino_b16")
METHOD_FEAT2GS_SAM = Method("feat2gs_sam", "SAM", exp_name, feature="sam_base")
METHOD_FEAT2GS_CLIP = Method("feat2gs_clip", "CLIP", exp_name, feature="clip_b16")
METHOD_FEAT2GS_RADIO = Method("feat2gs_radio", "RADIO", exp_name, feature="radio")
METHOD_FEAT2GS_MAE = Method("feat2gs_mae", "MAE", exp_name, feature="mae_b16")
METHOD_FEAT2GS_DIFT = Method("feat2gs_dift", "SD", exp_name, feature="dift")
METHOD_FEAT2GS_IUVRGB = Method("feat2gs_iuvrgb", "IUVRGB", exp_name, feature="iuvrgb")

METHOD_R_MAE_I = Method("feat2gs_radio-mae_b16-iuvrgb", "Feat2gs w/ radio+mae+iuvrgb", exp_name, feature="radio-mae_b16-iuvrgb")
METHOD_R_MAE = Method("feat2gs_radio-mae_b16", "Feat2gs w/ radio+mae", exp_name, feature="radio-mae_b16")
METHOD_R_I = Method("feat2gs_radio-iuvrgb", "Feat2gs w/ radio+iuvrgb", exp_name, feature="radio-iuvrgb")

METHOD_DN2_C_S = Method("feat2gs_dinov2_b14-clip_b16-sam_base", "Feat2gs w/ dinov2+clip+sam", exp_name, feature="dinov2_b14-clip_b16-sam_base")
METHOD_MT_DF_R_MAE_MD_I = Method("feat2gs_mast3r-dift-radio-mae_b16-midas_l16-iuvrgb", "Feat2gs w/ mast3r+sd+radio+mae+midas+iuvrgb", exp_name, feature="mast3r-dift-radio-mae_b16-midas_l16-iuvrgb")
METHOD_DT_I = Method("feat2gs_dust3r-iuvrgb", "Feat2gs w/ dust3r+iuvrgb", exp_name, feature="dust3r-iuvrgb")

METHOD_ALL = Method("feat2gs_dust3r-mast3r-dift-dino_b16-dinov2_b14-radio-clip_b16-mae_b16-midas_l16-sam_base-iuvrgb", "Feat2gs w/ concat all", exp_name, feature="dust3r-mast3r-dift-dino_b16-dinov2_b14-radio-clip_b16-mae_b16-midas_l16-sam_base-iuvrgb")

METHOD_IUVRGB_DIFT = Method("feat2gs_iuvrgb-dift", "Feat2gs w/ iuvrgb+dift", exp_name, feature="iuvrgb-dift")
METHOD_IUVRGB_DIFT_SAM = Method("feat2gs_iuvrgb-dift-sam_base", "Feat2gs w/ iuvrgb+dift+sam", exp_name, feature="iuvrgb-dift-sam_base")
METHOD_IUVRGB_DIFT_SAM_DINOV2 = Method("feat2gs_iuvrgb-dift-sam_base-dinov2_b14", "Feat2gs w/ iuvrgb+dift+sam+dinov2", exp_name, feature="iuvrgb-dift-sam_base-dinov2_b14")
METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP = Method("feat2gs_iuvrgb-dift-sam_base-dinov2_b14-clip_b16", "Feat2gs w/ iuvrgb+dift+sam+dinov2+clip", exp_name, feature="iuvrgb-dift-sam_base-dinov2_b14-clip_b16")
METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS = Method("feat2gs_iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16", "Feat2gs w/ iuvrgb+dift+sam+dinov2+clip+midas", exp_name, feature="iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16")
METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE = Method("feat2gs_iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16", "Feat2gs w/ iuvrgb+dift+sam+dinov2+clip+midas+mae", exp_name, feature="iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16")
METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE_DINO = Method("feat2gs_iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16", "Feat2gs w/ iuvrgb+dift+sam+dinov2+clip+midas+mae+dino", exp_name, feature="iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16")
METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE_DINO_DUST3R = Method("feat2gs_iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r", "Feat2gs w/ iuvrgb+dift+sam+dinov2+clip+midas+mae+dino+dust3r", exp_name, feature="iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r")
METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE_DINO_DUST3R_MAST3R = Method("feat2gs_iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r-mast3r", "Feat2gs w/ iuvrgb+dift+sam+dinov2+clip+midas+mae+dino+dust3r+mast3r", exp_name, feature="iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r-mast3r")


METHOD_RADIO_MAST3R = Method("feat2gs_radio-mast3r", "Feat2gs w/ radio+mast3r", exp_name, feature="mast3r-radio")
METHOD_RADIO_MAST3R_DUST3R = Method("feat2gs_radio-mast3r-dust3r", "Feat2gs w/ radio+mast3r+dust3r", exp_name, feature="dust3r-mast3r-radio")
METHOD_RADIO_MAST3R_DUST3R_DINO = Method("feat2gs_radio-mast3r-dust3r-dino_b16", "Feat2gs w/ radio+mast3r+dust3r+dino", exp_name, feature="dust3r-mast3r-dino_b16-radio")
METHOD_RADIO_MAST3R_DUST3R_DINO_MAE = Method("feat2gs_radio-mast3r-dust3r-dino_b16-mae_b16", "Feat2gs w/ radio+mast3r+dust3r+dino+mae", exp_name, feature="radio-mast3r-dust3r-dino_b16-mae_b16")
METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS = Method("feat2gs_radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16", "Feat2gs w/ radio+mast3r+dust3r+dino+mae+midas", exp_name, feature="radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16")
METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP = Method("feat2gs_radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16", "Feat2gs w/ radio+mast3r+dust3r+dino+mae+midas+clip", exp_name, feature="radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16")
METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP_DINOV2 = Method("feat2gs_radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14", "Feat2gs w/ radio+mast3r+dust3r+dino+mae+midas+clip+dinov2", exp_name, feature="radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14")
METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP_DINOV2_SAM = Method("feat2gs_radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base", "Feat2gs w/ radio+mast3r+dust3r+dino+mae+midas+clip+dinov2+sam", exp_name, feature="radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base")
METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP_DINOV2_SAM_DIFT = Method("feat2gs_radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base-dift", "Feat2gs w/ radio+mast3r+dust3r+dino+mae+midas+clip+dinov2+sam+dift", exp_name, feature="radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base-dift")


@dataclass(frozen=True)
class Metric:
    tag: str
    full_name: str

    # The order determines what the metric means:
    # 1: higher is better
    # 0: there's no concept of ranking
    # -1: lower is better
    order: int
    group: str | None = None


METRIC_PSNR = Metric("PSNR", "PSNR", 1, "2d")
METRIC_SSIM = Metric("SSIM", "SSIM", 1, "2d")
METRIC_LPIPS = Metric("LPIPS", "LPIPS", -1, "2d")
METRIC_RUNTIME = Metric("runtime", "Time (min.)", -1)
METRIC_COLMAP_ATE = Metric("ATE", "ATE", -1)
METRIC_RPE_TRANS = Metric("RPE_trans", "RPE_{trans}", -1)
METRIC_RPE_ROT = Metric("RPE_rot", "RPE_{rot}", -1)

METRIC_ACC = Metric("Accuracy", "Acc.", -1, "3d")
METRIC_ACC_MED = Metric("Acc_med", r"$Acc._{{m}}$", -1, "3d")
METRIC_COMP = Metric("Completion", "Comp.", -1, "3d")
METRIC_COMP_MED = Metric("Comp_med", r"$Comp._{{m}}$", -1, "3d")
METRIC_DIST = Metric("Distance", "Dist.", -1, "3d")
METRIC_DIST_MED = Metric("Dist_med", r"$Dist._{{m}}$", -1, "3d")


@dataclass(frozen=True)
class Dataset:
    tag: str
    full_name: str
    n_views: int | None = None

DATASET_CASUAL = Dataset("Casual", "Casual")
DATASET_DL3DV = Dataset("DL3DV", "DL3DV")
DATASET_TANDT = Dataset("Tanks", "T\&T")
DATASET_MVIMGNET = Dataset("MVimgNet", "MVImgNet")
DATASET_MIPNERF360 = Dataset("MipNeRF360", "MipNeRF 360")
DATASET_LLFF = Dataset("LLFF", "LLFF")
DATASET_CO3D = Dataset("co3d", "CO3D", 3)
DATASET_DTU = Dataset("DTU", "DTU")

@dataclass(frozen=True)
class Scene:
    tag: str
    full_name: str
    dataset: Dataset
    n_views: int | None = None

    def __post_init__(self):
        if self.n_views is None:
            if self.dataset.n_views is None:
                with open(DATA_JSON_PATH, 'r') as f:
                    data = json.load(f)
                    dataset_data = data.get(self.dataset.tag, {})
                    scene_data = dataset_data.get(self.tag, {})
                    train_views = scene_data.get('train', [])
                    object.__setattr__(self, 'n_views', len(train_views))
            else:
                object.__setattr__(self, 'n_views', self.dataset.n_views)


def load_scenes_from_json(dataset: Dataset) -> list[Scene]:
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)
    dataset_data = data.get(dataset.tag, {})
    return [
        Scene(
            tag,
            tag.capitalize(),
            dataset,
            n_views=(len(scene_data.get('train', []))
                     if dataset.n_views is None
                     else dataset.n_views)
        )
        for tag, scene_data in dataset_data.items()
    ]

SCENES_CASUAL = load_scenes_from_json(DATASET_CASUAL)
SCENES_DL3DV = load_scenes_from_json(DATASET_DL3DV)
SCENES_MVIMGNET = load_scenes_from_json(DATASET_MVIMGNET)
SCENES_MIPNERF360 = load_scenes_from_json(DATASET_MIPNERF360)
SCENES_TANDT = load_scenes_from_json(DATASET_TANDT)
SCENES_LLFF = load_scenes_from_json(DATASET_LLFF)
SCENES_DTU = load_scenes_from_json(DATASET_DTU)
SCENES_CO3D = [
    Scene(f"co3d_{name}", name.capitalize(), DATASET_CO3D)
    for name in ("bench", "hydrant")  # demo scenes
]

def parse_pose_eval_file(file_path: Path) -> dict:
    """Parse the pose evaluation file and extract metrics."""
    metrics = {}
    with file_path.open("r") as f:
        content = f.read().strip()
        # Split content by commas
        items = content.split(", ")
        for item in items:
            key, value = item.split(": ")
            metrics[key] = float(value)
    return metrics

def load_metrics(
    scenes: Iterable[Scene],
    methods: Iterable[Method],
    metrics: Iterable[Metric],
) -> pd.DataFrame:
    metrics = tuple(metrics)
    data = defaultdict(list)

    for method in methods:
        for scene in scenes:
            # Load the scene's metrics.
            METRICS_FILE_PATH = METRICS_PATH / f"{scene.dataset.tag}/{scene.tag}/{scene.n_views}_views/{method.parent}/{method.init}"
            if scene.dataset.tag == "DTU":
                METRICS_FILE_PATH = METRICS_PATH / f"{scene.dataset.tag}/{scene.tag}/{scene.n_views}_views/{method.parent}_n10/gt"
            if "feat2gs" in method.tag:
                METRICS_FILE_PATH /= method.feature

            result_name = METRICS_FILE_PATH / "results.json"
            metric_3d_name = METRICS_FILE_PATH / "3d_metrics.json"

            try:
                with (result_name).open("r") as fi:
                    scene_metrics = json.load(fi)[f"ours_{method.iter}"]
            except FileNotFoundError:
                scene_metrics = {}
                print(f"No results.json for {METRICS_FILE_PATH}")

            if metric_3d_name.exists():
                try:
                    with (metric_3d_name).open("r") as fi:
                        metrics_3d = json.load(fi)[f"ours_{method.iter}"]["threshold_0.0"]
                        # scene_metrics["Accuracy"] = metrics_3d["Accuracy"]
                        # scene_metrics["Acc_med"] = metrics_3d["Acc_med"]
                        # scene_metrics["Completion"] = metrics_3d["Completion"]
                        # scene_metrics["Comp_med"] = metrics_3d["Comp_med"]
                        # scene_metrics["Distance"] = metrics_3d["Distance"]
                        # scene_metrics["Dist_med"] = metrics_3d["Dist_med"]

                        #TODO: remove this
                        scene_metrics["Accuracy"] = metrics_3d["Accuracy"] * 1000.
                        scene_metrics["Acc_med"] = metrics_3d["Acc_med"] * 1000.
                        scene_metrics["Completion"] = metrics_3d["Completion"] * 1000.
                        scene_metrics["Comp_med"] = metrics_3d["Comp_med"] * 1000.
                        scene_metrics["Distance"] = metrics_3d["Distance"] * 1000.
                        scene_metrics["Dist_med"] = metrics_3d["Dist_med"] * 1000.

                except KeyError:
                    scene_metrics["Accuracy"] = None
                    scene_metrics["Acc_med"] = None
                    scene_metrics["Completion"] = None
                    scene_metrics["Comp_med"] = None
                    scene_metrics["Distance"] = None
                    scene_metrics["Dist_med"] = None
                    print(f"No 3d_metrics.json for {METRICS_FILE_PATH}")

            try:
                cam_metrics = parse_pose_eval_file(METRICS_FILE_PATH / "pose/pose_eval.txt")
                scene_metrics.update({k: cam_metrics.get(k, None) for k in ["ATE", "RPE_trans", "RPE_rot"]})
            except FileNotFoundError:
                scene_metrics.update({k: None for k in ["ATE", "RPE_trans", "RPE_rot"]})

            # Select the metrics we care about.
            for metric in metrics:
                data["scene_tag"].append(scene.tag)
                data["scene_full_name"].append(scene.full_name)

                data["dataset_tag"].append(scene.dataset.tag)
                data["dataset_full_name"].append(scene.dataset.full_name)

                data["method_tag"].append(method.tag)
                data["method_full_name"].append(method.full_name)

                data["metric_tag"].append(metric.tag)
                data["metric_full_name"].append(metric.full_name)
                data["metric_value"].append(scene_metrics.get(metric.tag, None))

    return pd.DataFrame(data)
