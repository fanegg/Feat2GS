import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

METRICS_PATH = Path("/home/chenyue/output/Feat2gs/output/eval/")
DATA_JSON_PATH = Path("/home/chenyue/dataset/Feat2GS_Dataset/dataset_split.json")

@dataclass(frozen=True)
class Feature:
    tag: str
    full_name: str


@dataclass(frozen=True)
class Method(Feature):
    init: str = "dust3r"
    color: str | None = None
    iter: int = 8000

@dataclass(frozen=True)
class Metric(Feature):
    # The order determines what the metric means:
    # 1: higher is better
    # 0: there's no concept of ranking
    # -1: lower is better
    order: int
    group: str | None = None

@dataclass(frozen=True)
class Dataset(Feature):
    n_views: int | None = None

@dataclass(frozen=True)
class Scene(Feature):
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


DUST3R = Feature("dust3r", "DUSt3R")
MAST3R = Feature("mast3r", "MASt3R")
MIDAS = Feature("midas_l16", "MiDaS")
DINOV2 = Feature("dinov2_b14", "DINOv2")
DINO16 = Feature("dino_b16", "DINO")
SAM = Feature("sam_base", "SAM")
CLIP = Feature("clip_b16", "CLIP")
RADIO = Feature("radio", "RADIO")
MAE = Feature("mae_b16", "MAE")
DIFT = Feature("dift", "SD")
IUVRGB = Feature("iuvrgb", "IUVRGB")
ZERO123 = Feature("zero123", "Zero123")
# VIT = Feature("vit", "VIT")
# RESNET50 = Feature("resnet50", "RESNET")
# MASKCLIP = Feature("maskclip", "MASKCLIP")

DINO16_FEATUP = Feature("dino16_featup", r"DINO$^+$")
DINOV2_FEATUP = Feature("dinov2_featup", r"DINOv2$^+$")
CLIP_FEATUP = Feature("clip_featup", r"CLIP$^+$")

R_MAE_I = Feature("radio-mae_b16-iuvrgb", "RADIO+MAE+IUVRGB")
R_MAE = Feature("radio-mae_b16", "RADIO+MAE")
R_I = Feature("radio-iuvrgb", "RADIO+IUVRGB")

DN2_C_S = Feature("dinov2_b14-clip_b16-sam_base", "DINOv2+CLIP+SAM")
MT_DF_R_MAE_MD_I = Feature("mast3r-dift-radio-mae_b16-midas_l16-iuvrgb", "MASt3R+SD+RADIO+MAE+MiDaS+IUVRGB")
DT_I = Feature("dust3r-iuvrgb", "DUSt3R+IUVRGB")

ALL = Feature("dust3r-mast3r-dift-dino_b16-dinov2_b14-radio-clip_b16-mae_b16-midas_l16-sam_base-iuvrgb", "ALL")

I_DF = Feature("iuvrgb-dift", "IUVRGB+SD")
I_DF_S = Feature("iuvrgb-dift-sam_base", "IUVRGB+SD+SAM")
I_DF_S_DN2 = Feature("iuvrgb-dift-sam_base-dinov2_b14", "IUVRGB+SD+SAM+DINOv2")
I_DF_S_DN2_C = Feature("iuvrgb-dift-sam_base-dinov2_b14-clip_b16", "IUVRGB+SD+SAM+DINOv2+CLIP")
I_DF_S_DN2_C_MD = Feature("iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16", "IUVRGB+SD+SAM+DINOv2+CLIP+MiDaS")
I_DF_S_DN2_C_MD_MAE = Feature("iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16", "IUVRGB+SD+SAM+DINOv2+CLIP+MiDaS+MAE")
I_DF_S_DN2_C_MD_MAE_DN = Feature("iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16", "IUVRGB+SD+SAM+DINOv2+CLIP+MiDaS+MAE+DINO")
I_DF_S_DN2_C_MD_MAE_DN_DT = Feature("iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r", "IUVRGB+SD+SAM+DINOv2+CLIP+MiDaS+MAE+DINO+DUSt3R")
I_DF_S_DN2_C_MD_MAE_DN_DT_MT = Feature("iuvrgb-dift-sam_base-dinov2_b14-clip_b16-midas_l16-mae_b16-dino_b16-dust3r-mast3r", "IUVRGB+SD+SAM+DINOv2+CLIP+MiDaS+MAE+DINO+DUSt3R+MASt3R")

MT_R = Feature("mast3r-radio", "RADIO+MASt3R")
DT_MT_R = Feature("dust3r-mast3r-radio", "RADIO+MASt3R+DUSt3R")
DT_MT_DN_R = Feature("dust3r-mast3r-dino_b16-radio", "RADIO+MASt3R+DUSt3R+DINO")
R_MT_DT_DN_MAE = Feature("radio-mast3r-dust3r-dino_b16-mae_b16", "RADIO+MASt3R+DUSt3R+DINO+MAE")
R_MT_DT_DN_MAE_MD = Feature("radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16", "RADIO+MASt3R+DUSt3R+DINO+MAE+MiDaS")
R_MT_DT_DN_MAE_MD_C = Feature("radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16", "RADIO+MASt3R+DUSt3R+DINO+MAE+MiDaS+CLIP")
R_MT_DT_DN_MAE_MD_C_DN2 = Feature("radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14", "RADIO+MASt3R+DUSt3R+DINO+MAE+MiDaS+CLIP+DINOv2")
R_MT_DT_DN_MAE_MD_C_DN2_S = Feature("radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base", "RADIO+MASt3R+DUSt3R+DINO+MAE+MiDaS+CLIP+DINOv2+SAM")
R_MT_DT_DN_MAE_MD_C_DN2_S_DF = Feature("radio-mast3r-dust3r-dino_b16-mae_b16-midas_l16-clip_b16-dinov2_b14-sam_base-dift", "RADIO+MASt3R+DUSt3R+DINO+MAE+MiDaS+CLIP+DINOv2+SAM+SD")

METHOD_G = Method("feat2gs-G", r"\textbf{G}eometry")
METHOD_T = Method("feat2gs-T", r"\textbf{T}exture")
METHOD_A = Method("feat2gs-A", r"\textbf{A}ll")
METHOD_Gft = Method("feat2gs-Gft", r"\textbf{G}eometry")
METHOD_Tft = Method("feat2gs-Tft", r"\textbf{T}exture")
METHOD_Aft = Method("feat2gs-Aft", r"\textbf{A}ll")

METHOD_G_DTU_N10000 = Method("feat2gs-G_n10000", r"\textbf{G}eometry_{n10000}", init="gt")
METHOD_G_DTU_N1000 = Method("feat2gs-G_n1000", r"\textbf{G}eometry_{n1000}", init="gt")
METHOD_G_DTU_N100 = Method("feat2gs-G_n100", r"\textbf{G}eometry_{n100}", init="gt")

METHOD_G_DTU_N10 = Method("feat2gs-G_n10", r"\textbf{G}eometry", init="gt")
METHOD_T_DTU_N10 = Method("feat2gs-T_n10", r"\textbf{T}exture", init="gt")
METHOD_A_DTU_N10 = Method("feat2gs-A_n10", r"\textbf{A}ll", init="gt")

METHOD_G_DTU_N1 = Method("feat2gs-G_n1", r"\textbf{G}eometry_{n1}", init="gt")
METHOD_G_DTU_N0 = Method("feat2gs-G_n0", r"\textbf{G}eometry_{n0}", init="gt")
METHOD_G_DTU_N1N2 = Method("feat2gs-G_n1n2", r"\textbf{G}eometry_{n1e^{-2}}", init="gt")

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

DATASET_CASUAL = Dataset("Casual", "Casual")
DATASET_DL3DV = Dataset("DL3DV", "DL3DV")
DATASET_TANDT = Dataset("Tanks", "Tanks and Temples")   #T\&T
DATASET_MVIMGNET = Dataset("MVimgNet", "MVImgNet")
DATASET_MIPNERF360 = Dataset("MipNeRF360", "MipNeRF 360")
DATASET_LLFF = Dataset("LLFF", "LLFF")
DATASET_CO3D = Dataset("co3d", "CO3D", 3)
DATASET_DTU = Dataset("DTU", "DTU")


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
# SCENES_DTU = [
#     Scene(name, name.capitalize(), DATASET_DTU)
#     for name in ("scan1", "scan4", "scan9", "scan23", "scan75")
# ]
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
    features: Iterable[Feature],
    metrics: Iterable[Metric],
) -> pd.DataFrame:
    metrics = tuple(metrics)

    data = defaultdict(list)

    for method in methods:
        for feature in features:
            for scene in scenes:
                # Load the scene's metrics.
                METRICS_FILE_PATH = METRICS_PATH / f"{scene.dataset.tag}/{scene.tag}/{scene.n_views}_views/{method.tag}/{method.init}/{feature.tag}"
                if scene.dataset.tag == "DTU":
                    METRICS_FILE_PATH = METRICS_PATH / f"{scene.dataset.tag}/{scene.tag}/{scene.n_views}_views/{method.tag}_n10/gt/{feature.tag}"
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
                    scene_metrics["ATE"] = cam_metrics.get("ATE", None)
                    scene_metrics["RPE_trans"] = cam_metrics.get("RPE_trans", None)
                    scene_metrics["RPE_rot"] = cam_metrics.get("RPE_rot", None)
                except FileNotFoundError:
                    scene_metrics["ATE"], scene_metrics["RPE_trans"], scene_metrics["RPE_rot"] = None, None, None

                # Select the metrics we care about.
                for metric in metrics:
                    data["scene_tag"].append(scene.tag)
                    data["scene_full_name"].append(scene.full_name)

                    data["dataset_tag"].append(scene.dataset.tag)
                    data["dataset_full_name"].append(scene.dataset.full_name)

                    data["method_tag"].append(method.tag)
                    data["method_full_name"].append(method.full_name)

                    data["feature_tag"].append(feature.tag)
                    data["feature_full_name"].append(feature.full_name)

                    data["metric_tag"].append(metric.tag)
                    data["metric_full_name"].append(metric.full_name)
                    data["metric_value"].append(scene_metrics.get(metric.tag, None))

    return pd.DataFrame(data)

