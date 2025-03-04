# Disable Ruff's rules on star imports for common.
# ruff: noqa: F405, F403

from collections import defaultdict
from pathlib import Path

from common import *
from table import make_latex_table

OUT_PATH = Path("tables")

caption = f"{exp_name} persecne DTU"
# caption = "InstantSplat perscne"

METHODS = (
    METHOD_FEAT2GS_DUST3R,
    METHOD_FEAT2GS_MAST3R,
    METHOD_FEAT2GS_MIDAS,
    METHOD_FEAT2GS_DINOV2,
    METHOD_FEAT2GS_DINO16,
    METHOD_FEAT2GS_SAM,
    METHOD_FEAT2GS_CLIP,
    METHOD_FEAT2GS_RADIO,
    METHOD_FEAT2GS_MAE,
    METHOD_FEAT2GS_DIFT,
    METHOD_FEAT2GS_IUVRGB,
    # METHOD_INSTANTSPLAT,

    # METHOD_R_MAE_I,
    # METHOD_R_MAE,
    # METHOD_R_I,

    # METHOD_DN2_C_S,
    # METHOD_MT_DF_R_MAE_MD_I,
    # METHOD_DT_I,

    # METHOD_IUVRGB_DIFT,
    # METHOD_IUVRGB_DIFT_SAM,
    # METHOD_IUVRGB_DIFT_SAM_DINOV2,
    # METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP,
    # METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS,
    # METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE,
    # METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE_DINO,
    # METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE_DINO_DUST3R,
    # METHOD_IUVRGB_DIFT_SAM_DINOV2_CLIP_MIDAS_MAE_DINO_DUST3R_MAST3R,

    # METHOD_RADIO_MAST3R,
    # METHOD_RADIO_MAST3R_DUST3R,
    # METHOD_RADIO_MAST3R_DUST3R_DINO,
    # METHOD_RADIO_MAST3R_DUST3R_DINO_MAE,
    # METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS,
    # METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP,
    # METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP_DINOV2,
    # METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP_DINOV2_SAM,
    # METHOD_RADIO_MAST3R_DUST3R_DINO_MAE_MIDAS_CLIP_DINOV2_SAM_DIFT,

    # METHOD_ALL,
)


METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
    METRIC_ACC,
    METRIC_COMP,
    METRIC_DIST,
)

# SCENES = (*SCENES_LLFF, *SCENES_DL3DV, *SCENES_CASUAL, *SCENES_MIPNERF360, *SCENES_MVIMGNET, *SCENES_TANDT)
SCENES = (*SCENES_DTU, )
SCENES_PER_ROW = 4

if __name__ == "__main__":
    df = load_metrics(SCENES, METHODS, METRICS)
    grouped = df.groupby(["dataset_tag", "scene_tag", "metric_tag", "method_tag"])[
        "metric_value"
    ].mean()

    chunks = []
    chunks_id = 0
    table_begins = []
    for i in range(0, len(SCENES), SCENES_PER_ROW):
        chunks_id += 1
        row_scenes = SCENES[i : i + SCENES_PER_ROW]
        rows = defaultdict(list)
        multi_headers = []
        for scene in row_scenes:
            multi_headers.append(
                (f"{scene.full_name} ({scene.dataset.full_name})", len(METRICS))
            )
            for method in METHODS:
                for metric in METRICS:
                    rows[f"{method.full_name}"].append(
                        grouped[scene.dataset.tag][scene.tag][metric.tag][method.tag]
                    )

        table, table_begin, table_end = make_latex_table(
            rows,
            [metric.full_name for metric in METRICS] * len(row_scenes),
            # [2, 4, 4] * len(row_scenes),
            [2, 4, 4, 3, 3, 3] * len(row_scenes),
            [metric.order for metric in METRICS] * len(row_scenes),
            multi_headers=multi_headers,
        )
        chunks.append(table)
        table_begins.append(table_begin)
        if len(SCENES) % SCENES_PER_ROW == 0:
            if chunks_id < len(SCENES) // SCENES_PER_ROW:
                chunks += ["\\midrule",]
        else:
            if chunks_id <= (len(SCENES) - 1) // SCENES_PER_ROW:
                chunks += ["\\midrule",]

    table_begin = [
        "%%% BEGIN AUTOGENERATED %%%",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{table*}[t]",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
    ] + table_begins[0]

    table_end = table_end + [
        "}",
        "\\vspace{5pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:recon}",
        "\\vspace{-11pt}",
        "\\end{table*}",
        "%%% END AUTOGENERATED %%%",
    ]

    chunks = table_begin + chunks + table_end
    caption = caption.replace(" ", "_")
    out_path = OUT_PATH / f"{caption}.tex"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(out_path).open("w") as f:
        f.write("\n".join(chunks))
