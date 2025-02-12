# Disable Ruff's rules on star imports for common.
# ruff: noqa: F405, F403

from collections import defaultdict
from pathlib import Path

from gta.common import *
from gta.table import make_latex_table

OUT_PATH = Path("tables")

caption = "GTA"
USE_COLORS = True #True

FEATURES = (
    DUST3R,
    MAST3R,
    MIDAS,
    DINOV2,
    DINO16,
    SAM,
    CLIP,
    RADIO,
    MAE,
    DIFT,
    IUVRGB,
    # ZERO123
    # R_MAE_I,
    # DN2_C_S,
    # DT_MT_DN_R,
    # ALL,
)

METHODS = (
    METHOD_G,
    METHOD_T,
    METHOD_A,
    # METHOD_Gft,
    # METHOD_Tft,
    # METHOD_Aft,

    # METHOD_G_DTU_N1N2,
    # METHOD_G_DTU_N0,
    # METHOD_G_DTU_N1,
    # METHOD_G_DTU_N10,
    # METHOD_T_DTU_N10,
    # METHOD_A_DTU_N10,
    # METHOD_G_DTU_N100,
    # METHOD_G_DTU_N1000,
    # METHOD_G_DTU_N10000,

    # METHOD_T_DTU,
    # METHOD_A_DTU,
)

METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
    # METRIC_ACC,
    # METRIC_ACC_MED,
    # METRIC_COMP,
    # METRIC_COMP_MED,
    # METRIC_DIST,
    # METRIC_DIST_MED,
)

SCENES = (
    (*SCENES_LLFF, *SCENES_DL3DV, *SCENES_CASUAL), 
    (*SCENES_MIPNERF360, *SCENES_MVIMGNET, *SCENES_TANDT),
    # (*SCENES_DTU,),
)


if __name__ == "__main__":
    chunks = []
    chunks_id = 0
    for row_scenes in SCENES:
        chunks_id += 1
        df = load_metrics(row_scenes, METHODS, FEATURES, METRICS)
        # row_datasets = tuple(set(scene.dataset for scene in row_scenes))
        row_datasets = tuple(dict.fromkeys(scene.dataset for scene in row_scenes))

        grouped = df.groupby(["dataset_tag", "method_tag", "metric_tag", "feature_tag"])[
            "metric_value"
        ].mean()

        rows = defaultdict(list)
        multi_headers = []
        model_headers = []
        for dataset in row_datasets:
            num_scenes = sum([scene.dataset.tag == dataset.tag for scene in row_scenes])
            print(num_scenes)
            multi_headers.append(
                (f"{dataset.full_name}", len(METRICS) * len(METHODS))
            )

            for method in METHODS:
                model_headers.append((method.full_name, len(METRICS)))
                for feature in FEATURES:
                    for metric in METRICS:
                        rows[f"{feature.full_name}"].append(
                            grouped[dataset.tag][method.tag][metric.tag][feature.tag]
                        )

        table, table_begin, table_end = make_latex_table(
            rows,
            [metric.full_name for metric in METRICS] * len(row_datasets) * len(METHODS),
            [2, 4, 4] * len(row_datasets) * len(METHODS),
            # [2, 4, 4, 3, 3, 3] * len(row_datasets) * len(METHODS),
            # [2, 4, 4, 3, 3, 3, 3, 3, 3] * len(row_datasets) * len(METHODS),
            [metric.order for metric in METRICS] * len(row_datasets) * len(METHODS),
            multi_headers=multi_headers,
            model_headers=model_headers,
            use_colors=USE_COLORS

        )

        chunks.append(table)

        if chunks_id < len(SCENES):
            chunks += ["\\midrule",]


    table_begin = [
        "%%% BEGIN AUTOGENERATED %%%",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{table*}[t!]",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
    ] + table_begin

    table_end = table_end + [
        "}",
        # "\\vspace{5pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:GTA}",
        # "\\vspace{-11pt}",
        "\\end{table*}",
        "%%% END AUTOGENERATED %%%",
    ]

    chunks = table_begin + chunks + table_end
    caption = caption.replace(" ", "_")
    out_path = OUT_PATH / f"{caption}.tex"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(out_path).open("w") as f:
        f.write("\n".join(chunks))