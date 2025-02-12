# Disable Ruff's rules on star imports for common.
# ruff: noqa: F405, F403

from collections import defaultdict
from pathlib import Path
from scipy import stats

from gta.common import *
from gta.table import make_latex_table

OUT_PATH = Path("tables")

caption = "GTA cross dataset"
USE_COLORS = True   #False

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
    # ZERO123,
    
    # DINOV2,
    # DINOV2_FEATUP,
    # DINO16,
    # DINO16_FEATUP,
    # CLIP,
    # CLIP_FEATUP,

    # MT_R,
    # DT_MT_R,
    # DT_MT_DN_R,
    # R_MT_DT_DN_MAE,
    # R_MT_DT_DN_MAE_MD,
    # R_MT_DT_DN_MAE_MD_C,
    # R_MT_DT_DN_MAE_MD_C_DN2,
    # R_MT_DT_DN_MAE_MD_C_DN2_S,
    # R_MT_DT_DN_MAE_MD_C_DN2_S_DF,

    # I_DF,
    # I_DF_S,
    # I_DF_S_DN2,
    # I_DF_S_DN2_C,
    # I_DF_S_DN2_C_MD,
    # I_DF_S_DN2_C_MD_MAE,
    # I_DF_S_DN2_C_MD_MAE_DN,
    # I_DF_S_DN2_C_MD_MAE_DN_DT,
    # I_DF_S_DN2_C_MD_MAE_DN_DT_MT,
    # ALL,

    # R_MAE_I,
    # R_MAE,
    # R_I,

    # DN2_C_S,
    # MT_DF_R_MAE_MD_I,
    # DT_I,
)

METHODS = (
    METHOD_G,
    METHOD_T,
    METHOD_A,
    # METHOD_Gft,
    # METHOD_Tft,
    # METHOD_Aft,
)

METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
)

SCENES = (
    (*SCENES_CASUAL,),
    (*SCENES_DL3DV,),
    (*SCENES_LLFF,),
    (*SCENES_TANDT,),
    (*SCENES_MVIMGNET,), 
    (*SCENES_MIPNERF360,), 
)



if __name__ == "__main__":
    chunks = []
    df = load_metrics(sum(SCENES, ()), METHODS, FEATURES, METRICS)

    grouped = df.groupby(["method_tag", "metric_tag", "feature_tag"])[
        "metric_value"
    ].mean()

    # grouped = df.groupby(["method_tag", "metric_tag", "feature_tag"])[
    #     "metric_value"
    # ].median()

    # grouped = df.groupby(["method_tag", "metric_tag", "feature_tag"])["metric_value"].apply(
    #     lambda x: stats.trim_mean(x, proportiontocut=0.6)
    # )


    rows = defaultdict(list)
    multi_headers = []
    model_headers = []

    multi_headers.append(
        (f"All Datasets ({len(sum(SCENES, ()))} scenes)", len(METRICS) * len(METHODS))
    )

    for method in METHODS:
        model_headers.append((method.full_name, len(METRICS)))
        for feature in FEATURES:
            for metric in METRICS:
                rows[rf"{feature.full_name}{'$^*$' if 'ft' in method.tag else ''}"].append(
                    grouped[method.tag][metric.tag][feature.tag]
                )

    table, table_begin, table_end = make_latex_table(
        rows,
        [metric.full_name for metric in METRICS] * len(METHODS),
        [2, 4, 4] * len(METHODS),
        [metric.order for metric in METRICS] * len(METHODS),
        multi_headers=multi_headers,
        model_headers=model_headers,
        use_colors=USE_COLORS
    )

    chunks.append(table)

    table_begin = [
        "%%% BEGIN AUTOGENERATED %%%",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{table*}[t]",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
    ] + table_begin

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