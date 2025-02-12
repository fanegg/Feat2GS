from collections.abc import Iterable
from math import isnan
from typing import Literal, Tuple

import numpy as np
from einops import repeat
from jaxtyping import Float, Int


def round_values(
    values: Float[np.ndarray, "row col"],
    precisions: Int[np.ndarray, " col"],
) -> Float[np.ndarray, "row col"]:
    """Round values to the specified precision."""
    quantized = np.zeros_like(values)
    r, _ = values.shape
    precisions = repeat(precisions, "c -> r c", r=r)
    for precision in np.unique(precisions):
        mask = precisions == precision
        quantized[mask] = np.round(values[mask], precision)
    return quantized


def compute_ranks_for_column(
    values: Float[np.ndarray, " row"],
    order: Literal[-1, 0, 1],
) -> Int[np.ndarray, " row"]:
    # If the entries are unordered, return an arbitrary ranking.
    if order == 0:
        return np.full_like(values, 1e5, dtype=np.int32)

    # Handle NaNs.
    values = np.copy(values)
    values[np.isnan(values)] = -order * np.inf

    # Find and rank unique values.
    ranked_unique_values = np.sort(np.unique(values))
    if order == 1:
        ranked_unique_values = ranked_unique_values[::-1]

    # Assign ranks to the original entries.
    ranks = np.zeros_like(values, dtype=np.int32)
    for rank, value in enumerate(ranked_unique_values):
        ranks[values == value] = rank

    return np.int32(ranks)


def compute_ranks(
    values: Float[np.ndarray, "row col"],
    orders: Int[np.ndarray, " col"],
) -> Int[np.ndarray, "row col"]:
    ranks = np.zeros_like(values, dtype=np.int64)
    _, c = values.shape
    for col in range(c):
        ranks[:, col] = compute_ranks_for_column(values[:, col], orders[col].item())
    return ranks



TableRows = dict[str, list[float | None]]  # Maps method names to rows of results.
MultiHeaders = Iterable[tuple[str, int]]  # Each element is (text, num columns).
ModelHeaders = Iterable[tuple[str, int]]  # Each element is (text, num columns).

maskout = []#['IUVRGB']
def make_latex_table(
    results: TableRows,
    metrics: list[str],
    precisions: list[int],
    rank_orders: list[Literal[-1, 0, 1]],
    none_str: str = "N/A",
    multi_headers: MultiHeaders | None = None,
    model_headers: ModelHeaders | None = None,
    color_low: Tuple[float, float, float] = (1.0, 0.816, 0.816),  # #ffd0d0 (light red)
    color_mid: Tuple[float, float, float] = (1.0, 0.941, 0.816),  # #fff0d0 (light yellow)
    color_high: Tuple[float, float, float] = (0.565, 0.753, 0.376), # #90c060 (medium green)
    use_colors: bool = True,
    metric_types: bool = False,
) -> str:
    data = np.array(list(results.values()), dtype=np.float64)
    data_rounded = round_values(data, np.array(precisions))
    
    exclude_mask = np.array([method not in maskout for method in results.keys()])

    # color table by RANKing metrics per datasets
    def interpolate_color(value, values, order, color_low, color_mid, color_high):
        if isnan(value) or value is None:
            return None
        
        sorted_values = sorted(values, reverse=(order == 1))
        rank = sorted_values.index(value)
        
        normalized_rank = rank / (len(values) - 1)
        mean_normalized = 0.5
        
        if normalized_rank <= mean_normalized:
            t = normalized_rank / mean_normalized
            return tuple(c1 + (c2 - c1) * t for c1, c2 in zip(color_high, color_mid))
        else:
            t = (normalized_rank - mean_normalized) / (1 - mean_normalized)
            return tuple(c1 + (c2 - c1) * t for c1, c2 in zip(color_mid, color_low))

    colors = np.zeros((data_rounded.shape[0], data_rounded.shape[1], 3))
    best_values = np.zeros(data_rounded.shape, dtype=bool)
    for col in range(data_rounded.shape[1]):
        column_values = data_rounded[exclude_mask, col]
        if use_colors:
            colors[exclude_mask, col] = [
                interpolate_color(value, column_values, rank_orders[col], color_low, color_mid, color_high)
            for value in column_values
            ]

        if rank_orders[col] == 1:
            best_values[:, col] = column_values == np.max(column_values)
        elif rank_orders[col] == -1:
            best_values[:, col] = column_values == np.min(column_values)

    # # color table CROSS dataset by computing RANKing of ALL datasets
    # num_metrics = len(set(metrics))
    # column_values = []
    # start = 0
    # for _, span in multi_headers:
    #     end = start + span
    #     dataset_data = data_rounded[exclude_mask][:, start:end].reshape(-1, num_metrics)
    #     column_values.append(np.tile(dataset_data, span // num_metrics))
    #     start = end
    # column_values = np.concatenate(column_values, axis=1)

    # colors = np.zeros((data_rounded.shape[0], data_rounded.shape[1], 3))
    # for col in range(data_rounded.shape[1]):
    #     colors[exclude_mask, col] = [
    #         interpolate_color(value, column_values[:,col], rank_orders[col], color_low, color_mid, color_high)
    #         for value in data_rounded[exclude_mask, col]
    #     ]


    # # color table by computing an MEDIAN/MEAN per datasets
    # max_values = np.max(data_rounded[exclude_mask], axis=0)
    # min_values = np.min(data_rounded[exclude_mask], axis=0)
    # # mean_values = np.mean(data_rounded[exclude_mask], axis=0)
    # mean_values = np.median(data_rounded[exclude_mask], axis=0)

    # # color table CROSS dataset by computing an MEDIAN/MEAN of ALL datasets
    # num_metrics = len(set(metrics))
    # max_values = np.zeros(data_rounded.shape[1])
    # min_values = np.zeros(data_rounded.shape[1])
    # mean_values = np.zeros(data_rounded.shape[1])
    # start = 0
    # for _, span in multi_headers:
    #     end = start + span
    #     dataset_data = data_rounded[:, start:end]
    #     dataset_max = np.max(dataset_data[exclude_mask].reshape(-1, num_metrics), axis=0)
    #     dataset_min = np.min(dataset_data[exclude_mask].reshape(-1, num_metrics), axis=0)
    #     dataset_mean = np.median(dataset_data[exclude_mask].reshape(-1, num_metrics), axis=0)
    #     # dataset_mean = np.mean(dataset_data[exclude_mask].reshape(-1, num_metrics), axis=0)
    #     # dataset_mean = np.quantile(dataset_data[exclude_mask].reshape(-1, num_metrics), 0.4, axis=0)
    #     max_values[start:end] = np.tile(dataset_max, span // num_metrics)
    #     min_values[start:end] = np.tile(dataset_min, span // num_metrics)
    #     mean_values[start:end] = np.tile(dataset_mean, span // num_metrics)
    #     start = end

    # def interpolate_color(value, min_val, max_val, mean_val, order, color_low, color_mid, color_high):
    #     if isnan(value) or value is None:
    #         return None
    #     if min_val == max_val:
    #         return color_mid
    #     normalized = (value - min_val) / (max_val - min_val)
    #     mean_normalized = (mean_val - min_val) / (max_val - min_val)
    #     # mean_normalized = 0.5

    #     if order != 1:
    #         normalized = 1 - normalized
    #         mean_normalized = 1 - mean_normalized
    #     if normalized <= mean_normalized:
    #         t = normalized / mean_normalized
    #         return tuple(c1 + (c2 - c1) * t for c1, c2 in zip(color_low, color_mid))
    #     else:
    #         t = (normalized - mean_normalized) / (1 - mean_normalized)
    #         return tuple(c1 + (c2 - c1) * t for c1, c2 in zip(color_mid, color_high))

    # colors = np.zeros((data_rounded.shape[0], data_rounded.shape[1], 3))
    # for col in range(data_rounded.shape[1]):
    #     colors[exclude_mask, col] = [
    #         interpolate_color(value, min_values[col], max_values[col], mean_values[col], rank_orders[col], color_low, color_mid, color_high)
    #         for value in data_rounded[exclude_mask, col]
    #     ]

    def format_value(value, precision):
        if value is None or isnan(value):
            return none_str
        formatted = f"{value:.{precision}f}"
        return formatted[1:] if formatted.startswith("0") else formatted

    cells = [
        [
            method_name,
            *[
                (
                    none_str
                    if (value is None or isnan(value))
                    else (
                        format_value(value, precisions[col])
                        if method_name in maskout
                        else f"\\cellcolor[rgb]{{{colors[row,col,0]:.2f},{colors[row,col,1]:.2f},{colors[row,col,2]:.2f}}}{format_value(value, precisions[col])}"
                        if use_colors
                        else (
                            f"\\textbf{{{format_value(value, precisions[col])}}}"
                            if best_values[row, col]
                            else format_value(value, precisions[col])
                        )
                    )
                )
                for col, value in enumerate(row_values)
            ],
        ]
        for row, (method_name, row_values) in enumerate(results.items())
    ]

    # cells = [
    #     [
    #         method_name,
    #         *[
    #             (
    #                 none_str
    #                 if (value is None or isnan(value))
    #                 else (
    #                     f"{value:.{precisions[col]}f}"
    #                     if method_name in maskout
    #                     else f"\\cellcolor[rgb]{{{colors[row,col,0]:.2f},{colors[row,col,1]:.2f},{colors[row,col,2]:.2f}}}{value:.{precisions[col]}f}"
    #                     if use_colors
    #                     else (
    #                         f"\\textbf{{{value:.{precisions[col]}f}}}"
    #                         if best_values[row, col]
    #                         else f"{value:.{precisions[col]}f}"
    #                     )
    #                 )
    #             )
    #             for col, value in enumerate(row_values)
    #         ],
    #     ]
    #     for row, (method_name, row_values) in enumerate(results.items())
    # ]

    rank_symbols = {
        0: "",
        1: "$\\uparrow$",
        -1: "$\\downarrow$",
    }

    # Add arrows to the metric names.
    # metrics = [
    #     f"{metric}{rank_symbols[rank_order]}"
    #     for metric, rank_order in zip(metrics, rank_orders)
    # ]

    metrics = [
        f"\\fontsize{{8.5pt}}{{9pt}}\\selectfont{{{metric}{rank_symbols[rank_order]}}}"
        for metric, rank_order in zip(metrics, rank_orders)
    ]

    # Add a row for the headers.
    cells = [["Feature", *metrics], *cells]

    # Figure out the maximum width for each column.
    widths = np.array([[len(cell) for cell in row] for row in cells])
    max_widths = np.max(widths, axis=0)

    cells = [
        [
            (cell.rjust if row > 0 and col > 0 else cell.ljust)(max_widths[col])
            for col, cell in enumerate(row_cells)
        ]
        for row, row_cells in enumerate(cells)
    ]

    # Join the cells into LaTeX rows.
    rows = [" & ".join(row) + " \\\\" for row in cells]
    # rows = [" & ".join(row) for row in cells]

    # Create the multi headers.
    if multi_headers is None:
        multi_header_rows = []
    else:
        columns = [
            f"\multicolumn{{{span}}}{{{'c|' if i < len(multi_headers) - 1 else 'c'}}}"
            f"{{{text}}}"
            for i, (text, span) in enumerate(multi_headers)
        ]
        multi_header_rows = [
            " & ".join(("\multicolumn{1}{c|}{}", *columns)) + " \\\\",
            "\\midrule",
        ]

    if model_headers is None:
        model_headers = []
        metric_type_rows = []
    else:
        model_columns = [
            f"\multicolumn{{{span}}}{{{'c|' if i < len(model_headers) - 1 else 'c'}}}"
            f"{{{text}}}"
            for i, (text, span) in enumerate(model_headers)
        ]

        if metric_types:
            metric_type_cols = []
            for i, (_, span) in enumerate(model_headers):
                metric_type_cols.extend([
                    f"\multicolumn{{3}}{{{'c|'}}}{{2D Metrics}}",
                    f"\multicolumn{{3}}{{{'c|' if i < len(model_headers) - 1 else 'c'}}}{{3D Metrics}}"
                ])
            metric_type_rows = [
                " & ".join(("\multicolumn{1}{c|}{}", *metric_type_cols)) + " \\\\",
                "\\midrule",
            ]
        else:
            metric_type_rows = []

        model_header_rows = [
            " & ".join(("\multicolumn{1}{c|}{}", *model_columns)) + " \\\\",
            "\\midrule",
        ]

        if metric_types:
            model_header_rows.extend(metric_type_rows)

    # Add the rules.
    header_row, *other_rows = rows
    # column_specifications = "c" * len(metrics)
    # if model_headers is not None:
    #     chunks = []
    #     i = 0
    #     for _, span in model_headers:
    #         chunks.append("|")
    #         chunks.append(column_specifications[i : i + span])
    #         i += span
    #     column_specifications = "".join(chunks)

    column_width=0.9
    column_specifications = "".join(f">{{\\raggedleft\\arraybackslash}}p{{{column_width}cm}}" for _ in metrics)

    # 对于model_headers,也使用固定宽度和右对齐
    if model_headers is not None:
        chunks = []
        i = 0
        for _, span in model_headers:
            chunks.append("|")

            if metric_types:
                chunks.append("".join(f">{{\\raggedleft\\arraybackslash}}p{{{column_width}cm}}" for _ in range(3)))
                chunks.append("|")  # 添加2D和3D指标之间的分隔线
                chunks.append("".join(f">{{\\raggedleft\\arraybackslash}}p{{{column_width}cm}}" for _ in range(3)))
            else:
                chunks.append("".join(f">{{\\raggedleft\\arraybackslash}}p{{{column_width}cm}}" for _ in range(span)))

            i += span
        column_specifications = "".join(chunks)

    # if multi_headers is not None:
    #     chunks = []
    #     i = 0
    #     for _, span in multi_headers:
    #         chunks.append("|")
    #         chunks.append(column_specifications[i : i + span])
    #         i += span
    #     column_specifications = "".join(chunks)


    row_begin = [
        f"\\begin{{tabular}}{{l{column_specifications}}}",
        "\\toprule",
    ]

    rows = [
        *multi_header_rows,
        *model_header_rows,
        header_row,
        "\\midrule",
        *other_rows,
    ]

    row_end = [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(rows), row_begin, row_end