import os
import zipfile
from pathlib import Path

import requests
import pandas as pd

# For Matplotlib backend
import matplotlib.pyplot as plt

# For Plotly backend (comment these two lines if you don't want Plotly at all)
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://raman-baseline-api.onrender.com"
ZIP_OUTPUT = "batch_results.zip"
EXTRACT_DIR = "baseline_xlsx"

# Choose plotting backend: "plotly" or "matplotlib"
PLOT_BACKEND = "plotly" 

# Local .txt input spectra
TXT_PATHS = [
    "Lysozyme TA 25_01.txt",
    "BSA dentro TA 30_01_3.txt",
    "GPS 20.txt",
    "SARS-PMI-RS-134-N-Protein-Lyophilized-1-785nm-G600x100-3A.txt",
]

# Methods to apply (names must match the API)
'''
based_on_local_minima = "Based on local minima"
based_on_polynomials = "Based on Polynomials"
based_on_wavelets = "Based on Wavelets"
bubblefill = "BubbleFill"
vancouver = "Vancouver"
consensus_modeling = "Consensus Modeling"
''' 

METHODS = [
    "Vancouver",
    "Based on Wavelets",
    "Consensus Modeling",
]

# Methods and weights used INSIDE consensus modeling
  
CONSENSUS_METHODS = [
    "Vancouver",
    "Based on Wavelets",
]
CONSENSUS_WEIGHTS = [0.3, 0.7]


# ============================================================
# 1) Call API: Multiple_Spectra_Analysis
# ============================================================

def run_multiple_spectra(
    txt_paths,
    methods,
    consensus_methods=None,
    consensus_weights=None,
    out_zip="baseline_results.zip",
):
    """
    Send several .txt spectra to /Multiple_Spectra_Analysis and save the ZIP
    with the resulting .xlsx files.
    """
    url = f"{BASE_URL}/Multiple_Spectra_Analysis"

    # Build query parameters
    params = []
    for m in methods:
        params.append(("Methods", m))

    if consensus_methods:
        for m in consensus_methods:
            params.append(("Consensus_methods", m))
    if consensus_weights:
        for w in consensus_weights:
            params.append(("Consensus_weights", str(w)))

    # Build files list
    files = []
    for p in txt_paths:
        files.append(("files", (os.path.basename(p), open(p, "rb"), "text/plain")))

    r = requests.post(url, params=params, files=files)

    print("Status code:", r.status_code)
    if r.ok:
        with open(out_zip, "wb") as f:
            f.write(r.content)
        print("ZIP saved as:", out_zip)
    else:
        raise RuntimeError(f"API error:\n{r.text}")


# ============================================================
# 2) Unzip and return path to first XLSX
# ============================================================

def extract_first_xlsx(zip_path, extract_dir):
    """
    Extract the first .xlsx file inside the ZIP and return its path.
    """
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError("No .xlsx files found inside the ZIP.")
        first_xlsx_name = xlsx_names[0]
        zf.extract(first_xlsx_name, extract_dir)
        print("First XLSX extracted:", first_xlsx_name)

    return Path(extract_dir) / first_xlsx_name


# ============================================================
# 3) Rename / reorder columns for nicer labels
# ============================================================

def prepare_dataframe(xlsx_path):
    """
    Read the Excel file, rename columns, and reorder them.
    Returns the processed DataFrame and a pretty title based on filename.
    """
    df = pd.read_excel(xlsx_path)

    # Column containing the Raman shift
    x_col_name = "x"

    # Mapping from API column names to plot labels
    rename_map = {
        "y_original": "Raw spectrum",
        "y_no_baseline_Based_on_local_minima": "Method based on local minima",
        "y_no_baseline_Based_on_Polynomials": "Weighted Piecewise Polynomial Method",
        "y_no_baseline_Based_on_Wavelets": "Wavelets-based Method",
        "y_no_baseline_BubbleFill": "BubbleFill Method",
        "y_no_baseline_Vancouver": "Vancouver Method",
        "y_no_baseline_Consensus_Modeling": "Consensus Modeling",
    }
    df = df.rename(columns=rename_map)

    # Desired plotting order
    desired_order = [
        "Raw spectrum",
        "Vancouver Method",
        "Weighted Piecewise Polynomial Method",
        "Wavelets-based Method",
        "Method based on local minima",
        "BubbleFill Method",
        "Consensus Modeling",
    ]

    # Keep only columns that exist for this file
    y_cols_present = [c for c in desired_order if c in df.columns]
    final_cols = [x_col_name] + y_cols_present
    df = df[final_cols]

    title_name = Path(xlsx_path).stem
    return df, title_name


# ============================================================
# 4a) Plot with Plotly
# ============================================================

def plot_with_plotly(df, title_name):
    """
    Plot stacked Raman spectra using Plotly.
    """
    x = df.iloc[:, 0]
    y_cols = df.columns[1:]
    n_plots = len(y_cols)

    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=y_cols,
    )

    for i, col in enumerate(y_cols, start=1):
        fig.add_trace(
            go.Scatter(x=x, y=df[col], mode="lines", name=col),
            row=i, col=1
        )

    # Common Y range for plots 2..n (methods only)
    if n_plots > 1:
        df_rest = df[list(y_cols[1:])]
        ymin = df_rest.min().min()
        ymax = df_rest.max().max()
    else:
        ymin = df[y_cols[0]].min()
        ymax = df[y_cols[0]].max()

    fig.update_layout(
        template="plotly_white",
        height=250 * n_plots,
        title={
            "text": f"Raman Spectra {title_name}",
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=False,
    )

    fig.update_xaxes(title_text="Raman shift (cm⁻¹)", row=n_plots, col=1)

    for i in range(1, n_plots + 1):
        if i == 1:
            fig.update_yaxes(title_text="Intensity (a.u.)", row=i, col=1)
        else:
            fig.update_yaxes(
                title_text="Intensity (a.u.)",
                range=[ymin, ymax],
                row=i, col=1,
            )

    fig.show(renderer="browser")

    # Optional: save high-resolution PNG (requires `kaleido`)
    # fig.write_image(f"{title_name}.png", width=1200, height=250 * n_plots, scale=3)


# ============================================================
# 4b) Plot with Matplotlib
# ============================================================

def plot_with_matplotlib(df, title_name):
    """
    Plot stacked Raman spectra using Matplotlib.
    """
    x = df.iloc[:, 0]
    y_cols = df.columns[1:]
    n_plots = len(y_cols)

    fig, axes = plt.subplots(
        n_plots, 1, sharex=True, figsize=(8, 2.5 * n_plots)
    )

    # If there is only one subplot, axes is not a list → wrap it
    if n_plots == 1:
        axes = [axes]

    # Common Y range for plots 2..n
    if n_plots > 1:
        df_rest = df[list(y_cols[1:])]
        ymin = df_rest.min().min()
        ymax = df_rest.max().max()
    else:
        ymin = df[y_cols[0]].min()
        ymax = df[y_cols[0]].max()

    for ax, col in zip(axes, y_cols):
        ax.plot(x, df[col], linewidth=1)
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(col, fontsize=9)
        if col != y_cols[0]:  # methods only, not the raw spectrum
            ax.set_ylim(ymin, ymax)

    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.suptitle(f"Raman Spectra {title_name}", y=0.98)
    fig.tight_layout()
    fig.show()
    # Optional: fig.savefig(f"{title_name}_mpl.png", dpi=300)


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    # 1) Call the API and get ZIP
    run_multiple_spectra(
        txt_paths=TXT_PATHS,
        methods=METHODS,
        consensus_methods=CONSENSUS_METHODS,
        consensus_weights=CONSENSUS_WEIGHTS,
        out_zip=ZIP_OUTPUT,
    )

    # 2) Extract first XLSX and load
    xlsx_path = extract_first_xlsx(ZIP_OUTPUT, EXTRACT_DIR)
    df, title_name = prepare_dataframe(xlsx_path)

    # 3) Plot with the selected backend
    if PLOT_BACKEND.lower() == "plotly":
        plot_with_plotly(df, title_name)
    elif PLOT_BACKEND.lower() == "matplotlib":
        plot_with_matplotlib(df, title_name)
    else:
        raise ValueError("PLOT_BACKEND must be 'plotly' or 'matplotlib'.")
