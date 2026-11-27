from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import streamlit as st
from matplotlib.ticker import FuncFormatter, LogLocator
from PIL import Image
from scipy.interpolate import PchipInterpolator

st.set_page_config(
    page_title="Respons Spektra vs Desain SNI 1726-2019",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = BASE_DIR / "rsa" / "hasil_spektra.json"
DEFAULT_INFO = BASE_DIR / "data_stasiun.txt"
DEFAULT_LOGO = BASE_DIR / "logo_bmkg.png"

TL_DEFAULT = 12.0
G_TO_GAL = 980.665
PSA_PATTERN = re.compile(r"\d{14}_(?:IA_)?([A-Z0-9]+)_((?:00)?HN[ENZ])_", re.I)

if "psa_uploader_key" not in st.session_state:
    st.session_state["psa_uploader_key"] = 0


@st.cache_data
def muat_json_spektra(json_path: str) -> dict[str, dict]:
    path = Path(json_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(item.get("kode", "")).strip().upper(): item
        for item in data
        if str(item.get("kode", "")).strip()
    }


@st.cache_data
def muat_info_stasiun(info_path: str) -> dict[str, str]:
    path = Path(info_path)
    if not path.exists():
        return {}
    df_info = pd.read_csv(path, sep="\t")
    info = {}
    for _, row in df_info.iterrows():
        kode = str(row.get("Kode", "")).strip().upper()
        detail = str(row.get("Detail", "")).strip()
        if kode:
            info[kode] = detail
    return info


def buat_array_periode(periode_maks: float, interval: float) -> np.ndarray:
    n_steps = int(np.floor(periode_maks / interval + 1e-12))
    T = np.round(np.arange(0, n_steps + 1) * interval, 8)
    T = T[T <= periode_maks + 1e-12]
    if T.size == 0 or T[-1] < periode_maks - 1e-12:
        T = np.append(T, round(float(periode_maks), 8))
    return T


def hitung_sa_gal(
    T: np.ndarray,
    sds: float,
    sd1: float,
    t0: float,
    ts: float,
    tl: float = 12.0,
) -> np.ndarray:
    sa = np.zeros_like(T, dtype=float)

    m1 = (T >= 0.0) & (T < t0)
    m2 = (T >= t0) & (T <= ts)
    m3 = (T > ts) & (T <= tl)
    m4 = T > tl

    if t0 > 0:
        sa[m1] = sds * (0.4 + 0.6 * (T[m1] / t0))
    else:
        sa[m1] = sds

    sa[m2] = sds
    sa[m3] = sd1 / T[m3]
    sa[m4] = (sd1 * tl) / (T[m4] ** 2)

    return sa * G_TO_GAL


def baca_psa_dari_bytes(raw_bytes: bytes, periode_maks: float) -> pd.DataFrame:
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=r"\s+",
        header=None,
        names=["Periode", "PSA"],
        engine="python",
    )
    df["Periode"] = pd.to_numeric(df["Periode"], errors="coerce")
    df["PSA"] = pd.to_numeric(df["PSA"], errors="coerce")
    df = df.dropna(subset=["Periode", "PSA"])
    df = df[df["Periode"] <= periode_maks]
    df = df.sort_values(by="Periode").drop_duplicates(subset=["Periode"], keep="first")
    df["PSA"] *= 100
    return df


def extract_datetime_from_filename(filename: str) -> str:
    try:
        timestamp = filename[:14]
        date_time = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        wib_zone = pytz.timezone("Asia/Jakarta")
        date_time_wib = pytz.utc.localize(date_time).astimezone(wib_zone)
        return date_time_wib.strftime("%d %B %Y %H:%M:%S") + " WIB"
    except Exception:
        return ""


def plot_psa_spline(ax, x, y, label: str, color: str, style: str = "-", min_positive: float | None = None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]

    if len(x_unique) >= 3:
        try:
            x_new = np.linspace(x_unique.min(), x_unique.max(), 500)
            pchip = PchipInterpolator(x_unique, y_unique)
            y_smooth = pchip(x_new)
            if min_positive is not None:
                y_smooth = np.where(y_smooth < min_positive, min_positive, y_smooth)
            else:
                y_smooth = np.where(y_smooth < 0, 0, y_smooth)
            ax.plot(x_new, y_smooth, color=color, linestyle=style, linewidth=1.8, label=label)
            return
        except Exception:
            pass

    if min_positive is not None:
        y_unique = np.where(y_unique < min_positive, min_positive, y_unique)
    ax.plot(x_unique, y_unique, color=color, linestyle=style, linewidth=1.8, label=label)


def _format_log_y(val: float, _pos: int) -> str:
    if val <= 0:
        return ""
    if val < 1:
        return f"{val:.4f}".rstrip("0").rstrip(".")
    if np.isclose(val, round(val)):
        return f"{int(round(val))}"
    return f"{val:g}"


def tambah_logo_kiri_atas_judul(fig, ax, logo_path: str | Path = DEFAULT_LOGO):
    path = Path(logo_path)
    if not path.exists():
        return
    try:
        img = Image.open(path).convert("RGBA")
        ax_pos = ax.get_position()
        logo_size = 0.085
        logo_x = max(0.02, ax_pos.x0 - 0.01)
        logo_y = min(0.985 - logo_size, ax_pos.y1 + 0.02)
        logo_ax = fig.add_axes([logo_x, logo_y, logo_size, logo_size])
        logo_ax.imshow(img)
        logo_ax.axis("off")
    except Exception:
        pass


def buat_figure_stasiun(
    stasiun: str,
    komponen_data: dict[str, pd.DataFrame],
    desain_data: dict[str, pd.DataFrame],
    timestamp_str: str,
    lokasi_detail: str,
    periode_maks: float,
    ylim_min: float,
    ylim_max: float,
    use_log_y: bool,
    warna_komp: dict[str, str],
    warna_desain: dict[str, str],
    figsize_tuple: tuple[float, float],
    logo_path: str | Path = DEFAULT_LOGO,
):
    fig, ax = plt.subplots(figsize=figsize_tuple)
    label_klas = {"E": "(SE)", "D": "(SD)", "C": "(SC)"}

    min_pos = max(1e-6, float(ylim_min) * 0.1) if use_log_y else None

    for komp in ["HNE", "HNN", "HNZ"]:
        if komp in komponen_data and not komponen_data[komp].empty:
            df_k = komponen_data[komp]
            plot_psa_spline(
                ax,
                df_k["Periode"].values,
                df_k["PSA"].values,
                f"Respons Spektra {komp}",
                warna_komp[komp],
                "-",
                min_positive=min_pos,
            )

    for klas in ["E", "D", "C"]:
        df_d = desain_data.get(klas)
        if df_d is not None and not df_d.empty:
            y_vals = df_d["SA"].values
            if use_log_y and min_pos is not None:
                y_vals = np.where(y_vals < min_pos, min_pos, y_vals)
            ax.plot(
                df_d["Periode"].values,
                y_vals,
                color=warna_desain[klas],
                linestyle="--",
                linewidth=1.8,
                label=f"Desain Respon Spektra SNI 1726-2019 {label_klas[klas]}",
            )

    if use_log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_log_y))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

    judul_lokasi = f"Stasiun Akselerograf {stasiun}" + (f" - {lokasi_detail}" if lokasi_detail else "")
    ot_line = f"OT: {timestamp_str}\n" if timestamp_str else ""
    ax.set_title(
        f"Respon Spektra Vs Desain Respon Spektra\n"
        f"{judul_lokasi}\n"
        f"{ot_line}"
        f"Badan Meteorologi Klimatologi dan Geofisika (BMKG)",
        fontsize=10.5,
    )
    ax.set_xlabel("Periode (detik)", fontsize=8)
    ax.set
