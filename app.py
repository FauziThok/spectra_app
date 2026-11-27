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
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import PchipInterpolator

st.set_page_config(
    page_title="Respons Spektra vs Desain SNI 1726-2019",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = BASE_DIR / "rsa" / "hasil_spektra.json"
DEFAULT_INFO = BASE_DIR / "data_stasiun.txt"

TL_DEFAULT = 12.0
G_TO_GAL = 980.665
PSA_PATTERN = re.compile(r"\d{14}_(?:IA_)?([A-Z0-9]+)_((?:00)?HN[ENZ])_", re.I)


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
):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    warna_komp = {"HNE": "black", "HNN": "green", "HNZ": "blue"}
    warna_desain = {"E": "red", "D": "orange", "C": "gold"}
    label_klas = {"E": "(SE)", "D": "(SD)", "C": "(SC)"}

    min_pos = 1e-6 if use_log_y else None

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
            if use_log_y:
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
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

    judul_lokasi = f"Stasiun Akselerograf {stasiun}" + (f" - {lokasi_detail}" if lokasi_detail else "")
    ot_line = f"OT: {timestamp_str}\n" if timestamp_str else ""
    ax.set_title(
        f"Respon Spektra Vs Desain Respon Spektra\n"
        f"{judul_lokasi}\n"
        f"{ot_line}"
        f"Badan Meteorologi Klimatologi dan Geofisika (BMKG)",
        fontsize=10.5,
    )
    ax.set_xlabel("Periode (detik)", fontsize=10.5)
    ax.set_ylabel("SA (gal)", fontsize=10.5)
    ax.set_xlim(0, periode_maks)
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig


st.title("Respons Spektra vs Desain Spektra (SNI 1726-2019)")

with st.sidebar:
    st.header("Pengaturan")
    periode_maks = st.number_input(
        "Periode Maksimum (detik)",
        min_value=0.5,
        max_value=20.0,
        value=4.0,
        step=0.5,
    )
    interval = st.number_input(
        "Interval Periode (detik)",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.01,
        format="%.3f",
    )

    st.divider()
    st.subheader("Pengaturan Sumbu Y")
    use_log_y = st.checkbox("Gunakan skala logaritmik (log) sumbu Y", value=True)
    samakan_skala_y = st.checkbox("Samakan skala sumbu Y antar stasiun", value=True)
    batas_y_manual = st.checkbox(
        "Atur batas sumbu Y manual",
        value=True if use_log_y else False,
    )

    if batas_y_manual:
        default_ymin = 0.01 if use_log_y else 0.0
        default_ymax = 10000.0 if use_log_y else 500.0
        min_allowed_y = 1e-6 if use_log_y else 0.0

        ymin_input = st.number_input(
            "Batas Bawah Y (gal)",
            min_value=min_allowed_y,
            value=default_ymin,
            step=0.01 if use_log_y else 10.0,
            format="%.4f" if use_log_y else "%.2f",
        )
        ymax_input = st.number_input(
            "Batas Atas Y (gal)",
            min_value=min_allowed_y * 10,
            value=default_ymax,
            step=100.0 if use_log_y else 10.0,
            format="%.2f",
        )
        if ymax_input <= ymin_input:
            st.error("Batas Atas Y harus lebih besar dari Batas Bawah Y.")

    st.divider()
    json_path_input = st.text_input("Path hasil_spektra.json", value=str(DEFAULT_JSON))
    info_path_input = st.text_input("Path data_stasiun.txt (opsional)", value=str(DEFAULT_INFO))

by_kode = muat_json_spektra(json_path_input)
info_stasiun = muat_info_stasiun(info_path_input)

if not by_kode:
    st.warning(f"Data dari `{json_path_input}` belum terbaca atau kosong.")

uploaded_files = st.file_uploader(
    "Upload file *.psa5",
    type=["psa5"],
    accept_multiple_files=True,
)

if uploaded_files:
    T = buat_array_periode(periode_maks, interval)
    data_stasiun: dict[str, dict[str, pd.DataFrame]] = {}
    filename_per_stasiun: dict[str, str] = {}

    for up in uploaded_files:
        m = PSA_PATTERN.match(up.name)
        if m:
            st_kode = m.group(1).upper()
            komp = m.group(2).upper()[-3:]
            data_stasiun.setdefault(st_kode, {})[komp] = baca_psa_dari_bytes(
                up.getvalue(), periode_maks=periode_maks
            )
            filename_per_stasiun.setdefault(st_kode, up.name)
        else:
            st.info(f"Lewati `{up.name}` (format nama tidak sesuai pola).")

    desain_per_stasiun: dict[str, dict[str, pd.DataFrame]] = {}
    max_y_per_stasiun: dict[str, float] = {}

    for st_kode, komp_map in data_stasiun.items():
        if not komp_map:
            continue
        max_psa = max(df_k["PSA"].max() for df_k in komp_map.values())
        max_desain = 0.0

        item = by_kode.get(st_kode)
        if item:
            spektra_map = item.get("spektra", {})
            for kelas in ["SC", "SD", "SE"]:
                param = spektra_map.get(kelas) or spektra_map.get(kelas.lower())
                if param and None not in (param.get("T0"), param.get("Ts"), param.get("SDS"), param.get("SD1")):
                    sa_gal = hitung_sa_gal(
                        T=T,
                        sds=float(param["SDS"]),
                        sd1=float(param["SD1"]),
                        t0=float(param["T0"]),
                        ts=float(param["Ts"]),
                        tl=TL_DEFAULT,
                    )
                    klas_huruf = kelas[-1].upper()  # C, D, E
                    desain_per_stasiun.setdefault(st_kode, {})[klas_huruf] = pd.DataFrame({
                        "Periode": T,
                        "SA": sa_gal,
                    })
                    max_desain = max(max_desain, float(np.max(sa_gal)))
        else:
            st.warning(f"Kode stasiun `{st_kode}` tidak ditemukan di `{json_path_input}` — hanya menampilkan PSA.")

        max_y_per_stasiun[st_kode] = max(float(max_psa), max_desain)

    global_ylim = (max(max_y_per_stasiun.values()) * 1.2) if max_y_per_stasiun else 1.0

    for st_kode, komp_map in data_stasiun.items():
        if not komp_map:
            continue

        if batas_y_manual:
            ylim_min_val = float(ymin_input)
            ylim_max_val = float(ymax_input) if float(ymax_input) > float(ymin_input) else float(ymin_input) * 10.0
        else:
            ylim_min_val = 0.01 if use_log_y else 0.0
            ylim_max_val = global_ylim if samakan_skala_y else max(max_y_per_stasiun.get(st_kode, 1.0) * 1.2, 1.0)
            if use_log_y and ylim_max_val <= ylim_min_val:
                ylim_max_val = ylim_min_val * 100.0

        ts_str = extract_datetime_from_filename(filename_per_stasiun.get(st_kode, ""))
        loc_str = info_stasiun.get(st_kode, "")

        fig = buat_figure_stasiun(
            stasiun=st_kode,
            komponen_data=komp_map,
            desain_data=desain_per_stasiun.get(st_kode, {}),
            timestamp_str=ts_str,
            lokasi_detail=loc_str,
            periode_maks=periode_maks,
            ylim_min=ylim_min_val,
            ylim_max=ylim_max_val,
            use_log_y=use_log_y,
        )

        st.subheader(f"Stasiun: {st_kode}" + (f" — {loc_str}" if loc_str else ""))

        col_plot, _ = st.columns([1.3, 1])
        with col_plot:
            st.pyplot(fig, use_container_width=True)

        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=300)
        buf_png.seek(0)
        st.download_button(
            label=f"Download Plot SA_{st_kode}.png",
            data=buf_png,
            file_name=f"SA_{st_kode}.png",
            mime="image/png",
            key=f"dl_png_{st_kode}",
        )
        plt.close(fig)

        desain_st = desain_per_stasiun.get(st_kode, {})
        if desain_st:
            st.caption(f"Download File CSV Spektrum Desain (Sa dalam gal) — Stasiun {st_kode}:")
            zip_buf = io.BytesIO()
            kelas_tersedia = [k for k in ["C", "D", "E"] if k in desain_st]
            cols_csv = st.columns(len(kelas_tersedia))

            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx_k, klas_huruf in enumerate(kelas_tersedia):
                    df_sa_out = pd.DataFrame({
                        "T": desain_st[klas_huruf]["Periode"].values,
                        "Sa": desain_st[klas_huruf]["SA"].values,
                    })
                    csv_str = df_sa_out.to_csv(index=False)
                    csv_filename = f"{st_kode}_s{klas_huruf.lower()}.csv"
                    zf.writestr(csv_filename, csv_str)

                    with cols_csv[idx_k]:
                        st.download_button(
                            label=f"Download {csv_filename}",
                            data=csv_str.encode("utf-8"),
                            file_name=csv_filename,
                            mime="text/csv",
                            key=f"dl_csv_{st_kode}_{klas_huruf}",
                        )

            zip_buf.seek(0)
            st.download_button(
                label=f"Download Semua CSV {st_kode} (.zip)",
                data=zip_buf,
                file_name=f"{st_kode}_sa_csv.zip",
                mime="application/zip",
                key=f"dl_zip_{st_kode}",
            )
else:
    st.caption("Silakan unggah satu atau beberapa file `.psa5` untuk menampilkan grafik.")
