from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Respons Spektra vs Desain SNI 1726-2019",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = BASE_DIR / "rsa" / "hasil_spektra.json"
DEFAULT_INFO = BASE_DIR / "data_stasiun.txt"

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
    json_path_input = st.text_input("Path hasil_spektra.json", value=str(DEFAULT_JSON))
    info_path_input = st.text_input("Path data_stasiun.txt (opsional)", value=str(DEFAULT_INFO))

by_kode = muat_json_spektra(json_path_input)
info_stasiun = muat_info_stasiun(info_path_input)

if not by_kode:
    st.warning(f"Data dari `{json_path_input}` belum terbaca atau kosong.")
else:
    st.success(f"Berhasil memuat {len(by_kode)} stasiun dari `{json_path_input}`.")

uploaded_files = st.file_uploader(
    "Upload file *.psa5",
    type=["psa5"],
    accept_multiple_files=True,
)

if uploaded_files:
    terdeteksi: dict[str, list[str]] = {}
    for up in uploaded_files:
        m = PSA_PATTERN.match(up.name)
        if m:
            st_kode = m.group(1).upper()
            komp = m.group(2).upper()[-3:]
            terdeteksi.setdefault(st_kode, []).append(komp)
        else:
            st.info(f"Lewati `{up.name}` (format nama tidak sesuai pola).")

    for st_kode, komps in terdeteksi.items():
        detail = info_stasiun.get(st_kode, "")
        ada_di_json = "Ada di JSON" if st_kode in by_kode else "Tidak ada di JSON"
        st.write(
            f"- **{st_kode}**"
            + (f" ({detail})" if detail else "")
            + f" -> komponen: `{', '.join(sorted(set(komps)))}` | *{ada_di_json}*"
        )