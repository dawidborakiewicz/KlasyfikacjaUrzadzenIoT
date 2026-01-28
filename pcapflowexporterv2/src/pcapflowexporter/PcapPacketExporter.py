import csv
import subprocess
import sys
from pathlib import Path
from shutil import which
import re

import pandas as pd

RESOURCE_FOLDER = Path(r"../../res_pcaps/")
OUT_CSV = Path("tcp_ana_and_protocols.csv")

LABELING_ENABLED = True
MAPPING_XLSX = Path("../../res_pcaps/device_group_labels_6classes.xlsx")

BASE_DISPLAY_FILTER = "ip"
PCAP_EXTS = (".pcap", ".pcapng")

TSHARK_FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "frame.time_delta",
    "frame.time_delta_displayed",

    "eth.src",

    "ip.src",
    "ip.proto",
    "ip.len", "ip.ttl", "ip.flags.df",
    "ip.dsfield.dscp",

    "tcp.srcport", "tcp.dstport",
    "tcp.len",
    "tcp.hdr_len",
    "tcp.flags",
    "tcp.window_size_value",

    "udp.srcport", "udp.dstport",
    "udp.length",

    "frame.protocols",
]


def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def norm_mac(s: str) -> str | None:
    if s is None:
        return None
    s = str(s).strip().lower()
    if not s:
        return None
    s = s.replace("-", ":")
    s = re.sub(r"\s+", "", s)

    if re.fullmatch(r"[0-9a-f]{12}", s):
        s = ":".join(s[i:i+2] for i in range(0, 12, 2))

    if not re.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", s):
        return None
    return s


def load_mac_labels(mapping_xlsx: Path) -> dict[str, str]:
    df = pd.read_excel(mapping_xlsx)
    required = {"MAC Address", "Klasa_urzadzenia"}
    if not required.issubset(set(df.columns)):
        die(f"Mapping musi mieć kolumny {required}. Jest: {list(df.columns)}")

    out: dict[str, str] = {}
    for mac_raw, cls in zip(df["MAC Address"], df["Klasa_urzadzenia"]):
        mac = norm_mac(mac_raw)
        if mac and pd.notna(cls):
            out[mac] = str(cls)

    if not out:
        die("Mapping nie zawiera poprawnych MAC-ów.")
    return out


def build_display_filter(labeling: bool, macs: list[str]) -> str:
    base = BASE_DISPLAY_FILTER.strip() or "frame"
    if not labeling:
        return base

    mac_set = ", ".join(macs)
    return f"eth.src in {{{mac_set}}} && {base}"


def run_tshark_lines(pcap: Path, display_filter: str):
    cmd = [
        "tshark", "-n",
        "-r", str(pcap),
        "-Y", display_filter,
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f",
    ]
    for f in TSHARK_FIELDS:
        cmd += ["-e", f]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    assert proc.stderr is not None

    for line in proc.stdout:
        yield line.rstrip("\n")

    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read()
        raise RuntimeError(err[:4000])


def iter_pcaps(root: Path):
    for ext in PCAP_EXTS:
        for p in root.rglob(f"*{ext}"):
            if p.name.startswith("._"):
                continue
            yield p


def export_folder_to_one_csv(
    resource_folder: Path,
    out_csv: Path,
    labeling_enabled: bool,
    mapping_xlsx: Path | None,
    continue_on_error: bool = True,
):
    if which("tshark") is None:
        die("Nie widzę 'tshark' w PATH. Zainstaluj Wireshark/tshark.")
    if not resource_folder.exists():
        die(f"Nie ma folderu: {resource_folder}")

    mac_to_label: dict[str, str] = {}
    if labeling_enabled:
        if mapping_xlsx is None or not mapping_xlsx.exists():
            die("Labeling włączony, ale nie ma mappingu XLSX.")
        mac_to_label = load_mac_labels(mapping_xlsx)

    display_filter = build_display_filter(labeling_enabled, sorted(mac_to_label.keys()))

    pcaps = sorted(set(iter_pcaps(resource_folder)))
    if not pcaps:
        die(f"Nie znaleziono PCAPów w: {resource_folder}")

    drop_output_fields = {"eth.src", "ip.src", "ip.dst"} if labeling_enabled else set()

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    total_rows = 0
    bad_pcaps = 0
    empty_pcaps = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)

        header_cols = None
        eth_src_idx = None
        keep_indices: list[int] = []

        for i, pcap in enumerate(pcaps, start=1):
            print(f"[{i}/{len(pcaps)}] {pcap}")

            try:
                lines = run_tshark_lines(pcap, display_filter)
                header = next(lines, None)
                if not header:
                    empty_pcaps += 1
                    continue

                header_cols = next(csv.reader([header]))

                if eth_src_idx is None:
                    try:
                        eth_src_idx = header_cols.index("eth.src")
                    except ValueError:
                        die(f"Brak kolumny eth.src w tshark output. Header: {header_cols}")

                if not keep_indices:
                    keep_indices = [idx for idx, name in enumerate(header_cols) if name not in drop_output_fields]

                if not wrote_header:
                    out_header = [header_cols[idx] for idx in keep_indices]
                    if labeling_enabled:
                        out_header.append("label")
                    writer.writerow(out_header)
                    wrote_header = True

                rows_written = 0

                for line in lines:
                    if not line:
                        continue

                    cols = next(csv.reader([line]))
                    if len(cols) < len(header_cols):
                        cols += [""] * (len(header_cols) - len(cols))

                    if labeling_enabled:
                        mac = norm_mac(cols[eth_src_idx])
                        if not mac:
                            continue
                        label = mac_to_label.get(mac)
                        if label is None:
                            continue

                        out_row = [cols[idx] for idx in keep_indices]
                        out_row.append(label)
                    else:
                        out_row = [cols[idx] for idx in keep_indices]

                    writer.writerow(out_row)
                    rows_written += 1

                if rows_written == 0:
                    empty_pcaps += 1
                total_rows += rows_written

            except Exception as e:
                bad_pcaps += 1
                if continue_on_error:
                    print(f"WARN: tshark failed for {pcap}: {e}", file=sys.stderr)
                    continue
                die(f"tshark failed for {pcap}: {e}")

    if not wrote_header:
        die("Nie zapisano żadnych wierszy (wszystkie PCAPy puste / błędne / przefiltrowane).")

    print(f"OK: {out_csv} | rows={total_rows} | empty_pcaps={empty_pcaps} | bad_pcaps={bad_pcaps}")


def export_single_pcap_to_csv(
    pcap_path: str | Path,
    out_csv_path: str | Path,
    labeling_enabled: bool = False,
    mapping_xlsx: str | Path | None = None,
) -> Path:
    """
    Eksportuje pojedynczy plik PCAP/PCAPNG do CSV przy użyciu tshark.

    - labeling_enabled=False:
        surowy eksport wszystkich pól z TSHARK_FIELDS (bez labeli)
    - labeling_enabled=True:
        * wymaga mapping_xlsx (MAC Address -> Klasa_urzadzenia)
        * odrzuca pakiety bez mapowania MAC
        * dodaje kolumnę 'label'
        * usuwa z outputu: eth.src, ip.src, ip.dst
    """
    pcap_path = Path(pcap_path)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not pcap_path.exists():
        die(f"Nie ma pliku PCAP: {pcap_path}")

    mac_to_label: dict[str, str] = {}
    if labeling_enabled:
        if mapping_xlsx is None:
            die("Labeling włączony, ale nie podano mapping_xlsx.")
        mac_to_label = load_mac_labels(Path(mapping_xlsx))

    display_filter = build_display_filter(labeling_enabled, sorted(mac_to_label.keys()))

    drop_output_fields = {"eth.src", "ip.src", "ip.dst"} if labeling_enabled else set()

    lines = run_tshark_lines(pcap_path, display_filter)
    header_line = next(lines, None)
    if not header_line:
        die(f"Pusty output tshark dla: {pcap_path} (brak pakietów po filtrze?)")

    header_cols = next(csv.reader([header_line]))

    eth_src_idx = None
    if labeling_enabled:
        try:
            eth_src_idx = header_cols.index("eth.src")
        except ValueError:
            die(f"Brak kolumny eth.src w tshark output. Header: {header_cols}")

    keep_indices = [i for i, name in enumerate(header_cols) if name not in drop_output_fields]

    wrote_any = False
    with out_csv_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)

        out_header = [header_cols[i] for i in keep_indices]
        if labeling_enabled:
            out_header.append("label")
        writer.writerow(out_header)

        for line in lines:
            if not line:
                continue

            cols = next(csv.reader([line]))
            if len(cols) < len(header_cols):
                cols += [""] * (len(header_cols) - len(cols))

            if labeling_enabled:
                mac = norm_mac(cols[eth_src_idx])  
                if not mac:
                    continue
                label = mac_to_label.get(mac)
                if label is None:
                    continue  

            out_row = [cols[i] for i in keep_indices]
            if labeling_enabled:
                out_row.append(label) 

            writer.writerow(out_row)
            wrote_any = True

    if not wrote_any:
        die(f"Nie zapisano żadnych wierszy dla: {pcap_path} (wszystko odfiltrowane?)")

    return out_csv_path




if __name__ == "__main__":
    # export_folder_to_one_csv(
    #     resource_folder=RESOURCE_FOLDER,
    #     out_csv=OUT_CSV,
    #     labeling_enabled=LABELING_ENABLED,
    #     mapping_xlsx=MAPPING_XLSX if LABELING_ENABLED else None,
    #     continue_on_error=True,
    # )
    export_single_pcap_to_csv(
        pcap_path=r"..\..\res_pcaps\2021_11_03_Idle.pcap",
        out_csv_path="single.csv",
        labeling_enabled=True,
        mapping_xlsx="../../res_pcaps/device_group_labels_6classes.xlsx",
    )
