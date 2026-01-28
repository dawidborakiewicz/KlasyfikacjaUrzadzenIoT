import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd


class CsvFlowLabeler:
    """
    Labelowanie flow CSV po MAC (np. src_mac) na podstawie XLSX z mapowaniem MAC->klasa.
    Działa na gotowych plikach CSV.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self._mac_to_label: dict[str, str] = {}

    @staticmethod
    def norm_mac(x):
        if pd.isna(x):
            return pd.NA
        x = str(x).strip().upper()
        if not x:
            return pd.NA

        x = x.replace("-", ":")
        x = re.sub(r"\s+", "", x)

        if re.fullmatch(r"[0-9A-F]{12}", x):
            x = ":".join(x[i:i + 2] for i in range(0, 12, 2))

        if not re.fullmatch(r"([0-9A-F]{2}:){5}[0-9A-F]{2}", x):
            return pd.NA
        return x

    @staticmethod
    def pick_column(df: pd.DataFrame, wanted: str, aliases: list[str]) -> str:
        cols_map = {c.strip().lower(): c for c in df.columns}
        keys_to_try = [wanted.strip().lower()] + [a.strip().lower() for a in aliases]
        for k in keys_to_try:
            if k in cols_map:
                return cols_map[k]
        raise ValueError(f"Nie znalazłem kolumny '{wanted}' (ani aliasów {aliases}). Mam: {list(df.columns)}")

    def load_labels_from_xlsx(
        self,
        xlsx_path: str | Path,
        mac_col_name: str = "MAC Address",
        class_col_name: str = "Klasa_urzadzenia",
        mac_aliases: Optional[list[str]] = None,
        class_aliases: Optional[list[str]] = None,
    ) -> dict[str, str]:
        xlsx_path = Path(xlsx_path)
        mac_aliases = mac_aliases or ["MAC", "Mac Address", "mac address", "Adres MAC", "MAC_Address"]
        class_aliases = class_aliases or ["Klasa urzadzenia", "Klasa", "class", "device_class", "Klasa_urządzenia"]

        df = pd.read_excel(xlsx_path)
        df.columns = df.columns.str.strip()

        mac_col = self.pick_column(df, mac_col_name, aliases=mac_aliases)
        class_col = self.pick_column(df, class_col_name, aliases=class_aliases)

        df[mac_col] = df[mac_col].apply(self.norm_mac)
        df = df.dropna(subset=[mac_col])

        self._mac_to_label = dict(zip(df[mac_col], df[class_col]))
        self.log.info("Wczytano labelki: %d unikalnych MAC -> klasa", len(self._mac_to_label))
        return self._mac_to_label

    def label_csv_file(
            self,
            input_csv_path: str | Path,
            output_csv_path: str | Path,
            mac_col_in_flows: str = "src_mac",
            out_label_col: str = "klasa_urzadzen",
    ) -> Path:
        """
        Dodaje kolumnę z labelą do pojedynczego CSV.
        Nie dropuje żadnych kolumn, ale usuwa wiersze bez dopasowanej etykiety.
        """
        if not self._mac_to_label:
            raise ValueError("Brak wczytanych labeli (MAC->klasa). Najpierw wywołaj load_labels_from_xlsx().")

        input_csv_path = Path(input_csv_path)
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_csv_path)

        if mac_col_in_flows not in df.columns:
            raise KeyError(
                f"Brak kolumny '{mac_col_in_flows}' w CSV. Kolumny: {df.columns.tolist()}"
            )

        df[mac_col_in_flows] = df[mac_col_in_flows].apply(self.norm_mac)

        mapped = df[mac_col_in_flows].map(self._mac_to_label)
        mask = mapped.notna()

        removed = int((~mask).sum())
        self.log.info("Usunięto flowy bez labela: %d/%d", removed, len(df))

        df = df.loc[mask].copy()
        df[out_label_col] = mapped.loc[mask].values

        self.log.info("Zostawiono z labelami: %d", len(df))

        df.to_csv(output_csv_path, index=False)
        self.log.info("Zapisano labeled CSV: %s", output_csv_path)
        return output_csv_path

    def label_csv_in_folder(
        self,
        folder_path: str | Path,
        output_folder: str | Path,
        pattern: str = "*.csv",
        mac_col_in_flows: str = "src_mac",
        out_label_col: str = "klasa_urzadzen",
        unknown_label: str = "unknown",
    ) -> list[Path]:
        """
        Labeluje wszystkie CSV w folderze i podfolderach (pattern domyślnie *.csv),
        zapisując wynik do output_folder z zachowaniem struktury katalogów.
        """
        folder_path = Path(folder_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(folder_path.rglob(pattern))
        if not csv_files:
            raise FileNotFoundError(f"Nie znalazłem plików pasujących do {pattern} w: {folder_path}")

        out_paths: list[Path] = []
        for csv_path in csv_files:
            rel = csv_path.relative_to(folder_path)
            out_path = output_folder / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            self.label_csv_file(
                input_csv_path=csv_path,
                output_csv_path=out_path,
                mac_col_in_flows=mac_col_in_flows,
                out_label_col=out_label_col,
                unknown_label=unknown_label,
            )
            out_paths.append(out_path)

        self.log.info("Zlabelowano %d plików CSV do folderu: %s", len(out_paths), output_folder)
        return out_paths