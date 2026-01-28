# import logging
# from dataclasses import dataclass
# from decimal import Decimal
# from pathlib import Path
# from typing import Optional
#
# import pandas as pd
#
# from pyflowmeter.sniffer import create_sniffer
# from pyflowmeter.features.flow_bytes import FlowBytes
# from pyflowmeter.features.context.packet_direction import PacketDirection
#
# @dataclass
# class ExportConfig:
#     pcap_exts: tuple[str, ...] = (".pcap", ".pcapng")
#
#
# class PcapFlowExporter:
#     """
#     PCAP -> flow CSV (pyflowmeter)
#     - pcap_to_csv: jeden PCAP -> CSV
#     - folder_to_csv: wszystkie PCAPy w folderze/podfolderach -> jeden scalony CSV
#     """
#
#     def __init__(self, config: Optional[ExportConfig] = None, logger: Optional[logging.Logger] = None):
#         self.config = config or ExportConfig()
#         self.log = logger or logging.getLogger(self.__class__.__name__)
#         self._install_pyflowmeter_hotfixes()
#
#     # ============================
#     # PYFLOWMETER HOTFIXES
#     # ============================
#     @staticmethod
#     def _install_pyflowmeter_hotfixes() -> None:
#         # Fix DivisionByZero in bulk rate
#         def _safe_get_bulk_rate(self, direction: PacketDirection):
#             if direction == PacketDirection.FORWARD:
#                 size = self.feature.forward_bulk_size
#                 dur = self.feature.forward_bulk_duration
#             else:
#                 size = self.feature.backward_bulk_size
#                 dur = self.feature.backward_bulk_duration
#
#             if dur == 0 or dur == Decimal(0):
#                 return Decimal(0)
#             return size / dur
#
#         FlowBytes.get_bulk_rate = _safe_get_bulk_rate
#
#         # Fix min() empty iterable for forward header bytes
#         _orig_min_fwd_hdr = FlowBytes.get_min_forward_header_bytes
#
#         def _safe_min_fwd_hdr(self):
#             try:
#                 return _orig_min_fwd_hdr(self)
#             except ValueError:
#                 return 0
#
#         FlowBytes.get_min_forward_header_bytes = _safe_min_fwd_hdr
#
#     # ============================
#     # CORE: PCAP -> CSV
#     # ============================
#     def pcap_to_csv(self, pcap_path: str | Path, output_csv_path: str | Path) -> Path | None:
#         pcap_path = Path(pcap_path)
#         output_csv_path = Path(output_csv_path)
#         output_csv_path.parent.mkdir(parents=True, exist_ok=True)
#
#         self.log.info("PCAP -> CSV: %s -> %s", pcap_path, output_csv_path)
#
#         sniffer = create_sniffer(
#             input_file=str(pcap_path),
#             to_csv=True,
#             output_file=str(output_csv_path),
#         )
#
#         sniffer.start()
#         try:
#             # czekamy aż doczyta plik
#             sniffer.join()
#         finally:
#             # ważne: stop() często powoduje finalny flush flowów do CSV
#             try:
#                 sniffer.stop()
#             except Exception:
#                 pass
#             sniffer.join()
#
#         # Diagnostyka: AsyncSniffer potrafi mieć exception w wątku
#         exc = getattr(sniffer, "exception", None)
#         if exc:
#             self.log.warning("Sniffer error for %s: %r", pcap_path, exc)
#             return None
#
#         if not output_csv_path.exists():
#             self.log.warning("Brak CSV dla %s (nie utworzono pliku)", pcap_path)
#             return None
#
#         if output_csv_path.stat().st_size == 0:
#             # To jest OK w batchu: brak flowów
#             self.log.info("Brak flowów w PCAP (pusty CSV): %s", pcap_path)
#             return None
#
#         return output_csv_path
#
#     # ============================
#     # BATCH: folder -> one CSV
#     # ============================
#     def folder_to_csv(
#         self,
#         folder_path: str | Path,
#         output_csv_path: str | Path,
#         temp_dir: Optional[str | Path] = None,
#         add_source_pcap_column: bool = True,
#
#     ) -> Path:
#         """
#         Przechodzi po wszystkich plikach .pcap/.pcapng w folderach i podfolderach
#         i scala wyniki do jednego CSV.
#
#         - Tworzy tymczasowe CSV per PCAP
#         - Potem dopisuje wiersze do output_csv_path (append), więc nie zjada RAMu
#         - Opcjonalnie dodaje kolumnę 'source_pcap' (bardzo przydatne)
#         """
#         folder_path = Path(folder_path)
#         output_csv_path = Path(output_csv_path)
#         output_csv_path.parent.mkdir(parents=True, exist_ok=True)
#
#         temp_dir = Path(temp_dir) if temp_dir else (output_csv_path.parent / "_tmp_flows")
#         temp_dir.mkdir(parents=True, exist_ok=True)
#
#         pcap_files: list[Path] = []
#         for ext in self.config.pcap_exts:
#             pcap_files.extend(folder_path.rglob(f"*{ext}"))
#             pcap_files = [p for p in pcap_files if p.stat().st_size > 0]
#         pcap_files = sorted(
#             p for p in pcap_files
#             if not p.name.startswith("._")
#             and p.name.lower() != ".ds_store"
#             and p.name.lower() != "thumbs.db"
#         )
#
#         if not pcap_files:
#             raise FileNotFoundError(f"Nie znalazłem plików PCAP/PCAPNG w: {folder_path}")
#
#         self.log.info("Znaleziono %d plików PCAP", len(pcap_files))
#         processed = 0
#         no_flows = 0
#         failed = 0
#         skipped = 0
#         combined_written = False
#
#         for i, pcap in enumerate(pcap_files, start=1):
#             if pcap.name.startswith("._"):
#                 skipped += 1
#                 continue
#
#             safe_name = pcap.relative_to(folder_path).as_posix().replace("/", "__")
#             part_csv = temp_dir / f"{safe_name}.csv"
#
#             self.log.info("[%d/%d] Przetwarzam: %s", i, len(pcap_files), pcap)
#
#             try:
#                 csv_path = self.pcap_to_csv(pcap, part_csv)
#                 if csv_path is None:
#                     no_flows += 1
#                     continue
#
#                 part_df = pd.read_csv(csv_path)
#                 if add_source_pcap_column:
#                     part_df["source_pcap"] = pcap.as_posix()
#
#                 if not combined_written:
#                     part_df.to_csv(output_csv_path, index=False)
#                     combined_written = True
#                 else:
#                     part_df.to_csv(output_csv_path, mode="a", index=False, header=False)
#
#                 processed += 1
#
#             except Exception as e:
#                 failed += 1
#                 self.log.warning("[%d/%d] FAIL: %s | %s", i, len(pcap_files), pcap, e)
#                 continue
#
#         self.log.info(
#             "Podsumowanie: processed=%d, no_flows=%d, failed=%d, skipped=%d, total=%d",
#             processed, no_flows, failed, skipped, len(pcap_files),
#         )
#
#         if not combined_written:
#             raise RuntimeError(
#                 "Nie zapisano żadnych flowów do pliku wynikowego (wszystkie PCAPy były puste lub błędne).")
#
#         return output_csv_path