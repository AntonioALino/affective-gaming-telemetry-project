from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


@dataclass
class SessionRecord:
    session_id:  str
    ecg_signal:  np.ndarray
    sampling_hz: float
    valence:     Optional[float]
    arousal:     Optional[float]
    felt_emotion: Optional[str]


class MAHNOBLoader:
    ECG_CHANNEL_KEYWORDS = ("ECG", "EXG1", "EXG", "RESP")

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._validate_root()


    def available_sessions(self) -> List[Path]:
        return sorted(d for d in self._root.iterdir() if d.is_dir())

    def load_session(self, session_dir: Path) -> Optional[SessionRecord]:
        bdf_path = self._find_file(session_dir, "*.bdf")
        if bdf_path is None:
            self._warn(session_dir.name, "arquivo .bdf não encontrado")
            return None

        ecg, sfreq = self._read_ecg(bdf_path)
        if ecg is None:
            self._warn(session_dir.name, "canal ECG não localizado no BDF")
            return None

        valence, arousal, emotion = self._read_annotations(session_dir)

        return SessionRecord(
            session_id=session_dir.name,
            ecg_signal=ecg,
            sampling_hz=sfreq,
            valence=valence,
            arousal=arousal,
            felt_emotion=emotion,
        )

    def load_n_sessions(self, n: int = 20) -> List[SessionRecord]:
        sessions  = self.available_sessions()[:n]
        records   = []
        for s in sessions:
            print(f"  Carregando {s.name}...", end=" ", flush=True)
            rec = self.load_session(s)
            if rec:
                print("OK")
                records.append(rec)
            else:
                print("skip")
        print(f"  Total carregado: {len(records)}/{len(sessions)}")
        return records

    def to_dataframe(self, records: List[SessionRecord]) -> pd.DataFrame:
        rows = []
        for r in records:
            rows.append({
                "session_id":    r.session_id,
                "sampling_hz":   r.sampling_hz,
                "valence":       r.valence,
                "arousal":       r.arousal,
                "felt_emotion":  r.felt_emotion,
                "ecg_length_s":  len(r.ecg_signal) / r.sampling_hz,
            })
        return pd.DataFrame(rows)

    def _validate_root(self) -> None:
        if not self._root.exists():
            raise FileNotFoundError(
                f"Diretório MAHNOB não encontrado: {self._root}\n"
                "Baixe o dataset em https://mahnob-db.eu/hci-tagging/"
            )

    def _find_file(self, directory: Path, pattern: str) -> Optional[Path]:
        matches = list(directory.glob(pattern))
        return matches[0] if matches else None

    def _read_ecg(self, bdf_path: Path) -> tuple[Optional[np.ndarray], float]:
        import mne
        mne.set_log_level("WARNING")

        raw   = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
        sfreq = raw.info["sfreq"]

        ecg_ch = next(
            (ch for ch in raw.ch_names
             if any(k in ch.upper() for k in self.ECG_CHANNEL_KEYWORDS)),
            None
        )
        if ecg_ch is None:
            return None, sfreq

        signal = raw.get_data(picks=ecg_ch)[0]
        return signal, sfreq

    def _read_annotations(
        self, session_dir: Path
    ) -> tuple[Optional[float], Optional[float], Optional[str]]:
        xml_path = self._find_file(session_dir, "*.xml")
        if xml_path is None:
            return None, None, None

        try:
            root = ET.parse(str(xml_path)).getroot()
            valence = self._xml_float(root, "feltVlnc")
            arousal = self._xml_float(root, "feltArsl")
            emotion = self._xml_text(root, "feltEmo")
            return valence, arousal, emotion
        except ET.ParseError as e:
            print(f"    XML parse error: {e}")
            return None, None, None

    @staticmethod
    def _xml_float(root: ET.Element, tag: str) -> Optional[float]:
        el = root.find(f".//{tag}")
        if el is not None and el.text:
            try:
                return float(el.text)
            except ValueError:
                pass
        return None

    @staticmethod
    def _xml_text(root: ET.Element, tag: str) -> Optional[str]:
        el = root.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else None

    @staticmethod
    def _warn(session: str, reason: str) -> None:
        print(f"    [AVISO] {session}: {reason}")


if __name__ == "__main__":
    loader = MAHNOBLoader("./mahnob_hci")
    records = loader.load_n_sessions(n=5)
    df = loader.to_dataframe(records)
    print(df)