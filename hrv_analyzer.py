from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class HRVMetrics:
    session_id:  str
    RMSSD:       float
    MeanNN:      float
    SDNN:        float
    pNN50:       float
    condition:   Optional[str] = None
    simulated:   bool = False


class HRVAnalyzer:
    VALENCE_BINS   = [0, 3, 6, 9]
    VALENCE_LABELS = ["Negative", "Neutral", "Positive"]

    def __init__(self) -> None:
        self._check_import()

    def from_ecg(
        self,
        ecg_signal: np.ndarray,
        sampling_hz: float,
        session_id: str = "unknown",
        valence: Optional[float] = None,
    ) -> Optional[HRVMetrics]:
        import neurokit2 as nk

        try:
            _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=int(sampling_hz))
            hrv_df    = nk.hrv_time(rpeaks, sampling_rate=int(sampling_hz), show=False)

            return HRVMetrics(
                session_id = session_id,
                RMSSD      = self._safe_float(hrv_df, "HRV_RMSSD"),
                MeanNN     = self._safe_float(hrv_df, "HRV_MeanNN"),
                SDNN       = self._safe_float(hrv_df, "HRV_SDNN"),
                pNN50      = self._safe_float(hrv_df, "HRV_pNN50"),
                condition  = self._classify_valence(valence),
            )
        except Exception as exc:
            print(f"  [HRVAnalyzer] Erro em {session_id}: {exc}")
            return None

    def from_sessions(self, sessions: list) -> pd.DataFrame:
        metrics: List[HRVMetrics] = []

        for rec in sessions:
            result = self.from_ecg(
                ecg_signal  = rec.ecg_signal,
                sampling_hz = rec.sampling_hz,
                session_id  = rec.session_id,
                valence     = rec.valence,
            )
            if result:
                metrics.append(result)

        return self.to_dataframe(metrics)

    def from_hr_timeseries(
        self,
        df: pd.DataFrame,
        hr_col: str = "hr",
        time_col: str = "elapsed_s",
    ) -> pd.DataFrame:

        hr   = df[hr_col].dropna().values
        time = df[time_col].dropna().values[:len(hr)]

        if len(hr) < 10:
            return pd.DataFrame()

        nn_approx = 60_000.0 / np.clip(hr, 30, 200)
        diff_nn   = np.diff(nn_approx)

        rows = []
        window_size = 30
        step        = 15

        start = 0
        while start + window_size < time[-1]:
            mask = (time >= start) & (time < start + window_size)
            nn_win   = nn_approx[mask]
            diff_win = np.diff(nn_win)

            if len(nn_win) < 5:
                start += step
                continue

            rows.append({
                "window_start_s": start,
                "RMSSD_proxy":    float(np.sqrt(np.mean(diff_win ** 2))) if len(diff_win) else np.nan,
                "MeanNN_proxy":   float(nn_win.mean()),
                "SDNN_proxy":     float(nn_win.std()),
                "HR_mean":        float(hr[mask].mean()) if mask.any() else np.nan,
                "HR_std":         float(hr[mask].std())  if mask.any() else np.nan,
            })
            start += step

        return pd.DataFrame(rows)

    @staticmethod
    def to_dataframe(metrics: List[HRVMetrics]) -> pd.DataFrame:
        return pd.DataFrame([vars(m) for m in metrics])


    def _classify_valence(self, valence: Optional[float]) -> Optional[str]:
        if valence is None or np.isnan(valence):
            return None
        idx = np.digitize(valence, self.VALENCE_BINS) - 1
        idx = np.clip(idx, 0, len(self.VALENCE_LABELS) - 1)
        return self.VALENCE_LABELS[idx]

    @staticmethod
    def _safe_float(df: pd.DataFrame, col: str) -> float:
        try:
            return float(df[col].values[0])
        except (KeyError, IndexError, TypeError):
            return np.nan

    @staticmethod
    def _check_import() -> None:
        try:
            import neurokit2
        except ImportError:
            raise ImportError(
                "neurokit2 não encontrado. Execute: pip install neurokit2"
            )


if __name__ == "__main__":
    import numpy as np
    from data_simulator import DataSimulator

    sim = DataSimulator()
    ts_df, _ = sim.raw_hr_timeseries(duration_seconds=300, n_events=3)

    analyzer = HRVAnalyzer()
    hrv_windows = analyzer.from_hr_timeseries(ts_df)
    print("HRV por janela (Apple Watch proxy):")
    print(hrv_windows.round(2))