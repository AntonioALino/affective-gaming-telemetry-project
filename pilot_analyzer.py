from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class WindowMetrics:
    participant: str
    event_n: int
    window: str
    hr_mean: float
    hr_std: float
    pupil_left_mean: float
    pupil_left_std: float
    pupil_right_mean: float
    pupil_right_std: float
    ambient_light_mean: float
    n_frames: int


class PilotAnalyzer:
    MIN_LIGHT_LUX = 40
    MAX_LIGHT_LUX = 220

    PRE_WINDOW_S = 30
    POST_WINDOW_S = 60

    def __init__(self, pilot_dir: str | Path = "./pilot_data") -> None:
        self._root = Path(pilot_dir)

    def available_participants(self) -> List[Path]:
        return sorted(self._root.glob("participant_*"))

    def analyze_all(self) -> pd.DataFrame:
        participants = self.available_participants()
        if not participants:
            raise FileNotFoundError(
                f"Nenhum dado piloto encontrado em {self._root}.\n"
                "Execute PilotCollector.collect() primeiro."
            )

        all_metrics: List[WindowMetrics] = []
        for p_dir in participants:
            pid = p_dir.name.replace("participant_", "")
            metrics = self.analyze_participant(pid, p_dir)
            all_metrics.extend(metrics)
            print(f"  {p_dir.name}: {len(metrics)} janelas analisadas")

        return self._to_dataframe(all_metrics)

    def analyze_participant(
            self,
            participant_id: str,
            p_dir: Optional[Path] = None,
    ) -> List[WindowMetrics]:
        if p_dir is None:
            p_dir = self._root / f"participant_{participant_id}"

        phys_path = p_dir / "physiological_data.csv"
        events_path = p_dir / "events.csv"

        if not phys_path.exists() or not events_path.exists():
            print(f"  [AVISO] Dados incompletos para {participant_id}")
            return []

        phys = pd.read_csv(phys_path)
        events = pd.read_csv(events_path)

        results = []
        for _, ev_row in events.iterrows():
            ev_t = float(ev_row["elapsed_s"])

            pre = self._window(phys, ev_t - self.PRE_WINDOW_S, ev_t)
            post = self._window(phys, ev_t, ev_t + self.POST_WINDOW_S)

            for label, window_df in [("Pre-event", pre), ("Post-event", post)]:
                m = self._compute_metrics(participant_id, int(ev_row["event_n"]),
                                          label, window_df)
                if m:
                    results.append(m)

        return results

    def summary(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        if data_frame.empty or "window" not in data_frame.columns:
            print("  [AVISO] DataFrame vazio ou sem a coluna 'window'. Nenhuma métrica para resumir.")
            return pd.DataFrame()

        cols_to_summarize = ["hr_std", "pupil_left_mean", "pupil_right_mean", "ambient_light_mean"]
        return (
            data_frame.groupby("window")[cols_to_summarize]
            .agg(["mean", "std"])
            .round(4)
        )

    def _window(self, data_frame: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
        time_filtered = data_frame[(data_frame["elapsed_s"] >= t_start) & (data_frame["elapsed_s"] < t_end)]

        if time_filtered.empty:
            return time_filtered

        light_filtered = time_filtered[(time_filtered["ambient_light"] >= self.MIN_LIGHT_LUX) &
                                       (time_filtered["ambient_light"] <= self.MAX_LIGHT_LUX)]

        if len(light_filtered) < 0.5 * len(time_filtered):
            print(f"  [AVISO LUZ] Janela ({t_start:.1f}-{t_end:.1f}s) rejeitada por iluminação fora do padrão.")
            return pd.DataFrame()

        return light_filtered

    @staticmethod
    def _compute_metrics(
            participant: str,
            event_n: int,
            window: str,
            df: pd.DataFrame,
    ) -> Optional[WindowMetrics]:
        if len(df) < 5:
            return None

        hr_vals = df["hr"].dropna()
        p_left_vals = df["pupil_left"].dropna()
        p_right_vals = df["pupil_right"].dropna()
        light_vals = df["ambient_light"].dropna()

        return WindowMetrics(
            participant=participant,
            event_n=event_n,
            window=window,
            hr_mean=float(hr_vals.mean()) if len(hr_vals) > 0 else np.nan,
            hr_std=float(hr_vals.std()) if len(hr_vals) > 1 else np.nan,
            pupil_left_mean=float(p_left_vals.mean()) if len(p_left_vals) > 0 else np.nan,
            pupil_left_std=float(p_left_vals.std()) if len(p_left_vals) > 1 else np.nan,
            pupil_right_mean=float(p_right_vals.mean()) if len(p_right_vals) > 0 else np.nan,
            pupil_right_std=float(p_right_vals.std()) if len(p_right_vals) > 1 else np.nan,
            ambient_light_mean=float(light_vals.mean()) if len(light_vals) > 0 else np.nan,
            n_frames=len(df),
        )

    @staticmethod
    def _to_dataframe(metrics: List[WindowMetrics]) -> pd.DataFrame:
        return pd.DataFrame([vars(m) for m in metrics])


if __name__ == "__main__":
    analyzer = PilotAnalyzer("./pilot_data")
    df = analyzer.analyze_all()
    print("\nResumo por janela:")
    print(analyzer.summary(df))