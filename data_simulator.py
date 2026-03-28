from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    n_mahnob_sessions: int = 20
    n_pilot_participants: int = 8
    n_events_per_participant: int = 5
    random_seed: int = 42

    rmssd_negative: float = 32.0
    rmssd_neutral:  float = 42.0
    rmssd_positive: float = 48.0
    rmssd_noise:    float = 6.0

    hr_std_pre:   float = 4.2
    hr_std_post:  float = 6.8
    hr_std_noise: float = 0.8

    pupil_pre:   float = 0.0220
    pupil_post:  float = 0.0285
    pupil_noise: float = 0.003


class DataSimulator:

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self._cfg = config or SimulationConfig()
        self._rng = np.random.default_rng(self._cfg.random_seed)


    def mahnob_sessions(self) -> pd.DataFrame:
        cfg = self._cfg
        records = []
        conditions = ["Negative", "Neutral", "Positive"]
        rmssd_map  = {
            "Negative": cfg.rmssd_negative,
            "Neutral":  cfg.rmssd_neutral,
            "Positive": cfg.rmssd_positive,
        }
        valence_map = {
            "Negative": [2, 2, 3],
            "Neutral":  [4, 5, 6],
            "Positive": [7, 8, 8],
        }

        for i in range(cfg.n_mahnob_sessions):
            cond = conditions[i % 3]
            records.append({
                "session":   f"sim_{i + 1:02d}",
                "RMSSD":     rmssd_map[cond] + self._rng.normal(0, cfg.rmssd_noise),
                "feltVlnc":  self._rng.choice(valence_map[cond]),
                "condition": cond,
                "simulated": True,
            })

        df = pd.DataFrame(records)
        self._warn("MAHNOB-HCI", cfg.n_mahnob_sessions)
        return df

    def pilot_sessions(self) -> pd.DataFrame:
        cfg = self._cfg
        records = []

        for pid in range(1, cfg.n_pilot_participants + 1):
            for ev in range(1, cfg.n_events_per_participant + 1):
                for window, hr_base, pupil_base in [
                    ("Pre-event",  cfg.hr_std_pre,  cfg.pupil_pre),
                    ("Post-event", cfg.hr_std_post, cfg.pupil_post),
                ]:
                    records.append({
                        "participant": f"P{pid:02d}",
                        "event_n":     ev,
                        "window":      window,
                        "hr_std":      hr_base  + self._rng.normal(0, cfg.hr_std_noise),
                        "pupil_mean":  pupil_base + self._rng.normal(0, cfg.pupil_noise),
                        "hr_mean":     75.0 + self._rng.normal(0, 8),
                        "simulated":   True,
                    })

        df = pd.DataFrame(records)
        self._warn("Apple Watch + MediaPipe", cfg.n_pilot_participants)
        return df

    def raw_hr_timeseries(
        self,
        duration_seconds: int = 1800,
        sampling_hz: float = 1.0,
        n_events: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        timestamps = np.arange(0, duration_seconds, 1.0 / sampling_hz)
        hr_base    = 72.0 + self._rng.normal(0, 1, size=len(timestamps))

        event_times = sorted(
            self._rng.choice(
                np.arange(60, duration_seconds - 60),
                size=n_events, replace=False
            )
        )
        for et in event_times:
            mask = (timestamps >= et) & (timestamps < et + 60)
            hr_base[mask] -= self._rng.uniform(3, 8)

        pupil = 0.022 + self._rng.normal(0, 0.002, size=len(timestamps))
        for et in event_times:
            mask = (timestamps >= et) & (timestamps < et + 30)
            pupil[mask] += self._rng.uniform(0.003, 0.007)

        ts_df = pd.DataFrame({
            "elapsed_s":   timestamps,
            "hr":          np.clip(hr_base, 45, 130),
            "pupil_ratio": np.clip(pupil, 0.010, 0.045),
        })

        ev_df = pd.DataFrame({
            "elapsed_s": event_times,
            "event_n":   range(1, n_events + 1),
        })

        return ts_df, ev_df


    @staticmethod
    def _warn(source: str, n: int) -> None:
        print(
            f"  [SIMULADO] {n} sessões de {source} geradas para demonstração.\n"
            "  → Substitua pelos dados reais quando disponíveis."
        )


if __name__ == "__main__":
    sim = DataSimulator()

    mahnob = sim.mahnob_sessions()
    print("MAHNOB simulado:")
    print(mahnob.groupby("condition")["RMSSD"].describe().round(2))

    pilot = sim.pilot_sessions()
    print("\nPiloto simulado:")
    print(pilot.groupby("window")[["hr_std", "pupil_mean"]].mean().round(4))
