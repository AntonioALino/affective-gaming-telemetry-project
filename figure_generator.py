from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


@dataclass
class FigureConfig:
    output_dir:    str   = "./figures"
    dpi:           int   = 300
    fig_width:     float = 12.0
    fig_height:    float = 5.0
    font_size:     int   = 11
    style:         str   = "whitegrid"
    palette:       str   = "muted"
    save_pdf:      bool  = True
    save_png:      bool  = True

    color_negative: str = "#D62728"
    color_neutral:  str = "#7F7F7F"
    color_positive: str = "#2CA02C"
    color_pre:      str = "#AEC7E8"
    color_post:     str = "#D62728"


class FigureGenerator:
    def __init__(self, config: Optional[FigureConfig] = None) -> None:
        self._cfg     = config or FigureConfig()
        self._out_dir = Path(self._cfg.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._apply_theme()


    def figure1_hrv(
        self,
        mahnob_df: pd.DataFrame,
        pilot_df:  pd.DataFrame,
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(self._cfg.fig_width,
                                                 self._cfg.fig_height))
        fig.suptitle(
            "Physiological Response to Grief-Adjacent Affect\n"
            "Heart Rate Variability: Clinical Dataset vs. Consumer Wearable",
            fontsize=self._cfg.font_size + 2,
            fontweight="bold",
            y=1.02,
        )

        self._plot_mahnob_hrv(axes[0], mahnob_df)
        self._plot_pilot_hrv(axes[1], pilot_df)

        plt.tight_layout()
        return self._save(fig, "figure1_hrv_comparison")

    def figure2_pupil(
        self,
        mahnob_df: pd.DataFrame,
        pilot_df:  pd.DataFrame,
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(self._cfg.fig_width,
                                                 self._cfg.fig_height))
        fig.suptitle(
            "Pupillary Response to Grief-Adjacent Affect\n"
            "MAHNOB-HCI vs. Webcam-Based Eye Tracking (MediaPipe)",
            fontsize=self._cfg.font_size + 2,
            fontweight="bold",
            y=1.02,
        )

        self._plot_mahnob_pupil(axes[0], mahnob_df)
        self._plot_pilot_pupil(axes[1], pilot_df)

        plt.tight_layout()
        return self._save(fig, "figure2_pupil_comparison")

    def figure3_convergence(
        self,
        mahnob_df: pd.DataFrame,
        pilot_df:  pd.DataFrame,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4))

        cond_order = ["Negative", "Neutral", "Positive"]
        if "condition" in mahnob_df.columns:
            m_means = (
                mahnob_df.groupby("condition")["RMSSD"]
                .mean()
                .reindex(cond_order)
                .dropna()
            )
            z_mahnob = (m_means - m_means.mean()) / m_means.std()
            ax.plot(range(len(z_mahnob)), z_mahnob.values,
                    "o--", color="#1F77B4", label="MAHNOB-HCI (ECG)", lw=2)
            ax.set_xticks(range(len(cond_order)))
            ax.set_xticklabels(cond_order)

        if "window" in pilot_df.columns and "hr_std" in pilot_df.columns:
            win_order = ["Pre-event", "Post-event"]
            p_means   = (
                pilot_df.groupby("window")["hr_std"]
                .mean()
                .reindex(win_order)
                .dropna()
            )
            z_pilot = (p_means - p_means.mean()) / p_means.std()
            ax.plot([0, 2], z_pilot.values,
                    "s-", color="#D62728", label="Apple Watch SE 3 (HR)", lw=2)

        ax.axhline(0, color="gray", linestyle=":", lw=1)
        ax.set_ylabel("Z-score (normalized HRV)", fontsize=10)
        ax.set_title("Convergence of HRV Pattern Across Datasets\n"
                     "(Z-score normalized to enable cross-scale comparison)",
                     fontsize=11)
        ax.legend(fontsize=10)

        plt.tight_layout()
        return self._save(fig, "figure3_convergence")


    def _plot_mahnob_hrv(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        cfg      = self._cfg
        order    = ["Negative", "Neutral", "Positive"]
        palette  = {
            "Negative": cfg.color_negative,
            "Neutral":  cfg.color_neutral,
            "Positive": cfg.color_positive,
        }
        plot_df  = df[df["condition"].isin(order)].copy() if "condition" in df.columns else df

        sns.boxplot(data=plot_df, x="condition", y="RMSSD",
                    order=order, palette=palette, ax=ax,
                    width=0.5, linewidth=1.5)
        sns.stripplot(data=plot_df, x="condition", y="RMSSD",
                      order=order, color="black", size=4,
                      alpha=0.5, jitter=True, ax=ax)

        n = len(plot_df)
        ax.set_title(f"Study 1: MAHNOB-HCI\n(Clinical ECG, N = {n})",
                     fontsize=cfg.font_size)
        ax.set_xlabel("Emotional Condition (Self-Report)", fontsize=cfg.font_size - 1)
        ax.set_ylabel("RMSSD (ms)", fontsize=cfg.font_size - 1)
        self._add_significance_bracket(ax, plot_df, "RMSSD")

    def _plot_pilot_hrv(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        cfg     = self._cfg
        order   = ["Pre-event", "Post-event"]
        palette = {"Pre-event": cfg.color_pre, "Post-event": cfg.color_post}

        sns.boxplot(data=df, x="window", y="hr_std",
                    order=order, palette=palette, ax=ax,
                    width=0.5, linewidth=1.5)
        sns.stripplot(data=df, x="window", y="hr_std",
                      order=order, color="black", size=4,
                      alpha=0.5, jitter=True, ax=ax)

        n_parts = df["participant"].nunique() if "participant" in df.columns else "?"
        ax.set_title(f"Study 2: Apple Watch SE 3\n(Consumer HR, N = {n_parts})",
                     fontsize=cfg.font_size)
        ax.set_xlabel("Narrative Event Window", fontsize=cfg.font_size - 1)
        ax.set_ylabel("HR Variability (SD, bpm)", fontsize=cfg.font_size - 1)
        self._add_significance_bracket(ax, df, "hr_std")

    def _plot_mahnob_pupil(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        if "pupil_mean" in df.columns and df["pupil_mean"].notna().any():
            cfg     = self._cfg
            order   = ["Negative", "Neutral", "Positive"]
            palette = {
                "Negative": cfg.color_negative,
                "Neutral":  cfg.color_neutral,
                "Positive": cfg.color_positive,
            }
            sns.boxplot(data=df, x="condition", y="pupil_mean",
                        order=order, palette=palette, ax=ax)
            ax.set_title("Study 1: MAHNOB-HCI Eye Tracker",
                         fontsize=self._cfg.font_size)
            ax.set_ylabel("Pupil Diameter (normalized)")
        else:
            self._placeholder(
                ax,
                "MAHNOB-HCI gaze data\nrequer extração separada\ndo arquivo _gaze.csv",
                "Study 1: MAHNOB-HCI Eye Tracker",
            )

    def _plot_pilot_pupil(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        if "pupil_mean" in df.columns and df["pupil_mean"].notna().any():
            cfg     = self._cfg
            order   = ["Pre-event", "Post-event"]
            palette = {"Pre-event": cfg.color_pre, "Post-event": cfg.color_post}
            sns.boxplot(data=df, x="window", y="pupil_mean",
                        order=order, palette=palette, ax=ax)
            n = df["participant"].nunique() if "participant" in df.columns else "?"
            ax.set_title(f"Study 2: MediaPipe (Webcam, N = {n})",
                         fontsize=self._cfg.font_size)
            ax.set_ylabel("Iris Diameter Ratio (MediaPipe landmarks)")
            ax.set_xlabel("Narrative Event Window")
        else:
            self._placeholder(
                ax,
                "Dados piloto pendentes\ncoletar com PilotCollector",
                "Study 2: MediaPipe (Webcam)",
            )


    def _apply_theme(self) -> None:
        sns.set_theme(style=self._cfg.style, palette=self._cfg.palette)
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size":   self._cfg.font_size,
        })

    def _save(self, fig: plt.Figure, stem: str) -> Path:
        paths = []
        if self._cfg.save_pdf:
            p = self._out_dir / f"{stem}.pdf"
            fig.savefig(p, dpi=self._cfg.dpi, bbox_inches="tight")
            paths.append(p)
        if self._cfg.save_png:
            p = self._out_dir / f"{stem}.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
        pdf_path = self._out_dir / f"{stem}.pdf"
        print(f"  ✓ Figura salva: {pdf_path}")
        return pdf_path

    @staticmethod
    def _add_significance_bracket(
        ax: plt.Axes, df: pd.DataFrame, col: str
    ) -> None:
        try:
            y_max = df[col].max()
            y_pos = y_max * 1.05
            ax.annotate("", xy=(1, y_pos), xytext=(0, y_pos),
                        arrowprops=dict(arrowstyle="-", color="black", lw=1))
            ax.text(0.5, y_pos * 1.02, "*", ha="center", fontsize=14)
        except Exception:
            pass

    @staticmethod
    def _placeholder(ax: plt.Axes, message: str, title: str) -> None:
        ax.text(0.5, 0.5, message, ha="center", va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title(title)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_simulator import DataSimulator

    sim    = DataSimulator()
    m_df   = sim.mahnob_sessions()
    p_df   = sim.pilot_sessions()

    gen = FigureGenerator()
    gen.figure1_hrv(m_df, p_df)
    gen.figure2_pupil(m_df, p_df)
    gen.figure3_convergence(m_df, p_df)
    print("Figuras geradas em ./figures/")