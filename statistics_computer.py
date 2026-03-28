from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    test_name:   str
    group_a:     str
    group_b:     str
    statistic:   float
    p_value:     float
    effect_size: float
    n_a:         int
    n_b:         int
    mean_a:      float
    mean_b:      float
    sd_a:        float
    sd_b:        float
    significant: bool

    def apa_string(self) -> str:
        p_str = "< .001" if self.p_value < .001 else f"= {self.p_value:.3f}"
        if "Mann" in self.test_name:
            return (f"Mann-Whitney U = {self.statistic:.2f}, "
                    f"p {p_str}, r = {self.effect_size:.2f}")
        elif "Wilcoxon" in self.test_name:
            return (f"Wilcoxon W = {self.statistic:.2f}, "
                    f"p {p_str}, r = {self.effect_size:.2f}")
        return f"{self.test_name}: stat = {self.statistic:.2f}, p {p_str}"

    def paper_paragraph(self) -> str:
        sig = "significantly" if self.significant else "not significantly"
        return (
            f"{self.group_a} (M = {self.mean_a:.2f}, SD = {self.sd_a:.2f}, "
            f"n = {self.n_a}) differed {sig} from {self.group_b} "
            f"(M = {self.mean_b:.2f}, SD = {self.sd_b:.2f}, n = {self.n_b}); "
            f"{self.apa_string()}."
        )


@dataclass
class CorrelationResult:
    var_a:       str
    var_b:       str
    r:           float
    p_value:     float
    n:           int
    method:      str   = "Spearman"

    def apa_string(self) -> str:
        p_str = "< .001" if self.p_value < .001 else f"= {self.p_value:.3f}"
        return (f"{self.method} r({self.n - 2}) = {self.r:.3f}, p {p_str}")


class StatisticsComputer:
    def compare_conditions(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        group_a: str,
        group_b: str,
    ) -> Optional[TestResult]:

        if df.empty or group_col not in df.columns or value_col not in df.columns:
            return None

        a = df.loc[df[group_col] == group_a, value_col].dropna().values
        b = df.loc[df[group_col] == group_b, value_col].dropna().values

        if len(a) < 3 or len(b) < 3:
            return None

        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r    = u / (len(a) * len(b))

        return TestResult(
            test_name   = "Mann-Whitney U",
            group_a     = group_a,
            group_b     = group_b,
            statistic   = float(u),
            p_value     = float(p),
            effect_size = float(r),
            n_a         = len(a),
            n_b         = len(b),
            mean_a      = float(np.mean(a)),
            mean_b      = float(np.mean(b)),
            sd_a        = float(np.std(a, ddof=1)),
            sd_b        = float(np.std(b, ddof=1)),
            significant = p < .05,
        )

    def compare_windows(
        self,
        df: pd.DataFrame,
        value_col: str,
        window_col: str = "window",
        pre_label:  str = "Pre-event",
        post_label: str = "Post-event",
    ) -> Optional[TestResult]:

        if df.empty or window_col not in df.columns or value_col not in df.columns:
            print(f"  [AVISO] Dados insuficientes para comparar '{value_col}'.")
            return None

        pre  = df.loc[df[window_col] == pre_label,  value_col].dropna().values
        post = df.loc[df[window_col] == post_label, value_col].dropna().values

        n = min(len(pre), len(post))
        if n < 5:
            print(f"  [AVISO] Amostras insuficientes para {value_col}: pre={len(pre)}, post={len(post)}")
            return None

        pre, post = pre[:n], post[:n]
        w, p      = stats.wilcoxon(pre, post)
        r         = self._wilcoxon_effect_size(pre, post)

        return TestResult(
            test_name   = "Wilcoxon signed-rank",
            group_a     = pre_label,
            group_b     = post_label,
            statistic   = float(w),
            p_value     = float(p),
            effect_size = float(r),
            n_a         = n,
            n_b         = n,
            mean_a      = float(np.mean(pre)),
            mean_b      = float(np.mean(post)),
            sd_a        = float(np.std(pre, ddof=1)),
            sd_b        = float(np.std(post, ddof=1)),
            significant = p < .05,
        )

    def correlate(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        method: str = "spearman",
    ) -> Optional[CorrelationResult]:

        if df.empty or col_a not in df.columns or col_b not in df.columns:
            return None

        clean = df[[col_a, col_b]].dropna()
        if len(clean) < 5:
            return None

        a, b = clean[col_a].values, clean[col_b].values

        if method == "spearman":
            r, p = stats.spearmanr(a, b)
        else:
            r, p = stats.pearsonr(a, b)

        return CorrelationResult(
            var_a   = col_a,
            var_b   = col_b,
            r       = float(r),
            p_value = float(p),
            n       = len(clean),
            method  = method.capitalize(),
        )

    def print_all(
        self,
        mahnob_df: pd.DataFrame,
        pilot_df: pd.DataFrame,
    ) -> None:
        sep = "=" * 60
        print(f"\n{sep}\nESTATÍSTICAS — copiar para a seção Results\n{sep}")

        # Study 1: Negativo vs Neutro
        print("\nStudy 1 — MAHNOB-HCI (RMSSD: Negative vs Neutral):")
        r1 = self.compare_conditions(mahnob_df, "RMSSD", "condition",
                                     "Negative", "Neutral")
        if r1:
            print(f"  {r1.paper_paragraph()}")
            print(f"  APA: {r1.apa_string()}")

        # Study 2: Pre vs Post (Heart Rate)
        print("\nStudy 2 — Apple Watch (HR variability: Pre vs Post evento):")
        r2 = self.compare_windows(pilot_df, "hr_std")
        if r2:
            print(f"  {r2.paper_paragraph()}")
            print(f"  APA: {r2.apa_string()}")

        # Study 3: Pre vs Post (Púpila Esquerda e Direita)
        print("\nStudy 3 — Webcam (Left Pupil: Pre vs Post evento):")
        r_pupil_l = self.compare_windows(pilot_df, "pupil_left_mean")
        if r_pupil_l:
            print(f"  {r_pupil_l.paper_paragraph()}")
            print(f"  APA: {r_pupil_l.apa_string()}")

        print("\nStudy 4 — Webcam (Right Pupil: Pre vs Post evento):")
        r_pupil_r = self.compare_windows(pilot_df, "pupil_right_mean")
        if r_pupil_r:
            print(f"  {r_pupil_r.paper_paragraph()}")
            print(f"  APA: {r_pupil_r.apa_string()}")

        print("\nCorrelação HR variability × Left Pupil dilation:")
        rc_l = self.correlate(pilot_df, "hr_std", "pupil_left_mean")
        if rc_l:
            print(f"  {rc_l.apa_string()}")

        print("\nCorrelação HR variability × Right Pupil dilation:")
        rc_r = self.correlate(pilot_df, "hr_std", "pupil_right_mean")
        if rc_r:
            print(f"  {rc_r.apa_string()}")

        print(f"\n{sep}")

    @staticmethod
    def _wilcoxon_effect_size(a: np.ndarray, b: np.ndarray) -> float:
        n   = len(a)
        w, p = stats.wilcoxon(a, b)
        mu   = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z    = (w - mu) / sigma if sigma > 0 else 0.0
        return abs(z) / np.sqrt(n)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_simulator import DataSimulator
    from pathlib import Path

    sim     = DataSimulator()
    mahnob  = sim.mahnob_sessions()
    pilot   = sim.pilot_sessions()

    sc = StatisticsComputer()
    sc.print_all(mahnob, pilot)