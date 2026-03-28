from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "core"))

from data_simulator      import DataSimulator, SimulationConfig
from mahnob_loader       import MAHNOBLoader
from hrv_analyzer        import HRVAnalyzer
from pilot_collector     import PilotCollector, CollectionConfig
from pilot_analyzer      import PilotAnalyzer
from statistics_computer import StatisticsComputer
from figure_generator    import FigureGenerator, FigureConfig



MAHNOB_DIR      = "./mahnob_hci"
PILOT_DIR       = "./pilot_data"
FIGURES_DIR     = "./figures"
REPORT_DIR      = "./report_templates"
N_MAHNOB        = 20
PARTICIPANTS    = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]

SENSOR_LOGGER_IP = os.environ.get("SENSOR_LOGGER_IP")



def step1_study1_mahnob():
    import pandas as pd

    mahnob_path = Path(MAHNOB_DIR)

    if mahnob_path.exists() and any(mahnob_path.iterdir()):
        loader   = MAHNOBLoader(MAHNOB_DIR)
        sessions = loader.load_n_sessions(N_MAHNOB)

        analyzer = HRVAnalyzer()
        df       = analyzer.from_sessions(sessions)
    else:
        print("  Dataset MAHNOB não encontrado — usando dados simulados.")
        print("  → Baixe em https://mahnob-db.eu/hci-tagging/")
        sim = DataSimulator(SimulationConfig(n_mahnob_sessions=N_MAHNOB))
        df  = sim.mahnob_sessions()

    df.to_csv(Path(PILOT_DIR) / "mahnob_results.csv", index=False)
    print(f"  Sessions analisadas: {len(df)}")
    return df


def step2_collect(participant_id: str) -> None:
    print(f"\n─── STEP 2: Coleta — {participant_id} ───────────────────")
    cfg       = CollectionConfig(
        sensor_logger_ip = SENSOR_LOGGER_IP,
        output_dir       = PILOT_DIR,
        duration_minutes = 35,
    )
    collector = PilotCollector(cfg)
    collector.collect(participant_id)


def step3_study2_pilot():
    print("\n─── STEP 3: Study 2 — Pilot Analysis ───────────────────")
    analyzer = PilotAnalyzer(PILOT_DIR)

    try:
        df = analyzer.analyze_all()
    except FileNotFoundError:
        df = pd.DataFrame()

    if df.empty:
        print("  [AVISO] Dados reais vazios ou sem eventos. Usando simulador...")
        sim = DataSimulator()
        df = sim.pilot_sessions()

    df.to_csv(Path(PILOT_DIR) / "pilot_summary.csv", index=False)
    print("\n  Resumo:")
    return df


def step4_statistics(mahnob_df, pilot_df) -> None:
    print("\n─── STEP 4: Statistics ──────────────────────────────────")
    sc = StatisticsComputer()
    sc.print_all(mahnob_df, pilot_df)


def step5_figures(mahnob_df, pilot_df) -> None:
    print("\n─── STEP 5: Figures ─────────────────────────────────────")
    Path(FIGURES_DIR).mkdir(exist_ok=True)
    gen = FigureGenerator(FigureConfig(output_dir=FIGURES_DIR))
    gen.figure1_hrv(mahnob_df, pilot_df)
    gen.figure2_pupil(mahnob_df, pilot_df)
    gen.figure3_convergence(mahnob_df, pilot_df)



def run_full_pipeline() -> None:
    mahnob_df = step1_study1_mahnob()
    pilot_df  = step3_study2_pilot()
    step4_statistics(mahnob_df, pilot_df)
    step5_figures(mahnob_df, pilot_df)
    print("\n✓ Pipeline completo. Verifique ./figures/ e ./report_templates/")


def run_collection(participant_id: str) -> None:
    step2_collect(participant_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grief Wearables Study — SBGames 2026"
    )
    parser.add_argument(
        "mode",
        choices=["analyze", "collect"],
        help="'analyze' roda o pipeline completo; 'collect' inicia uma sessão"
    )
    parser.add_argument(
        "--participant", "-p",
        default="P01",
        help="ID do participante para o modo collect (ex.: P01)"
    )
    args = parser.parse_args()

    if args.mode == "analyze":
        run_full_pipeline()
    elif args.mode == "collect":
        run_collection(args.participant)