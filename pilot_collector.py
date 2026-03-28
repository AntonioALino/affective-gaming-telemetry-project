from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import time
import csv


@dataclass
class CollectionConfig:
    sensor_logger_ip:   str   = os.environ.get("SENSOR_LOGGER_IP")
    sensor_logger_port: int   = 8080
    duration_minutes:   float = 35.0
    output_dir:         str   = "./pilot_data"
    webcam_index:       int   = 0
    webcam_width:       int   = 640
    webcam_height:      int   = 480
    http_timeout_s:     float = 0.3

    MODEL_PATH = "models/face_landmarker.task"

    @property
    def sensor_url(self) -> str:
        return f"http://{self.sensor_logger_ip}:{self.sensor_logger_port}/get"


@dataclass
class SessionFrame:
    timestamp:   float
    elapsed_s:   float
    hr:          Optional[float]
    hrv_ms:      Optional[float]
    pupil_left: Optional[float]
    pupil_right: Optional[float]
    ambient_light: Optional[float]



@dataclass
class NarrativeEvent:
    timestamp: float
    elapsed_s: float
    event_n:   int
    label:     str = ""


@dataclass
class SessionResult:
    participant_id: str
    frames:  List[SessionFrame]  = field(default_factory=list)
    events:  List[NarrativeEvent] = field(default_factory=list)


class PilotCollector:
    def __init__(self, config: Optional[CollectionConfig] = None) -> None:
        self._cfg = config or CollectionConfig()
        self._out_dir = Path(self._cfg.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)


    def collect(self, participant_id: str) -> SessionResult:
        import cv2
        import mediapipe as mp

        result  = SessionResult(participant_id=participant_id)
        p_dir   = self._participant_dir(participant_id)
        ev_cnt  = 0
        start   = time.time()

        face_mesh = self._build_face_mesh(mp)
        cap       = self._open_webcam(cv2)

        self._print_header(participant_id)

        try:
            while True:
                elapsed = time.time() - start
                if elapsed > self._cfg.duration_minutes * 60:
                    print(f"\n  Tempo de {self._cfg.duration_minutes} min atingido.")
                    break

                ret, frame = cap.read()
                if not ret:
                    continue

                ts          = time.time()

                p_left, p_right, light = self._extract_pupil(frame, face_mesh, cv2)
                hr, hrv = self._fetch_apple_watch(self)

                sf = SessionFrame(
                    timestamp=ts,
                    elapsed_s=elapsed,
                    pupil_left=p_left,
                    pupil_right=p_right,
                    ambient_light=light,
                    hr=hr,
                    hrv_ms=hrv
                )
                result.frames.append(sf)

                self._draw_hud(frame, cv2, elapsed, hr, p_left, p_right, light, ev_cnt)
                hr, hrv     = self._fetch_apple_watch(self)

                result.frames.append(sf)

                self._draw_hud(frame, cv2, elapsed, hr, p_left, p_right, light, ev_cnt)
                cv2.imshow("Grief Study — Coleta", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n  Encerrado pelo usuário.")
                    break
                if key == ord("e"):
                    ev_cnt += 1
                    ev = NarrativeEvent(ts, elapsed, ev_cnt)
                    result.events.append(ev)
                    print(f"  [EVENTO {ev_cnt}] t={elapsed:.1f}s")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._save(result, p_dir)

        return result

    def save_csv(self, result: SessionResult) -> None:
        self._save(result, self._participant_dir(result.participant_id))

    def _participant_dir(self, pid: str) -> Path:
        d = self._out_dir / f"participant_{pid}"
        d.mkdir(exist_ok=True)
        return d

    @staticmethod
    def _build_face_mesh(mp):
        return mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.7,
        )

    def _open_webcam(self, cv2):
        cap = cv2.VideoCapture(self._cfg.webcam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.webcam_height)
        return cap

    @staticmethod
    def _extract_pupil(frame, face_mesh, cv2):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ambient_light = float(gray.mean())

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, ambient_light

        try:
            lm = results.multi_face_landmarks[0].landmark

            li_left = [lm[i] for i in (468, 469, 470, 471, 472)]
            left_w = abs(li_left[1].x - li_left[3].x)
            left_h = abs(li_left[2].y - li_left[4].y)
            ratio_left = (left_w + left_h) / 2

            li_right = [lm[i] for i in (473, 474, 475, 476, 477)]
            right_w = abs(li_right[1].x - li_right[3].x)
            right_h = abs(li_right[2].y - li_right[4].y)
            ratio_right = (right_w + right_h) / 2

            h, w = frame.shape[:2]
            cx_l, cy_l = int(li_left[0].x * w), int(li_left[0].y * h)
            cx_r, cy_r = int(li_right[0].x * w), int(li_right[0].y * h)
            cv2.circle(frame, (cx_l, cy_l), int(left_w * w / 2), (255, 0, 0), 1)
            cv2.circle(frame, (cx_r, cy_r), int(right_w * w / 2), (0, 0, 255), 1)

            return float(ratio_left), float(ratio_right), ambient_light

        except Exception:
            return None, None, ambient_light

    @staticmethod
    def _fetch_apple_watch(self) -> Tuple[Optional[float], Optional[float]]:
        import os

        csv_path = "heartrate_log.csv"

        if not os.path.exists(csv_path):
            return None, None

        try:
            with open(csv_path, 'rb') as f:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
                last_line = f.readline().decode()

            colunas = last_line.strip().split(',')

            hr = float(colunas[2])
            hrv = None

            return hr, hrv

        except Exception as e:
            return None, None

    @staticmethod
    def _draw_hud(frame, cv2, elapsed, hr, p_left, p_right, light, ev_cnt) -> None:
        lines = [
            f"t={elapsed:.0f}s",
            f"HR={hr or '---'}",
            f"Pupil L={p_left:.4f}" if p_left else "Pupil L=---",
            f"Pupil R={p_right:.4f}" if p_right else "Pupil R=---",
            f"Light={light:.1f}/255",
            f"Events:{ev_cnt}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    @staticmethod
    def _print_header(participant_id: str) -> None:
        print(f"\n{'='*50}")
        print(f"  COLETA — Participante {participant_id}")
        print(f"  Controles: [e] evento  |  [q] encerrar")
        print(f"{'='*50}")

    def _save(self, result: SessionResult, out_dir: Path) -> None:
        import csv

        frames_path = out_dir / "physiological_data.csv"
        with open(frames_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "elapsed_s",
                               "pupil_left", "pupil_right", "ambient_light",
                               "hr", "hrv_ms"]
            )
            writer.writeheader()
            for fr in result.frames:
                writer.writerow(vars(fr))

        events_path = out_dir / "events.csv"
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "elapsed_s", "event_n", "label"]
            )
            writer.writeheader()
            for ev in result.events:
                writer.writerow(vars(ev))

        print(f"\n  ✓ Dados salvos em {out_dir}")
        print(f"    Frames : {len(result.frames)}")
        print(f"    Eventos: {len(result.events)}")


if __name__ == "__main__":
    cfg = CollectionConfig(
        sensor_logger_ip=os.environ.get("SENSOR_LOGGER_IP"),
        duration_minutes=35,
    )
    collector = PilotCollector(cfg)

    result = collector.collect("P01")
    print(f"\nColeta finalizada: {len(result.frames)} frames, "
          f"{len(result.events)} eventos")