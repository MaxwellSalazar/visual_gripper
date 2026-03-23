"""
calibration/force_vision_curve.py — Curva de Calibración Fuerza-Visión (F-δ)

PROPÓSITO (crítico para publicación Q1/Q2):
    Transforma la medida cualitativa (píxeles) en cuantitativa (Newtons),
    dando rigor científico al sistema mediante una curva de calibración.

PROCEDIMIENTO:
    1. Coloca el gripper contra una báscula de precisión o celda de carga
    2. Aplica fuerzas conocidas manualmente (empuja el gripper)
    3. El script captura la deflexión visual en ese instante
    4. Registra el par (δ_px, F_N) y ajusta una regresión lineal
    5. Guarda los coeficientes K y b en data/calibration.json

USO:
    python calibration/force_vision_curve.py

CONTROLES DURANTE LA CAPTURA:
    ESPACIO → Capturar punto de calibración actual
    'd'     → Borrar último punto
    'f'     → Ajustar y guardar la regresión
    'q'     → Salir
"""

import cv2
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from core.perception import MarkerPerception, CameraStream
from core.state_estimator import StateEstimator

# ── Opcional: matplotlib para visualizar la curva ────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # Sin GUI — genera imagen PNG
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Calibración F-δ] matplotlib no disponible — se omitirá la gráfica.")


def run_calibration():
    print("\n" + "="*60)
    print("CALIBRACIÓN FUERZA-VISIÓN")
    print("Coloca el gripper contra una báscula o celda de carga.")
    print("Para cada fuerza conocida, presiona ESPACIO para capturar.")
    print("="*60 + "\n")

    camera  = CameraStream()
    percept = MarkerPerception()
    estimator = StateEstimator()

    # Datos de calibración: lista de (delta_px, force_N)
    data_points: list[tuple[float, float]] = []

    # ── Auto-calibrar posición de reposo primero ──────────────────────────────
    print("[Paso 1] Mantén el gripper ABIERTO (sin contacto) por 3 segundos...")
    rest_centroids = []
    t0 = time.time()
    while time.time() - t0 < 3.0:
        ok, frame = camera.read()
        if not ok:
            continue
        centroid, debug_img, _ = percept.process(frame)
        if centroid:
            rest_centroids.append(centroid)
        elapsed = time.time() - t0
        cv2.putText(debug_img,
                    f"Capturando posición de reposo: {elapsed:.1f}/3.0s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Calibración F-δ", debug_img)
        cv2.waitKey(1)

    if len(rest_centroids) < 10:
        print("ERROR: No se detectó el marcador en reposo. "
              "Verifica la calibración HSV primero.")
        camera.release()
        cv2.destroyAllWindows()
        return

    estimator.auto_calibrate_rest(rest_centroids)

    print("\n[Paso 2] Aplica fuerzas conocidas al gripper.")
    print("  ESPACIO → capturar punto | 'd' → borrar último | 'f' → finalizar\n")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        centroid, debug_img, _ = percept.process(frame)
        state = estimator.update(centroid)
        delta = state["delta_px"]

        # HUD
        cv2.putText(debug_img,
                    f"delta = {delta:.1f} px  |  Puntos: {len(data_points)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(debug_img,
                    "ESPACIO=capturar  'd'=borrar  'f'=ajustar  'q'=salir",
                    (10, config.CAMERA_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Mostrar puntos capturados
        for i, (d, f) in enumerate(data_points[-5:]):
            cv2.putText(debug_img,
                        f"  Pt{i+1}: {d:.1f}px → {f:.3f}N",
                        (10, 60 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        cv2.imshow("Calibración F-δ", debug_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Solicitar la fuerza al usuario por terminal
            cv2.waitKey(100)
            try:
                force_input = input(f"  → Ingresa la fuerza aplicada ahora "
                                    f"(delta={delta:.1f}px) [N]: ")
                force_n = float(force_input)
                data_points.append((delta, force_n))
                print(f"    Punto {len(data_points)}: "
                      f"δ={delta:.1f}px, F={force_n:.3f}N ✓")
            except ValueError:
                print("    Valor inválido, intenta de nuevo.")

        elif key == ord('d') and data_points:
            removed = data_points.pop()
            print(f"  Punto eliminado: {removed}")

        elif key == ord('f'):
            if len(data_points) < 3:
                print("  Se necesitan al menos 3 puntos para la regresión.")
            else:
                _fit_and_save(data_points)
                break

        elif key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


def _fit_and_save(data_points: list):
    """Ajusta regresión lineal y guarda coeficientes."""
    deltas = np.array([d[0] for d in data_points])
    forces = np.array([d[1] for d in data_points])

    # Regresión lineal: F = K * δ + b
    coeffs  = np.polyfit(deltas, forces, 1)
    K, b    = float(coeffs[0]), float(coeffs[1])

    # R² para reportar en publicación
    f_pred  = K * deltas + b
    ss_res  = np.sum((forces - f_pred) ** 2)
    ss_tot  = np.sum((forces - forces.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print("\n" + "="*55)
    print("REGRESIÓN LINEAL F = K·δ + b")
    print(f"  K  = {K:.6f}  N/px")
    print(f"  b  = {b:.6f}  N")
    print(f"  R² = {r2:.4f}  (ideal: >0.99)")
    print("="*55)

    # Guardar JSON
    os.makedirs("data", exist_ok=True)
    calib = {
        "K": K, "b": b, "r2": r2,
        "n_points": len(data_points),
        "data_points": [{"delta_px": d, "force_n": f}
                        for d, f in data_points]
    }
    with open("data/calibration.json", "w") as fp:
        json.dump(calib, fp, indent=2)
    print("Guardado en data/calibration.json")
    print(f"\nCopia en config.py:\n"
          f"  CALIB_K = {K:.6f}\n"
          f"  CALIB_B = {b:.6f}")

    # Gráfica PNG para publicación
    if HAS_MPL:
        _plot_calibration_curve(deltas, forces, K, b, r2)


def _plot_calibration_curve(deltas, forces, K, b, r2):
    """Genera figura lista para publicación (300 dpi)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(deltas, forces, color="#D62728", s=60,
               zorder=5, label="Datos medidos")

    x_line = np.linspace(0, max(deltas) * 1.1, 200)
    ax.plot(x_line, K * x_line + b, "k--", linewidth=1.5,
            label=f"F = {K:.4f}·δ + {b:.4f}\n$R^2$ = {r2:.4f}")

    ax.set_xlabel("Deflexión visual δ [px]", fontsize=11)
    ax.set_ylabel("Fuerza F [N]",            fontsize=11)
    ax.set_title("Curva de Calibración Fuerza-Visión", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = "data/calibration_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfica guardada en {path} (300 dpi, lista para publicación)")


if __name__ == "__main__":
    run_calibration()
