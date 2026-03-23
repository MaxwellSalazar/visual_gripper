"""
tests/test_versatility.py — Prueba de Versatilidad con Objetos de Diferente Rigidez

HIPÓTESIS PARA LA PUBLICACIÓN:
    El sistema de control visual es agnóstico al tipo de objeto —
    la deformación del gripper es el "lenguaje universal" que funciona
    igualmente con objetos rígidos (piedra) y frágiles (tomate, huevo).

METODOLOGÍA:
    Para cada objeto de prueba:
      1. El gripper cierra hasta alcanzar la fuerza objetivo
      2. Se registran δ_máx, F_máx, tiempo_de_cierre, PWM_promedio
      3. Se repite N veces para estadística

OBJETOS SUGERIDOS (en orden de rigidez):
    piedra, pelota_tenis, esponja, tomate, huevo, uva

USO:
    python tests/test_versatility.py

CONTROLES:
    's' → Iniciar/siguiente trial para el objeto actual
    'n' → Siguiente objeto
    'q' → Guardar y salir
"""

import sys
import os
import time
import cv2
import csv
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from core.perception      import MarkerPerception, CameraStream
from core.state_estimator import StateEstimator
from core.pid_controller  import PIDController
from control.motor_driver import MotorDriver
from utils.logger         import ExperimentLogger


# ── Definición del experimento ────────────────────────────────────────────────
OBJECTS = [
    {"name": "piedra",       "stiffness": "rígido",   "color": (80,  80,  200)},
    {"name": "pelota_tenis", "stiffness": "semi-rígido","color": (80, 200, 200)},
    {"name": "esponja",      "stiffness": "blando",    "color": (80, 200, 80)},
    {"name": "tomate",       "stiffness": "frágil",    "color": (40, 140, 255)},
    {"name": "huevo",        "stiffness": "muy frágil","color": (40, 40,  255)},
]
TRIALS_PER_OBJECT = 5
MAX_GRASP_FRAMES  = 300   # Máximo de frames por trial (~5s a 60fps)


def run_versatility_test():
    print("\n" + "="*60)
    print("TEST DE VERSATILIDAD — Visual Soft Gripper")
    print(f"Objetos: {[o['name'] for o in OBJECTS]}")
    print(f"Trials por objeto: {TRIALS_PER_OBJECT}")
    print("="*60)

    camera    = CameraStream()
    percept   = MarkerPerception()
    estimator = StateEstimator()
    motor     = MotorDriver()
    logger    = ExperimentLogger("test_versatility")

    # Auto-calibrar P0
    print("\nCalibrando posición de reposo (3s, gripper abierto)...")
    rest_centroids = []
    t0 = time.time()
    while time.time() - t0 < 3.0:
        ok, frame = camera.read()
        if not ok:
            continue
        centroid, debug, _ = percept.process(frame)
        if centroid:
            rest_centroids.append(centroid)
        cv2.imshow("Versatility Test", debug)
        cv2.waitKey(1)

    if len(rest_centroids) < 5:
        print("ERROR: marcador no detectado.")
        camera.release()
        return
    estimator.auto_calibrate_rest(rest_centroids)

    all_results = []

    # ── Iterar por objetos ────────────────────────────────────────────────────
    for obj_idx, obj in enumerate(OBJECTS):
        obj_name  = obj["name"]
        obj_color = obj["color"]
        obj_results = []

        print(f"\n{'='*40}")
        print(f"OBJETO: {obj_name.upper()} ({obj['stiffness']})")
        print(f"Trials: {TRIALS_PER_OBJECT}")
        print(f"{'='*40}")
        print("Coloca el objeto en el gripper y presiona 's' para iniciar cada trial.")

        trial_count = 0
        skip_object = False

        while trial_count < TRIALS_PER_OBJECT and not skip_object:
            # Esperar confirmación del operador
            print(f"\n  Trial {trial_count+1}/{TRIALS_PER_OBJECT} — "
                  f"'s'=iniciar | 'n'=siguiente objeto | 'q'=salir")

            waiting = True
            while waiting:
                ok, frame = camera.read()
                if not ok:
                    continue
                centroid, debug, _ = percept.process(frame)
                state = estimator.update(centroid)

                cv2.putText(debug,
                            f"Objeto: {obj_name} | Trial {trial_count+1}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, obj_color, 2)
                cv2.putText(debug, "'s'=start 'n'=next_obj 'q'=quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)
                cv2.imshow("Versatility Test", debug)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    waiting = False
                elif key == ord('n'):
                    waiting   = False
                    skip_object = True
                elif key == ord('q'):
                    _save_results(all_results, logger)
                    camera.release()
                    motor.cleanup()
                    cv2.destroyAllWindows()
                    return

            if skip_object:
                break

            # ── Ejecutar trial ────────────────────────────────────────────────
            pid = PIDController()
            logger.start_trial(obj_idx * TRIALS_PER_OBJECT + trial_count + 1,
                                obj_name)
            estimator.reset_peaks()

            trial_data = {
                "object": obj_name,
                "trial" : trial_count + 1,
                "delta_max_px"     : 0.0,
                "force_max_n"      : 0.0,
                "grasp_time_ms"    : 0.0,
                "pwm_values"       : [],
                "reached_setpoint" : False,
            }

            t_start = time.perf_counter()
            motor.close_gripper(config.PWM_INITIAL)

            for frame_n in range(MAX_GRASP_FRAMES):
                ok, frame = camera.read()
                if not ok:
                    break

                centroid, debug, _ = percept.process(frame)
                state   = estimator.update(centroid)
                control = pid.compute(state)
                motor.apply_control(control)
                logger.log(state, control)

                delta = state.get("delta_px", 0.0)
                force = state.get("force_n",  0.0)

                trial_data["delta_max_px"] = max(trial_data["delta_max_px"], delta)
                trial_data["force_max_n"]  = max(trial_data["force_max_n"],  force)
                trial_data["pwm_values"].append(control.get("pwm", 0))

                if force >= config.FORCE_SETPOINT_N:
                    trial_data["reached_setpoint"] = True

                # HUD
                cv2.putText(debug,
                            f"{obj_name} | δ={delta:.1f}px F={force:.3f}N",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, obj_color, 2)
                cv2.imshow("Versatility Test", debug)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Detener si acción es STOP
                if control.get("action") == "stop":
                    break

            t_end = time.perf_counter()
            motor.stop()
            time.sleep(0.5)   # Pausa antes de abrir

            trial_data["grasp_time_ms"] = (t_end - t_start) * 1000
            trial_data["pwm_mean"]      = (sum(trial_data["pwm_values"]) /
                                           max(len(trial_data["pwm_values"]), 1))
            obj_results.append(trial_data)

            print(f"    → δ_max={trial_data['delta_max_px']:.1f}px  "
                  f"F_max={trial_data['force_max_n']:.3f}N  "
                  f"t={trial_data['grasp_time_ms']:.0f}ms  "
                  f"setpoint={'✓' if trial_data['reached_setpoint'] else '✗'}")

            motor.open_gripper()
            time.sleep(1.0)
            motor.stop()
            trial_count += 1

        all_results.extend(obj_results)

    _save_results(all_results, logger)
    camera.release()
    motor.cleanup()
    cv2.destroyAllWindows()


def _save_results(results: list, logger: ExperimentLogger):
    """Guarda resumen estadístico por objeto."""
    logger.close()
    if not results:
        return

    path = "data/versatility_summary.csv"
    os.makedirs("data", exist_ok=True)

    # Agrupar por objeto
    by_object = {}
    for r in results:
        n = r["object"]
        by_object.setdefault(n, []).append(r)

    print("\n" + "="*65)
    print("RESUMEN — VERSATILIDAD")
    print(f"{'Objeto':<15} {'δ_max_media':>12} {'F_max_media':>12} "
          f"{'t_media_ms':>12} {'Setpoint%':>10}")
    print("-"*65)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "delta_max_mean_px", "delta_max_std",
                    "force_max_mean_n", "force_max_std",
                    "grasp_time_mean_ms", "setpoint_rate"])

        for obj_name, trials in by_object.items():
            deltas  = [t["delta_max_px"]    for t in trials]
            forces  = [t["force_max_n"]     for t in trials]
            times   = [t["grasp_time_ms"]   for t in trials]
            sp_rate = sum(1 for t in trials if t["reached_setpoint"]) / len(trials)

            dm, ds = statistics.mean(deltas), statistics.stdev(deltas) if len(deltas)>1 else 0
            fm, fs = statistics.mean(forces), statistics.stdev(forces) if len(forces)>1 else 0
            tm     = statistics.mean(times)

            print(f"  {obj_name:<13} {dm:>10.1f}px  {fm:>10.3f}N  "
                  f"{tm:>10.0f}ms  {sp_rate*100:>8.0f}%")
            w.writerow([obj_name, round(dm,2), round(ds,2),
                        round(fm,4), round(fs,4),
                        round(tm,1), round(sp_rate,3)])

    print("="*65)
    print(f"\nGuardado en {path}")


if __name__ == "__main__":
    run_versatility_test()
