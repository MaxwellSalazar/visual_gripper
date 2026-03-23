"""
tests/test_latency.py — Análisis de Latencia del Pipeline

MÉTRICA CRÍTICA PARA LA PUBLICACIÓN:
    Tiempo desde que la cámara detecta el inicio de la deformación
    hasta que el motor recibe la señal de cambio de PWM.

    Latencia total = t_captura + t_percepcion + t_estimacion + t_control + t_motor

METODOLOGÍA:
    1. El gripper toca un objeto mientras el script corre a máxima velocidad
    2. Se detecta el frame donde δ supera un umbral de "inicio de contacto"
    3. Se mide el tiempo desde ese frame hasta el primer cambio de PWM
    4. Se repite N veces para obtener estadísticas robustas

USO:
    python tests/test_latency.py

OUTPUT:
    - Tabla de latencias por ciclo
    - CSV en data/latency_results.csv
    - Estadísticas: media, mediana, P95, P99 (para la tabla de resultados)
"""

import sys
import os
import time
import csv
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from core.perception   import MarkerPerception, CameraStream
from core.state_estimator import StateEstimator
from core.pid_controller  import PIDController
from control.motor_driver import MotorDriver


# Umbral de inicio de contacto para esta prueba
CONTACT_THRESHOLD_PX = 8.0     # δ > 8px → se considera contacto inicial
N_TRIALS             = 30       # Repeticiones para estadística robusta


def run_latency_test():
    print("\n" + "="*60)
    print("TEST DE LATENCIA — Visual Soft Gripper")
    print(f"Umbral de contacto: {CONTACT_THRESHOLD_PX} px")
    print(f"Número de trials: {N_TRIALS}")
    print("="*60)

    camera    = CameraStream()
    percept   = MarkerPerception()
    estimator = StateEstimator()
    pid       = PIDController()
    motor     = MotorDriver(simulation=True)   # Sin mover el motor real

    # Auto-calibrar posición de reposo
    print("\n[Paso 1] Calibrando posición de reposo (3s)...")
    rest_centroids = []
    t0 = time.time()
    while time.time() - t0 < 3.0:
        ok, frame = camera.read()
        if not ok:
            continue
        centroid, _, _ = percept.process(frame)
        if centroid:
            rest_centroids.append(centroid)
    if len(rest_centroids) < 5:
        print("ERROR: marcador no detectado.")
        return
    estimator.auto_calibrate_rest(rest_centroids)

    # ── Loop de prueba ────────────────────────────────────────────────────────
    latencies_ms = []
    cycle_times  = []   # Para calcular FPS real del pipeline

    print(f"\n[Paso 2] Ejecutando {N_TRIALS} ciclos de medición...")
    print("Presiona CTRL+C para detener antes de terminar.\n")

    import cv2

    for trial in range(N_TRIALS):
        # Esperar a que el gripper vuelva a reposo entre trials
        print(f"  Trial {trial+1}/{N_TRIALS} — Lleva el gripper a posición libre "
              f"y presiona cualquier tecla...", end=" ", flush=True)
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            centroid, debug, _ = percept.process(frame)
            state = estimator.update(centroid)
            cv2.imshow("Latency Test", debug)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:   # Cualquier tecla
                break

        # Medir latencia de este trial
        t_contact_detected = None
        t_motor_commanded  = None
        prev_pwm           = motor.current_pwm

        t_trial_start = time.perf_counter()
        frames_in_trial = 0

        for _ in range(200):   # Máximo 200 frames por trial (~3s a 60fps)
            t_frame_start = time.perf_counter()

            ok, frame = camera.read()
            if not ok:
                continue

            t_capture = time.perf_counter()

            centroid, debug, _ = percept.process(frame)
            t_percept = time.perf_counter()

            state = estimator.update(centroid)
            t_estim = time.perf_counter()

            control = pid.compute(state)
            t_ctrl  = time.perf_counter()

            motor.apply_control(control)
            t_motor = time.perf_counter()

            frames_in_trial += 1

            # Detectar inicio de contacto
            delta = state.get("delta_px", 0.0)
            if t_contact_detected is None and delta >= CONTACT_THRESHOLD_PX:
                t_contact_detected = t_capture
                print(f"δ={delta:.1f}px detectado", end=" ", flush=True)

            # Detectar primer cambio significativo en PWM tras el contacto
            new_pwm = control.get("pwm", 0.0)
            if (t_contact_detected is not None and
                    t_motor_commanded is None and
                    abs(new_pwm - prev_pwm) > 2.0):
                t_motor_commanded = t_motor

            prev_pwm = new_pwm
            cycle_times.append(t_motor - t_frame_start)

            # Mostrar frame
            info = (f"T{trial+1} | delta={delta:.1f}px | "
                    f"contact={'YES' if t_contact_detected else 'NO'}")
            cv2.putText(debug, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imshow("Latency Test", debug)
            cv2.waitKey(1)

            if t_motor_commanded is not None:
                break   # Tenemos los dos timestamps — trial completo

        # Calcular latencia de este trial
        if t_contact_detected and t_motor_commanded:
            lat_ms = (t_motor_commanded - t_contact_detected) * 1000
            latencies_ms.append(lat_ms)
            print(f"→ Latencia: {lat_ms:.2f} ms ✓")
        else:
            print("→ No se detectó contacto o cambio de PWM. Trial descartado.")

        pid.reset()

    camera.release()
    cv2.destroyAllWindows()

    # ── Estadísticas ──────────────────────────────────────────────────────────
    _print_and_save_stats(latencies_ms, cycle_times)


def _print_and_save_stats(latencies_ms: list, cycle_times: list):
    if not latencies_ms:
        print("\nNo hay datos válidos.")
        return

    mean_lat  = statistics.mean(latencies_ms)
    med_lat   = statistics.median(latencies_ms)
    std_lat   = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0
    sorted_l  = sorted(latencies_ms)
    p95       = sorted_l[int(len(sorted_l) * 0.95)]
    p99       = sorted_l[min(int(len(sorted_l) * 0.99), len(sorted_l)-1)]

    mean_fps  = 1.0 / statistics.mean(cycle_times) if cycle_times else 0

    print("\n" + "="*60)
    print("RESULTADOS DE LATENCIA")
    print(f"  Trials válidos  : {len(latencies_ms)}")
    print(f"  Media           : {mean_lat:.2f} ms")
    print(f"  Mediana         : {med_lat:.2f} ms")
    print(f"  Desv. estándar  : {std_lat:.2f} ms")
    print(f"  P95             : {p95:.2f} ms")
    print(f"  P99             : {p99:.2f} ms")
    print(f"  Mínima          : {min(latencies_ms):.2f} ms")
    print(f"  Máxima          : {max(latencies_ms):.2f} ms")
    print(f"  FPS pipeline    : {mean_fps:.1f}")
    print("="*60)

    # Guardar CSV
    os.makedirs("data", exist_ok=True)
    path = "data/latency_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "latency_ms"])
        for i, l in enumerate(latencies_ms):
            w.writerow([i+1, round(l, 3)])
        w.writerow([])
        w.writerow(["metric", "value_ms"])
        w.writerow(["mean",   round(mean_lat, 3)])
        w.writerow(["median", round(med_lat,  3)])
        w.writerow(["std",    round(std_lat,  3)])
        w.writerow(["p95",    round(p95, 3)])
        w.writerow(["p99",    round(p99, 3)])
    print(f"\nResultados guardados en {path}")


if __name__ == "__main__":
    run_latency_test()
