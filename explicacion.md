# Visual Gripper — Guía de Ejecución Completa

> Respaldo de la explicación del experimento:
> **"Control de Agarre Adaptativo mediante Servoing Visual de Deformación Propioceptiva en Grippers Flexibles de Bajo Costo"**

---

## Pestaña 1 — Instalación

### Paso 1: Instalar dependencias Python

Dentro de la carpeta del proyecto, ejecuta:

```bash
cd visual_gripper
pip install -r requirements.txt
```

> **Importante:** Necesitas Python 3.9+ y `opencv-contrib-python` (no el básico) para que ArUco funcione.

---

### Paso 2: Verificar que la cámara funciona

Prueba rápida antes de cualquier otra cosa:

```bash
python -c "import cv2; c=cv2.VideoCapture(0); print(c.read()[0])"
```

Debe imprimir `True`. Si imprime `False`, cambia el `0` por `1` o `2` en `config/settings.py → CAMERA_INDEX`.

---

### Paso 3: Flashear el Arduino

Abre `docs/visual_gripper_arduino.ino` en el IDE de Arduino y sube el sketch. El Arduino actúa como puente serial: recibe bytes de Python y controla el L298N.

> **Watchdog incluido:** Si Python se cuelga, el Arduino para el motor automáticamente después de 500 ms.

---

### Paso 4: Calibrar el color del marcador

```bash
python calibration/color_picker.py
```

Se abre una ventana con sliders de H/S/V. Ajusta hasta que solo aparezca tu marcador en blanco. Presiona `p` para imprimir los valores y cópialos en `config/settings.py`:

```python
MARKER_COLOR_LOWER = [H_low, S_low, V_low]
MARKER_COLOR_UPPER = [H_high, S_high, V_high]
```

---

### Paso 5: Calibración fuerza–deflexión

```bash
python calibration/calibrate_force.py
```

El script te pide aplicar fuerzas conocidas al dedo y mide el desplazamiento en píxeles. Al final genera la curva `F = k·δ + b` y la guarda en `data/calibration/force_calibration.json`. Copia los valores resultantes a `settings.py`:

```python
CALIB_SLOPE     = <valor generado>
CALIB_INTERCEPT = <valor generado>
```

---

## Pestaña 2 — Ejecución

Con todo configurado, ejecuta el loop principal:

```bash
python main.py
```

### Controles de teclado

| Tecla   | Acción |
|---------|--------|
| `SPACE` | Fijar la pose de reposo — gripper abierto, dedo sin carga. Hazlo **siempre primero**. |
| `g`     | Iniciar secuencia de agarre — el motor cierra según el PID. |
| `o`     | Abrir el gripper y volver al estado idle. |
| `r`     | Resetear el controlador PID (integral a cero). |
| `q`     | Salir — guarda el CSV automáticamente. |

### Flujo de una prueba típica

1. Coloca el objeto a agarrar.
2. Presiona `SPACE` con el dedo libre (gripper abierto, sin carga).
3. Presiona `g` — observa cómo el motor reduce velocidad al detectar contacto.
4. El estado cambia a `HOLDING` cuando la deflexión se estabiliza.
5. Presiona `o` para soltar, luego `q` para salir.
6. Analiza los resultados:

```bash
python tests/analyze_results.py
```

---

## Pestaña 3 — Resultados esperados

### Métricas en condiciones normales de operación

| Métrica | Valor esperado | Condición |
|---------|---------------|-----------|
| FPS real de cámara | 30–60 fps | Webcam USB estándar |
| Latencia de detección | < 35 ms (P95) | Laptop moderno |
| R² de calibración | > 0.97 | Regresión lineal F = kδ + b |

### Figuras generadas para el paper

Al ejecutar `python tests/analyze_results.py`, se generan 5 figuras en `data/figures/` en formato PDF y PNG, listas para incluir en LaTeX:

**Figura 1 — Deflexión y fuerza vs. tiempo**
Verás una curva que crece al hacer contacto y se estabiliza. Ideal para demostrar el régimen de control. Incluye líneas de umbral para los estados `contact`, `soft_stop` y `hard_stop`.

**Figura 2 — PWM vs. tiempo (respuesta PID)**
El motor baja de ~200 a ~80–100 al detectar contacto. Muestra claramente la acción del controlador PID adaptando la velocidad del motor.

**Figura 3 — Histograma de latencia**
Distribución de tiempos de detección (detección → comando). Publicable como evidencia de operación en tiempo real. Incluye media y percentil 95.

**Figura 4 — Descomposición PID (P, I, D)**
Muestra la contribución de cada término a lo largo del tiempo. En un agarre bien sintonizado, el término D es dominante al inicio del contacto (respuesta ante la derivada de la deflexión).

**Figura 5 — Retrato de fase (F vs. dδ/dt)**
Figura diferenciadora para publicaciones Q1/Q2. Muestra la dinámica completa del sistema en el espacio de fases: fuerza estimada versus velocidad de deflexión. Revela el comportamiento del controlador bajo distintas rigideces de objeto.

### Qué debes ver en el display durante la ejecución

- El PWM arranca alto (~180–200) mientras el dedo no toca el objeto.
- Al detectar deflexión, el PWM baja bruscamente (acción del término D + feed-forward).
- Se estabiliza en un valor bajo (~60–90) cuando el régimen cambia a `HOLDING`.
- Si el PWM **no baja**, los umbrales `DEFLECTION_CONTACT` en `settings.py` necesitan ajuste porque el dedo tiene más o menos rigidez que los valores por defecto.

> **Si el marcador se pierde frecuentemente:** Ajusta la iluminación o amplía el rango HSV con `color_picker.py`. La tasa de pérdida del marcador queda registrada en el CSV (columna `marker_found`) y debe estar por debajo del **2%** para resultados publicables.

---

## Pestaña 4 — Configuración en VSCode

**Sí puedes probarlo completamente en VSCode.** El proyecto tiene dos mecanismos que permiten trabajar sin hardware físico:

- El `MotorDriver` entra automáticamente en **modo simulación** si no detecta el puerto serial (imprime los comandos en consola).
- `cv2.VideoCapture` acepta una **ruta de video `.mp4`** en lugar de un índice de cámara.

### 1. Extensiones recomendadas

- Python (Microsoft)
- Pylance
- Python Debugger
- GitLens

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

En VSCode presiona `Ctrl+Shift+P` → "Python: Select Interpreter" → elige el `.venv`.

### 3. Probar SIN Arduino ni cámara física

El driver de motor detecta automáticamente si no hay serial disponible y entra en modo simulación. Para la cámara, usa un video pregrabado del dedo del gripper:

```python
# En config/settings.py:
CAMERA_INDEX = "test_video.mp4"   # ruta a un video de prueba
```

OpenCV acepta rutas de video en `VideoCapture`, no solo índices de cámara. Graba el dedo con el celular, aplica deformaciones manualmente, y úsalo como entrada de prueba.

### 4. Configurar launch.json para depurar con F5

Crea el archivo `.vscode/launch.json` con este contenido:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Main experiment",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "console": "integratedTerminal"
    },
    {
      "name": "Color picker",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/calibration/color_picker.py",
      "console": "integratedTerminal"
    },
    {
      "name": "Analyze results",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/tests/analyze_results.py",
      "console": "integratedTerminal"
    }
  ]
}
```

Podrás ejecutar cualquier módulo con `F5` y poner breakpoints directamente en el loop de control.

> **Tip para depurar el estimador de estado:** Pon un breakpoint en `estimation/state_estimator.py → update()` para ver en tiempo real los valores de δ, F̂ y el régimen mientras corres el experimento con el video de prueba.

---

## Notas adicionales

### Orden correcto para la primera ejecución real

1. `color_picker.py` — sin esto el tracker no encuentra el marcador y todo el pipeline falla silenciosamente.
2. `calibrate_force.py` — esto es lo que convierte el experimento en publicable (sin curva de calibración, los Newtons estimados no tienen respaldo científico).
3. `main.py` — el loop principal del experimento.

### Sobre el tipo de marcador (recomendación para Q1/Q2)

Usa **ArUco** (implementado en `perception/aruco_tracker.py`). Ventajas frente al marcador de color:

- Precisión sub-píxel en el centroide.
- Inmune a cambios de iluminación y color del objeto agarrado.
- En el paper puedes afirmar robustez ante condiciones de iluminación variables — imposible de argumentar con HSV.

Imprime el marcador en: **chev.me/arucogen/** → Diccionario `4×4`, ID `0`, tamaño 30 mm × 30 mm.

### Ajuste del PID (si el agarre es inestable)

| Síntoma | Ajuste |
|---------|--------|
| Motor oscila al hacer contacto | Reducir `PID_KP` |
| Respuesta muy lenta al contacto | Aumentar `PID_KD` |
| Deriva lenta en `HOLDING` | Aumentar `PID_KI` |
| Para antes de tocar el objeto | Reducir `DEFLECTION_CONTACT` |
| No para al tocar objetos frágiles | Reducir `DEFLECTION_SOFT_STOP` |
