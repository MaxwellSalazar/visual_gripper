"""
========================================================================
  control/motor_driver.py
  Motor Interface: DC Motor via L298N + Arduino (Serial PWM)
========================================================================
  Abstracts the serial communication with an Arduino acting as the
  PWM bridge for the L298N motor driver.

  Arduino sketch expected:
    void loop() {
      if (Serial.available() >= 2) {
        byte cmd = Serial.read();   // 'F'=forward, 'B'=brake, 'S'=stop
        byte pwm = Serial.read();   // 0–255
        if (cmd == 'F') { analogWrite(ENA, pwm); ... }
        if (cmd == 'S') { analogWrite(ENA, 0);   ... }
        if (cmd == 'B') { /* both IN pins HIGH = brake */ }
      }
    }

  If no Arduino is connected (e.g. desktop simulation), set
  SIMULATION_MODE = True in config/settings.py and a mock driver is used.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    SERIAL_PORT, SERIAL_BAUD, PWM_MAX, PWM_MIN, PWM_IDLE
)

# Optional: only import serial if available
try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


class MotorDriver:
    """
    Interface to the DC gripper motor via serial-connected Arduino + L298N.
    Falls back to simulation mode (prints commands) if serial is unavailable.

    Usage
    -----
    motor = MotorDriver()
    motor.open_connection()
    motor.set_pwm(180)          # close the gripper at PWM=180
    motor.brake()               # dynamic braking
    motor.stop()                # coast to stop
    motor.close_connection()
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud: int = SERIAL_BAUD,
        simulation: bool = not _SERIAL_AVAILABLE,
    ):
        self._port       = port
        self._baud       = baud
        self._simulation = simulation
        self._serial     = None
        self._current_pwm: int = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def open_connection(self) -> bool:
        """Open serial port. Returns True on success."""
        if self._simulation:
            print("[MotorDriver] SIMULATION MODE — no serial port.")
            return True
        if not _SERIAL_AVAILABLE:
            print("[MotorDriver] pyserial not installed. Run: pip install pyserial")
            return False
        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=1)
            time.sleep(2.0)  # Arduino resets on serial open; wait for it
            print(f"[MotorDriver] Connected → {self._port} @ {self._baud} baud")
            return True
        except serial.SerialException as e:
            print(f"[MotorDriver] ERROR: {e}")
            return False

    def close_connection(self) -> None:
        """Safely stop motor and close serial port."""
        self.stop()
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("[MotorDriver] Serial port closed.")

    # ------------------------------------------------------------------
    # Motor commands
    # ------------------------------------------------------------------

    def set_pwm(self, pwm: int, direction: str = "close") -> None:
        """
        Set motor speed (closing direction by default).

        Parameters
        ----------
        pwm       : duty cycle [0–255]
        direction : "close" | "open"
        """
        pwm = max(PWM_MIN, min(PWM_MAX, int(pwm)))
        self._current_pwm = pwm
        cmd_byte = b'F' if direction == "close" else b'B'

        if self._simulation:
            print(f"[MotorDriver] PWM={pwm:3d}  dir={direction}")
            return

        if self._serial and self._serial.is_open:
            try:
                self._serial.write(cmd_byte + bytes([pwm]))
            except Exception as e:
                print(f"[MotorDriver] Write error: {e}")

    def stop(self) -> None:
        """Coast stop — motor coasts to rest (low braking force)."""
        self._current_pwm = 0
        if self._simulation:
            print("[MotorDriver] STOP (coast)")
            return
        if self._serial and self._serial.is_open:
            self._serial.write(b'S' + bytes([0]))

    def brake(self) -> None:
        """Dynamic brake — short-circuit motor terminals for fast stop."""
        self._current_pwm = 0
        if self._simulation:
            print("[MotorDriver] BRAKE (dynamic)")
            return
        if self._serial and self._serial.is_open:
            self._serial.write(b'K' + bytes([0]))   # 'K' = brake in sketch

    @property
    def current_pwm(self) -> int:
        return self._current_pwm

    def __enter__(self):
        self.open_connection()
        return self

    def __exit__(self, *_):
        self.close_connection()
