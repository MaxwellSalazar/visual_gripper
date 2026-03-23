/*
 * ======================================================================
 *  visual_gripper_arduino.ino
 *  Arduino PWM Bridge for L298N Motor Driver
 *  DC Gripper Motor Controller — Serial Command Interface
 * ======================================================================
 *  Wiring (L298N):
 *    Arduino Pin 9  → ENA (PWM enable A)
 *    Arduino Pin 7  → IN1
 *    Arduino Pin 8  → IN2
 *    5V             → L298N +5V logic
 *    GND            → L298N GND
 *    Motor supply   → L298N +12V (or your motor voltage)
 *    Motor A Out    → Gripper motor terminals
 *
 *  Serial Protocol (2 bytes per command):
 *    Byte 1: Command char  'F'=forward(close) 'R'=reverse(open)
 *                          'S'=stop(coast)    'K'=brake(dynamic)
 *    Byte 2: PWM value [0–255]
 *
 *  Baud rate: 115200
 * ======================================================================
 */

// ── Pin definitions ────────────────────────────────────────────────────
const int PIN_ENA = 9;   // PWM pin (must be PWM-capable)
const int PIN_IN1 = 7;
const int PIN_IN2 = 8;

// ── Safety watchdog ────────────────────────────────────────────────────
// If no command received within WATCHDOG_MS, stop the motor.
// Prevents runaway if Raspberry Pi crashes mid-grasp.
const unsigned long WATCHDOG_MS = 500;
unsigned long lastCmdTime = 0;

// ── State ──────────────────────────────────────────────────────────────
char    currentCmd = 'S';
uint8_t currentPwm = 0;


void setup() {
  Serial.begin(115200);
  pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  motorStop();
  Serial.println("READY");   // signals Python that Arduino is live
}


void loop() {
  // ── Read serial command (blocking: wait for 2 bytes) ──────────────
  if (Serial.available() >= 2) {
    char    cmd = (char) Serial.read();
    uint8_t pwm =        Serial.read();
    lastCmdTime = millis();
    executeCommand(cmd, pwm);
  }

  // ── Watchdog: stop if no command for WATCHDOG_MS ──────────────────
  if (millis() - lastCmdTime > WATCHDOG_MS && currentCmd != 'S') {
    motorStop();
    currentCmd = 'S';
  }
}


void executeCommand(char cmd, uint8_t pwm) {
  currentCmd = cmd;
  currentPwm = pwm;

  switch (cmd) {
    case 'F':   // Forward — close gripper
      motorForward(pwm);
      break;
    case 'R':   // Reverse — open gripper
      motorReverse(pwm);
      break;
    case 'S':   // Coast stop
      motorStop();
      break;
    case 'K':   // Dynamic brake
      motorBrake();
      break;
    default:
      motorStop();
      break;
  }

  // Echo command for latency measurement on Python side
  Serial.write(cmd);
  Serial.write(pwm);
}


// ── Motor primitives ───────────────────────────────────────────────────

void motorForward(uint8_t pwm) {
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  analogWrite(PIN_ENA, pwm);
}

void motorReverse(uint8_t pwm) {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  analogWrite(PIN_ENA, pwm);
}

void motorStop() {
  // Coast: disable bridge, motor freewheels
  analogWrite(PIN_ENA, 0);
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
}

void motorBrake() {
  // Dynamic brake: both IN HIGH, PWM still active — shorts motor leads
  analogWrite(PIN_ENA, 255);
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, HIGH);
}
