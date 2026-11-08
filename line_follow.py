"""วิ่งตามเส้นอัตโนมัติ - กดปุ่มบนจอยค้างไว้ถึงจะวิ่ง

ปล่อยปุ่ม / จอยหลุด / ไม่เห็นเส้น = หยุดทันที
ดูภาพสดและปรับค่าที่ http://<IP ของ Pi>:8000
"""
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import pygame
import RPi.GPIO as GPIO

from car import Car

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "servo_config.json")))
CENTER = float(cfg["SERVO_CENTER"])
LEFT = float(cfg["SERVO_LEFT"])
RIGHT = float(cfg["SERVO_RIGHT"])

SERVO_PIN = 25
DEAD_MAN_BUTTON = 1

WIDTH, HEIGHT = 320, 240
ROI_TOP, ROI_BOTTOM = 125, 180
MIN_PIXELS = 75

TARGET_MAX_RPM = 360.0
KP, KI = 0.04, 0.15
MAX_CORRECTION = 45.0
PWM_MIN, PWM_MAX = 20.0, 60.0
RPM_ALPHA = 0.25

LOST_TIMEOUT = 1.0
LOOP_TIME = 0.05
PORT = 8000

state = {"thresh": 70, "gain": 4.0, "target": 120.0,
         "err": 0.0, "duty": CENTER, "px": 0, "found": False,
         "run": False, "joy": False,
         "lrpm": 0.0, "rrpm": 0.0, "lpwm": 0.0, "rpwm": 0.0}
frames = {"view": None, "mask": None}
lock = threading.Lock()
last_seen = 0.0
STOP = False

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(CENTER)

car = Car()


import os
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
ENC_EVERY = 6
RAMP_TIME = 1.0
SERVO_DEADBAND = 0.08


def vision():
    global last_seen
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    err_f = 0.0
    n = 0
    fps_t = time.time()

    while not STOP:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        n += 1

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        with lock:
            th = state["thresh"]

        roi = frame[ROI_TOP:ROI_BOTTOM, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        mask = cv2.inRange(gray, 0, th)
        mask = cv2.dilate(cv2.erode(mask, None, iterations=1), None, iterations=2)

        px = int(cv2.countNonZero(mask))
        found = px >= MIN_PIXELS
        cx = WIDTH // 2

        if found:
            m = cv2.moments(mask)
            cx = int(m["m10"] / m["m00"])
            err_f += 0.4 * (((cx - WIDTH / 2) / (WIDTH / 2)) - err_f)
            last_seen = time.time()

        with lock:
            state["err"] = err_f
            state["px"] = px
            state["found"] = found

        if n % 30 == 0:
            now = time.time()
            with lock:
                state['fps'] = 30.0 / max(1e-6, now - fps_t)
            fps_t = now

        if n % ENC_EVERY == 0:
            cv2.rectangle(frame, (0, ROI_TOP), (WIDTH - 1, ROI_BOTTOM), (0, 255, 255), 2)
            cv2.line(frame, (WIDTH // 2, ROI_TOP), (WIDTH // 2, ROI_BOTTOM), (255, 0, 0), 1)
            if found:
                cv2.line(frame, (cx, ROI_TOP), (cx, ROI_BOTTOM), (0, 0, 255), 2)
            _, j1 = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            j2b = None
            if n % (ENC_EVERY * 3) == 0:
                _, j2 = cv2.imencode(".jpg", mask, [cv2.IMWRITE_JPEG_QUALITY, 50])
                j2b = j2.tobytes()
            with lock:
                frames["view"] = j1.tobytes()
                if j2b is not None:
                    frames["mask"] = j2b



def control():
    pygame.init()
    pygame.joystick.init()
    js = None
    li = ri = 0.0
    lf = rf = 0.0
    lcmd = rcmd = 0.0
    lrpm = rrpm = 0.0
    run_t0 = None
    last_duty = -99.0
    os.makedirs(LOGDIR, exist_ok=True)
    csv_path = os.path.join(LOGDIR, 'run_' + time.strftime('%Y%m%d_%H%M%S') + '.csv')
    csvf = open(csv_path, 'w', buffering=1)
    csvf.write('wall,t,run,found,err,px,duty,lpwm,rpwm,lrpm,rrpm,fps\n')
    t0 = time.time()
    print('บันทึกข้อมูลลง', csv_path)

    while not STOP:
        pygame.event.pump()

        if pygame.joystick.get_count() == 0:
            js = None
            pygame.joystick.quit()
            pygame.joystick.init()
        elif js is None:
            js = pygame.joystick.Joystick(0)
            js.init()

        pressed = False
        if js is not None:
            try:
                pressed = bool(js.get_button(DEAD_MAN_BUTTON))
            except Exception:
                js = None
        if not pressed and os.path.exists('/tmp/GO'):
            pressed = True

        with lock:
            err = state["err"]
            gain = state["gain"]
            target = state["target"]

        a = max(-1.0, min(1.0, err * gain))
        if a >= 0:
            duty = CENTER - a * (CENTER - RIGHT)
        else:
            duty = CENTER + (-a) * (LEFT - CENTER)
        if abs(duty - last_duty) >= SERVO_DEADBAND:
            servo.ChangeDutyCycle(duty)
            last_duty = duty

        fresh = (time.time() - last_seen) < LOST_TIMEOUT
        run = pressed and fresh

        if run:
            if run_t0 is None:
                run_t0 = time.time()
            ramp = min(1.0, (time.time() - run_t0) / RAMP_TIME)
            base = max(15.0, min(60.0, target / TARGET_MAX_RPM * 50.0))
            lrpm = abs(car.get_left_rpm())
            rrpm = abs(car.get_right_rpm())
            lf += RPM_ALPHA * (lrpm - lf)
            rf += RPM_ALPHA * (rrpm - rf)

            le = target - lf
            li = max(-1000.0, min(1000.0, li + le * LOOP_TIME))
            lc = max(-MAX_CORRECTION, min(MAX_CORRECTION, KP * le + KI * li))
            lcmd = max(PWM_MIN, min(PWM_MAX, base + lc))
            rcmd = lcmd
            ri, rf = li, lf

            if ramp < 1.0:
                lcmd *= ramp
                rcmd *= ramp
                li = ri = 0.0

            car.set_left_speed(lcmd)
            car.set_right_speed(rcmd)
        else:
            run_t0 = None
            car.stop()
            li = ri = 0.0
            lf = rf = 0.0
            lcmd = rcmd = 0.0
            lrpm = rrpm = 0.0

        with lock:
            state["duty"] = duty
            state["run"] = run
            state["joy"] = js is not None
            state["lrpm"] = lrpm
            state["rrpm"] = rrpm
            state["lpwm"] = lcmd
            state["rpwm"] = rcmd
            fps_s = state['fps'] if 'fps' in state else 0.0
            found_s = state['found'] if 'found' in state else False
            px_s = state['px'] if 'px' in state else 0

        csvf.write('%s,%.2f,%d,%d,%.4f,%d,%.3f,%.1f,%.1f,%.0f,%.0f,%.1f\n' % (
            time.strftime('%H:%M:%S'), time.time() - t0, int(run), int(found_s),
            err, px_s, duty, lcmd, rcmd, lrpm, rrpm, fps_s))

        time.sleep(LOOP_TIME)


PAGE = """<html><head><meta charset="utf-8"><title>line follow</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}
img{width:640px;max-width:100%;border:1px solid #444;display:block;margin:8px 0}
button{font-size:17px;margin:3px;padding:9px 14px}
#st{font-size:18px;margin:10px 0;font-family:monospace;white-space:pre-wrap}
#go{font-size:26px;font-weight:bold;padding:10px;margin:8px 0}</style></head><body>
<div id="go">-</div>
<div id="st">-</div>
<div>
<button onclick="s('thresh',-5)">thresh -</button>
<button onclick="s('thresh',5)">thresh +</button>
<button onclick="s('gain',-0.2)">gain -</button>
<button onclick="s('gain',0.2)">gain +</button>
<button onclick="s('target',-10)">speed -</button>
<button onclick="s('target',10)">speed +</button>
</div>
<img src="/stream"><img src="/mask">
<script>
function s(k,v){fetch('/set?'+k+'='+v)}
setInterval(async()=>{try{const d=await (await fetch('/status')).json();
const go=document.getElementById('go');
go.textContent = d.run ? 'RUNNING' : 'STOPPED';
go.style.color = d.run ? '#4f4' : '#f55';
document.getElementById('st').textContent =
'joy='+(d.joy?'OK':'-')+'  line='+(d.found?'OK':'-')+'  err='+d.err.toFixed(2)+
'  duty='+d.duty.toFixed(2)+'  px='+d.px+'\\n'+
'thresh='+d.thresh+'  gain='+d.gain.toFixed(1)+'  speed='+d.target.toFixed(0)+' rpm\\n'+
'PWM  L='+d.lpwm.toFixed(1)+'  R='+d.rpwm.toFixed(1)+
'   RPM  L='+d.lrpm.toFixed(0)+'  R='+d.rrpm.toFixed(0)}catch(e){}},300)
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        elif u.path == "/set":
            with lock:
                if "thresh" in q:
                    state["thresh"] = max(5, min(250, state["thresh"] + int(float(q["thresh"][0]))))
                if "gain" in q:
                    state["gain"] = max(0.2, min(8.0, round(state["gain"] + float(q["gain"][0]), 2)))
                if "target" in q:
                    state["target"] = max(60.0, min(360.0, state["target"] + float(q["target"][0])))
            self.send_response(204)
            self.end_headers()

        elif u.path == "/status":
            with lock:
                b = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        elif u.path in ("/stream", "/mask"):
            key = "view" if u.path == "/stream" else "mask"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with lock:
                        buf = frames[key]
                    if buf:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n")
                    time.sleep(0.05)
            except Exception:
                pass

        else:
            self.send_response(404)
            self.end_headers()


threading.Thread(target=vision, daemon=True).start()
threading.Thread(target=control, daemon=True).start()

srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
print("เปิดเบราว์เซอร์ไปที่ http://<IP ของ Pi>:%d" % PORT)
print("กดปุ่ม %d บนจอยค้างไว้เพื่อให้รถวิ่ง | Ctrl+C เพื่อหยุด" % DEAD_MAN_BUTTON)

try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    STOP = True
    time.sleep(0.3)
    car.stop()
    servo.ChangeDutyCycle(CENTER)
    time.sleep(1.0)
    servo.stop()
    del servo
    car.cleanup()
    GPIO.cleanup()
    print("จบ")
