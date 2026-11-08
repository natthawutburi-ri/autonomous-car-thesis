"""วิ่งตามเส้นอัตโนมัติ - กดปุ่มบนจอยค้างไว้ถึงจะวิ่ง

ปล่อยปุ่ม / จอยหลุด / ไม่เห็นเส้น = หยุดทันที
ดูภาพสดและปรับค่าที่ http://<IP ของ Pi>:8000
"""
import json, os, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
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
ROI_TOP, ROI_BOTTOM = (int(os.environ.get("ROI_T","35")), int(os.environ.get("ROI_B","90"))) if os.environ.get("LANE","1")=="2" else (int(os.environ.get("ROI_T","125")), int(os.environ.get("ROI_B","180")))
MIN_PIXELS = 75
LANE_MODE = int(os.environ.get('LANE', '1'))
MIN_COL = int(os.environ.get('MIN_COL', '8'))
LANE_PX = float(os.environ.get('LANE_PX', '239'))
LANE_TOL = float(os.environ.get('LANE_TOL', '0.25'))
OBS_T = int(os.environ.get('OBS_T', '55'))
OBS_B = int(os.environ.get('OBS_B', '230'))
OBS_HALF = int(os.environ.get('OBS_HALF', '70'))
OBS_MIN = int(os.environ.get('OBS_MIN', '120'))
OBS_K2 = 0.000760843
OBS_K1 = -0.472975983
OBS_K0 = 88.6728346
BRAKE_CM = float(os.environ.get('BRAKE_CM', '55'))
CLEAR_CM = float(os.environ.get('CLEAR_CM', '62'))
AEB = int(os.environ.get('AEB', '1'))
obs_brake = False
obs_cm = -1.0
prev_xl = None
prev_xr = None
prev_cx = 160.0
AVOID = int(os.environ.get('AVOID', '0'))
AVOID_CM = float(os.environ.get('AVOID_CM', '55'))
AVOID_PX = float(os.environ.get('AVOID_PX', '112'))
AVOID_HOLD = float(os.environ.get('AVOID_HOLD', '2.0'))
AVOID_TOL = float(os.environ.get('AVOID_TOL', '25'))
avoid_dir = 0.0
avoid_until = 0.0

TARGET_MAX_RPM = 360.0
KP, KI = 0.04, 0.15
MAX_CORRECTION = 80.0
PWM_MIN, PWM_MAX = 10.0, 100.0
RPM_ALPHA = 0.12

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


# ---------- ชั้นสื่อสารกับ ROS 2 ----------
ROS_HOST = "127.0.0.1"
ROS_TX_PORT = 9870   # ส่งข้อมูลสถานะไปยังโนดสื่อสาร
ROS_RX_PORT = 9871   # รับคำสั่งจากโนดสื่อสาร
ros_stop = False
volt5 = 0.0
_ros_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def ros_send(run, found, err, px, duty, lpwm, rpwm, lrpm, rrpm, fps):
    try:
        _ros_sock.sendto(json.dumps({
            "run": bool(run), "found": bool(found),
            "err": float(err), "px": int(px), "duty": float(duty),
            "pwm_l": float(lpwm), "pwm_r": float(rpwm),
            "lrpm": float(lrpm), "rrpm": float(rrpm),
            "fps": float(fps), "volt": float(volt5),
        }).encode(), (ROS_HOST, ROS_TX_PORT))
    except Exception:
        pass


def ros_cmd_listener():
    global ros_stop
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", ROS_RX_PORT))
    s.settimeout(1.0)
    while not STOP:
        try:
            data, _ = s.recvfrom(1024)
            cmd = json.loads(data.decode())
        except Exception:
            continue
        if "target_rpm" in cmd:
            with lock:
                state["target"] = float(cmd["target_rpm"])
            print("ROS สั่งความเร็วเป้าหมาย", cmd["target_rpm"], "rpm")
        if "stop" in cmd:
            ros_stop = bool(cmd["stop"])
            print("ROS สั่งหยุด", ros_stop)


def volt_reader():
    global volt5
    import re
    import subprocess
    while not STOP:
        try:
            out = subprocess.check_output(
                ["vcgencmd", "pmic_read_adc", "EXT5V_V"], text=True)
            m = re.search(r"=\s*([\d.]+)V", out)
            if m:
                volt5 = float(m.group(1))
        except Exception:
            pass
        time.sleep(2.0)


def open_camera():
    import glob as _g
    cands = sorted(_g.glob("/dev/v4l/by-id/*-video-index0"))
    cands += ["/dev/video%d" % i for i in range(4)]
    for d in cands:
        cap = cv2.VideoCapture(d, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            print("เปิดกล้องที่", d)
            return cap
        cap.release()
    raise RuntimeError("ไม่พบกล้องที่ใช้งานได้")


def find_peaks(colsum, k=4):
    seg = colsum.astype(float).copy()
    out = []
    for _ in range(k):
        p = int(np.argmax(seg))
        if seg[p] < MIN_COL:
            break
        a = b = p
        while a > 0 and seg[a - 1] >= MIN_COL:
            a -= 1
        while b < seg.size - 1 and seg[b + 1] >= MIN_COL:
            b += 1
        xs = np.arange(a, b + 1, dtype=float)
        w = seg[a:b + 1].copy()
        out.append((float((xs * w).sum() / w.sum()), float(w.sum())))
        seg[a:b + 1] = 0.0
    return out


def pick_edges(peaks):
    if not peaks:
        return None, None
    tol = LANE_TOL * LANE_PX
    best = None
    n = len(peaks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted((peaks[i][0], peaks[j][0]))
            d = abs((b - a) - LANE_PX)
            if d <= tol and (best is None or d < best[0]):
                best = (d, a, b)
    if best is not None:
        return best[1], best[2]
    x0 = peaks[0][0]
    if prev_xl is not None and prev_xr is not None:
        if abs(x0 - prev_xl) <= abs(x0 - prev_xr):
            return x0, None
        return None, x0
    if prev_xl is not None:
        return x0, None
    if prev_xr is not None:
        return None, x0
    if x0 < WIDTH / 2.0:
        return x0, None
    return None, x0


def find_edge(colsum, lo, hi):
    seg = colsum[lo:hi]
    if seg.size == 0:
        return None
    p = int(np.argmax(seg))
    if seg[p] < MIN_COL:
        return None
    a = b = p
    while a > 0 and seg[a - 1] >= MIN_COL:
        a -= 1
    while b < seg.size - 1 and seg[b + 1] >= MIN_COL:
        b += 1
    x = np.arange(a, b + 1, dtype=float)
    w = seg[a:b + 1].astype(float)
    return lo + float((x * w).sum() / w.sum())


def vision():
    global last_seen, prev_xl, prev_xr, obs_brake, obs_cm, prev_cx
    global avoid_dir, avoid_until
    cap = open_camera()
    _f = os.environ.get('CAM_FMT', 'MJPG')
    print('รูปแบบภาพ:', _f)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*_f))
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
        cx = WIDTH // 2
        xl = xr = None

        if LANE_MODE == 2:
            colsum = (mask > 0).sum(axis=0)
            xl, xr = pick_edges(find_peaks(colsum))
            prev_xl, prev_xr = xl, xr
            if xl is not None and xr is not None:
                cx = int(round((xl + xr) / 2.0))
            elif xl is not None or xr is not None:
                _x1 = xl if xl is not None else xr
                _ca = _x1 + LANE_PX / 2.0
                _cb = _x1 - LANE_PX / 2.0
                if abs(_ca - prev_cx) <= abs(_cb - prev_cx):
                    cx = int(round(_ca))
                else:
                    cx = int(round(_cb))
            cx = max(0, min(WIDTH - 1, cx))
            found = (xl is not None) or (xr is not None)
            if found:
                prev_cx = float(cx)
        else:
            found = px >= MIN_PIXELS
            if found:
                m = cv2.moments(mask)
                cx = int(m["m10"] / m["m00"])

        if found:
            err_f += 0.4 * (((cx - WIDTH / 2) / (WIDTH / 2)) - err_f)
            last_seen = time.time()

        oa0 = max(0, min(WIDTH - 2, cx - OBS_HALF))
        oa1 = max(oa0 + 1, min(WIDTH, cx + OBS_HALF))
        obs = False
        obs_px = 0
        obs_row = -1
        obs_cx = -1.0
        if OBS_B > OBS_T:
            oroi = frame[OBS_T:OBS_B, oa0:oa1]
            og = cv2.GaussianBlur(cv2.cvtColor(oroi, cv2.COLOR_BGR2GRAY),
                                  (7, 7), 0)
            om = cv2.inRange(og, 0, th)
            om = cv2.dilate(cv2.erode(om, None, iterations=1), None,
                            iterations=2)
            obs_px = int(cv2.countNonZero(om))
            obs = obs_px >= OBS_MIN
            if obs:
                rr = np.nonzero(om.sum(axis=1) > 0)[0]
                if rr.size:
                    obs_row = int(OBS_T + rr.max())
                _cc = om.sum(axis=0).astype(float)
                if _cc.sum() > 0:
                    obs_cx = oa0 + float(
                        (np.arange(_cc.size) * _cc).sum() / _cc.sum())

        if obs and obs_row > 0:
            obs_cm = OBS_K2 * obs_row * obs_row + OBS_K1 * obs_row + OBS_K0
        else:
            obs_cm = -1.0
        err_out = err_f
        if AVOID:
            _straight = (xl is not None and xr is not None and
                         abs((xr - xl) - LANE_PX) <= AVOID_TOL)
            if obs and 0.0 < obs_cm <= AVOID_CM and _straight:
                avoid_dir = -1.0 if obs_cx >= cx else 1.0
                avoid_until = time.time() + AVOID_HOLD
            if time.time() < avoid_until:
                err_out = max(-1.0, min(1.0, err_f
                                        + avoid_dir * AVOID_PX / 160.0))
                prev_cx = 160.0 - avoid_dir * LANE_PX / 2.0
            else:
                avoid_dir = 0.0
        if AEB:
            _geo = (LANE_MODE != 2) or (xl is not None and xr is not None)
            if obs and 0.0 < obs_cm <= BRAKE_CM and _geo:
                obs_brake = True
            else:
                with lock:
                    _pr = bool(state.get('pressed', False))
                if not _pr:
                    obs_brake = False
        else:
            obs_brake = False

        with lock:
            state["err"] = err_out
            state["avoid"] = bool(time.time() < avoid_until)
            state["obs_cx"] = round(obs_cx, 1)
            state["px"] = px
            state["found"] = found
            state["xl"] = -1.0 if xl is None else round(xl, 1)
            state["xr"] = -1.0 if xr is None else round(xr, 1)
            state["lane"] = -1.0 if (xl is None or xr is None) else round(xr - xl, 1)
            state["obs"] = bool(obs)
            state["obs_px"] = obs_px
            state["obs_row"] = obs_row
            state["obs_cm"] = round(obs_cm, 1)
            state["brake"] = bool(obs_brake)

        if n % 30 == 0:
            now = time.time()
            with lock:
                state['fps'] = 30.0 / max(1e-6, now - fps_t)
            fps_t = now

        if n % ENC_EVERY == 0:
            cv2.rectangle(frame, (0, ROI_TOP), (WIDTH - 1, ROI_BOTTOM), (0, 255, 255), 2)
            cv2.line(frame, (WIDTH // 2, ROI_TOP), (WIDTH // 2, ROI_BOTTOM), (255, 0, 0), 1)
            if xl is not None:
                cv2.line(frame, (int(xl), ROI_TOP), (int(xl), ROI_BOTTOM), (0, 255, 0), 2)
            if xr is not None:
                cv2.line(frame, (int(xr), ROI_TOP), (int(xr), ROI_BOTTOM), (0, 255, 0), 2)
            if found:
                cv2.line(frame, (cx, ROI_TOP), (cx, ROI_BOTTOM), (0, 0, 255), 2)
            cv2.rectangle(frame, (oa0, OBS_T), (oa1 - 1, OBS_B),
                          (255, 0, 255), 1)
            if obs and obs_row > 0:
                cv2.line(frame, (oa0, obs_row), (oa1 - 1, obs_row),
                         (0, 165, 255), 3)
            if time.time() < avoid_until:
                _t = int(round(max(0, min(WIDTH - 1,
                                          160 + err_out * 160))))
                cv2.line(frame, (_t, ROI_TOP), (_t, ROI_BOTTOM),
                         (255, 255, 0), 2)
                cv2.rectangle(frame, (2, 2), (WIDTH - 3, HEIGHT - 3),
                              (255, 255, 0), 4)
            if obs_brake:
                cv2.rectangle(frame, (2, 2), (WIDTH - 3, HEIGHT - 3),
                              (0, 0, 255), 4)
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
    try:
        os.remove('/tmp/GO')
    except OSError:
        pass
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
    csvf.write('wall,t,run,found,err,px,duty,lpwm,rpwm,lrpm,rrpm,fps,thresh,gain,target\n')
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
        run = pressed and fresh and (not ros_stop) and (not obs_brake)

        if run:
            if run_t0 is None:
                run_t0 = time.time()
            ramp = min(1.0, (time.time() - run_t0) / RAMP_TIME)
            base = max(8.0, min(60.0, target / TARGET_MAX_RPM * 50.0))
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
            state["pressed"] = bool(pressed)
            state["joy"] = js is not None
            state["lrpm"] = lrpm
            state["rrpm"] = rrpm
            state["lpwm"] = lcmd
            state["rpwm"] = rcmd
            fps_s = state['fps'] if 'fps' in state else 0.0
            found_s = state['found'] if 'found' in state else False
            px_s = state['px'] if 'px' in state else 0

        csvf.write('%s,%.2f,%d,%d,%.4f,%d,%.3f,%.1f,%.1f,%.0f,%.0f,%.1f,%d,%.1f,%.0f\n' % (
            time.strftime('%H:%M:%S'), time.time() - t0, int(run), int(found_s),
            err, px_s, duty, lcmd, rcmd, lrpm, rrpm, fps_s,
            state['thresh'], state['gain'], state['target']))
        ros_send(run, found_s, err, px_s, duty, lcmd, rcmd, lrpm, rrpm, fps_s)

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
                    state["target"] = max(20.0, min(360.0, state["target"] + float(q["target"][0])))
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
threading.Thread(target=ros_cmd_listener, daemon=True).start()
threading.Thread(target=volt_reader, daemon=True).start()

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
