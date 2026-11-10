#!/usr/bin/env python3
"""
LipReader - Screen/Camera + Audio -> Live Transcript with Visual Speaker ID
1. MediaPipe FaceMesh finds faces and tracks lip movement
2. faster-whisper transcribes mic/system audio (works for whispers)
3. Lip movement in last 3s decides which on-screen speaker said it
"""
import argparse, sys, time, threading, json
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

try: import mss
except ImportError: mss = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_python = None
    mp_vision = None
try:
    import sounddevice as sd
except ImportError: sd = None
try: from faster_whisper import WhisperModel
except ImportError: WhisperModel = None

# FaceMesh lip landmarks
LEFT, RIGHT, UPPER, LOWER = 61, 291, 13, 14
LIP_POINTS = [61,185,40,39,37,267,269,270,291,13,14,0,17]

def fmt_time(s): return f"{int(s//60):02d}:{int(s%60):02d}"
def srt_time(s):
    ms=int((s-int(s))*1000); s=int(s)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d},{ms:03d}"

class Track:
    def __init__(self, tid, cx, cy, bbox, mar, mouth_box):
        self.id=tid; self.cx=cx; self.cy=cy; self.bbox=bbox
        self.prev_mar=mar; self.mar=mar; self.mouth_box=mouth_box
        self.movement=deque(); self.last_seen=time.time()
    def update(self, cx, cy, bbox, mar, mouth_box):
        now=time.time()
        self.movement.append((now, abs(mar-self.prev_mar)))
        while self.movement and self.movement[0][0] < now-5: self.movement.popleft()
        self.cx,self.cy,self.bbox,self.mar,self.mouth_box=cx,cy,bbox,mar,mouth_box
        self.prev_mar=mar; self.last_seen=now
    def movement_in_window(self, sec):
        now=time.time()
        return sum(v for t,v in self.movement if t>=now-sec)

class LipTracker:
    def __init__(self, model_path="face_landmarker.task"):
        if mp is None:
            raise RuntimeError("mediapipe not installed: pip install -U mediapipe")
        model_path = Path(model_path)
        if not model_path.exists():
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            print(f"Downloading FaceLandmarker model to {model_path} ...")
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.tracks = []
        self.next_id = 1
        self._last_ts = 0

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._last_ts = max(self._last_ts + 1, int(time.time() * 1000))
        result = self.landmarker.detect_for_video(mp_image, self._last_ts)
        detections = []
        if result.face_landmarks:
            for face in result.face_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in face], dtype=np.float32)
                mar = np.linalg.norm(pts[UPPER] - pts[LOWER]) / (np.linalg.norm(pts[LEFT] - pts[RIGHT]) + 1e-6)
                cx, cy = pts.mean(axis=0)
                x1, y1 = pts.min(axis=0).astype(int)
                x2, y2 = pts.max(axis=0).astype(int)
                lx1, ly1 = pts[LIP_POINTS].min(axis=0).astype(int)
                lx2, ly2 = pts[LIP_POINTS].max(axis=0).astype(int)
                detections.append((float(cx), float(cy), (int(x1), int(y1), int(x2), int(y2)), float(mar), (int(lx1), int(ly1), int(lx2), int(ly2))))
        used = set()
        for cx, cy, bbox, mar, mouth_box in detections:
            best = None
            best_idx = None
            best_dist = 1e9
            for i, tr in enumerate(self.tracks):
                if i in used:
                    continue
                d = np.hypot(cx - tr.cx, cy - tr.cy)
                if d < best_dist:
                    best_dist, best, best_idx = d, tr, i
            if best is not None and best_dist < 150:
                best.update(cx, cy, bbox, mar, mouth_box)
                used.add(best_idx)
            else:
                self.tracks.append(Track(self.next_id, cx, cy, bbox, mar, mouth_box))
                self.next_id += 1
                used.add(len(self.tracks) - 1)
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t.last_seen < 2.0]
        return self.tracks

    def active_speaker(self, window=3.0):
        if not self.tracks:
            return None
        scored = sorted([(t.movement_in_window(window), t.id) for t in self.tracks], reverse=True)
        return scored[0][1] if scored[0][0] > 0.05 else None

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


class AudioRecorder:
    def __init__(self, device=None, samplerate=16000):
        self.device=device; self.samplerate=samplerate
        self.buffer=np.zeros(0,dtype=np.float32); self.lock=threading.Lock(); self.stream=None
    def _cb(self, indata, frames, ti, status):
        mono = indata[:,0].astype(np.float32) if indata.ndim>1 else indata.astype(np.float32)
        with self.lock: self.buffer=np.concatenate([self.buffer, mono])
    def start(self):
        if sd is None: raise RuntimeError("sounddevice not installed")
        self.stream=sd.InputStream(samplerate=self.samplerate, channels=1,
            dtype='float32', device=self.device, callback=self._cb, blocksize=1600)
        self.stream.start()
    def stop(self):
        if self.stream: self.stream.stop(); self.stream.close()
    def take_if_enough(self, min_samples):
        with self.lock:
            if len(self.buffer) < min_samples: return None
            data=self.buffer.copy(); self.buffer=np.zeros(0,dtype=np.float32); return data

class LiveTranscriber(threading.Thread):
    def __init__(self, recorder, tracker, model_name, output_dir, session_start, chunk_seconds):
        super().__init__(daemon=True)
        self.recorder=recorder; self.tracker=tracker; self.chunk_seconds=chunk_seconds
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True)
        self.session_start=session_start; self.stop_flag=False
        self.segments=[]; self.last_lines=[]; self.lock=threading.Lock()
        print(f"Loading Whisper '{model_name}' - first run downloads it...")
        self.model=WhisperModel(model_name, device="cpu", compute_type="int8")
    def run(self):
        min_samples=int(16000*self.chunk_seconds)
        while not self.stop_flag:
            time.sleep(0.25)
            samples=self.recorder.take_if_enough(min_samples)
            if samples is None: continue
            chunk_start=time.time()-self.session_start
            rms=float(np.sqrt(np.mean(samples**2)))
            if rms < 0.0015: continue  # silence - low threshold helps whispers
            sid=self.tracker.active_speaker(self.chunk_seconds) if self.tracker else None
            label=f"Speaker {sid}" if sid is not None else "Unknown"
            try:
                segments,_=self.model.transcribe(samples, language="en", beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300, threshold=0.35),
                    condition_on_previous_text=False)
                for seg in segments:
                    text=seg.text.strip()
                    if not text: continue
                    start=chunk_start+seg.start
                    rec={"start":start,"end":chunk_start+seg.end,"speaker":label,"text":text}
                    with self.lock:
                        self.segments.append(rec)
                        self.last_lines=(self.last_lines+[f"[{fmt_time(start)}] {label}: {text}"])[-4:]
                        self.save_all()
                    print(f"[{fmt_time(start)}] {label}: {text}", flush=True)
            except Exception as e: print(f"Transcribe error: {e}", file=sys.stderr)
    def save_all(self):
        with open(self.output_dir/"transcript.txt","w") as f:
            for r in self.segments: f.write(f"[{fmt_time(r['start'])}] {r['speaker']}: {r['text']}\n")
        with open(self.output_dir/"transcript.json","w") as f: json.dump(self.segments,f,indent=2)
    def stop_and_save(self):
        self.stop_flag=True
        with self.lock:
            with open(self.output_dir/"transcript.srt","w") as f:
                for i,r in enumerate(self.segments,1):
                    f.write(f"{i}\n{srt_time(r['start'])} --> {srt_time(r['end'])}\n{r['speaker']}: {r['text']}\n\n")
            self.save_all()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source", choices=["screen","camera","video"], default="screen")
    p.add_argument("--video", default=None); p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--scale", type=float, default=0.6, help="screen capture scale for speed")
    p.add_argument("--model", default="base.en", help="tiny.en, base.en, small.en (best for whisper)")
    p.add_argument("--mic-device", default=None, help="index from --list-devices")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--output", default=None, help="Output folder. Default: folder containing main.py")
    p.add_argument("--chunk-seconds", type=float, default=3.0)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--save-lip-crops", action="store_true", help="save lip crops to train a true lip-reader later")
    args=p.parse_args()

    if args.list_devices:
        print(sd.query_devices() if sd else "sounddevice missing"); return

    if args.output:
        session_dir=Path(args.output)/datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        session_dir=Path(__file__).resolve().parent
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {session_dir}  - q to quit")

    tracker=LipTracker()
    recorder=None; transcriber=None; session_start=time.time()
    if not args.no_audio:
        if WhisperModel is None: print("faster-whisper missing, visual only"); args.no_audio=True
        else:
            dev=int(args.mic_device) if args.mic_device is not None else None
            recorder=AudioRecorder(device=dev); recorder.start()
            transcriber=LiveTranscriber(recorder,tracker,args.model,session_dir,session_start,args.chunk_seconds)
            transcriber.start()

    sct=None; cap=None; monitor=None
    if args.source=="screen":
        if mss is None: sys.exit("mss missing: pip install mss")
        sct=mss.mss(); monitor=sct.monitors[args.monitor]
    elif args.source=="camera": cap=cv2.VideoCapture(0)
    else: cap=cv2.VideoCapture(args.video)

    frame_idx=0
    while True:
        if args.source=="screen":
            frame=cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
            if args.scale!=1.0: frame=cv2.resize(frame,None,fx=args.scale,fy=args.scale)
        else:
            ret,frame=cap.read()
            if not ret: break
        frame_idx+=1
        tracks=tracker.process(frame)
        active=tracker.active_speaker(1.0)
        h,w=frame.shape[:2]
        for tr in tracks:
            x1,y1,x2,y2=tr.bbox; col=(0,255,255) if tr.id==active else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            cv2.putText(frame,f"Speaker {tr.id}",(x1,max(0,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
            if args.save_lip_crops and tr.mouth_box and frame_idx%10==0:
                lx1,ly1,lx2,ly2=tr.mouth_box; pad=20
                crop=frame[max(0,ly1-pad):min(h,ly2+pad), max(0,lx1-pad):min(w,lx2+pad)]
                if crop.size:
                    d=session_dir/"lip_crops"/f"speaker_{tr.id}"; d.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(d/f"{frame_idx:06d}.jpg"), cv2.resize(crop,(96,96)))
        # overlay transcript
        cv2.rectangle(frame,(0,h-110),(w,h),(0,0,0),-1)
        cv2.putText(frame,"LipReader - q=quit", (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        if transcriber:
            with transcriber.lock: lines=list(transcriber.last_lines)
            for i,line in enumerate(lines): cv2.putText(frame,line[:110],(10,h-80+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow("LipReader",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty("LipReader", cv2.WND_PROP_VISIBLE) < 1:
            break

    if transcriber: transcriber.stop_and_save()
    if recorder: recorder.stop()
    tracker.close()
    if cap: cap.release()
    cv2.destroyAllWindows(); print(f"Done. Files in {session_dir}")

if __name__=="__main__": main()