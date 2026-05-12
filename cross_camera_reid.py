"""
Cross-Camera Re-Identification Pipeline
========================================
YOLO (detection) + ByteTrack (tracking) + CLIP (feature extraction)
입력: 영상 파일 2개 (camera_a, camera_b)
출력: 바운딩박스 + Global ID 시각화 (나란히 표시)

설치:
    pip install ultralytics opencv-python torch torchvision
    pip install git+https://github.com/openai/CLIP.git
    pip install scipy numpy

실행:
    python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4
    python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4 --stride 4
"""

import argparse
import sys
import cv2
import numpy as np
import torch
import clip
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import cosine


# ── 설정 ──────────────────────────────────────────────────────────────────────

YOLO_MODEL      = "yolov8n.pt"      # nano: CPU도 충분
CLIP_MODEL      = "ViT-B/32"
PERSON_CLASS    = 0                  # YOLO COCO: 0 = person
MATCH_THRESHOLD = 0.35               # cosine distance 임계값 (낮을수록 엄격)
GALLERY_MAX_AGE = 300                # 몇 프레임 후 gallery에서 제거 (10fps 기준 ~30초)
DISPLAY_W       = 640                # 한 쪽 화면 너비
DISPLAY_H       = 480
FRAME_STRIDE    = 20                  # N프레임마다 1번만 YOLO+CLIP 실행


# ── 색상 팔레트 (Global ID별) ─────────────────────────────────────────────────

PALETTE = [
    (220,  80,  80), ( 80, 180,  80), ( 80, 120, 220),
    (200, 160,  50), (160,  80, 200), ( 50, 190, 190),
    (230, 120,  50), (100, 200, 150), (180,  80, 150),
    ( 80, 160, 230), (210, 210,  80), (130, 100, 230),
]

def get_color(gid: int):
    return PALETTE[gid % len(PALETTE)]


# ── CLIP feature extractor ────────────────────────────────────────────────────

class CLIPExtractor:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, bgr_crop: np.ndarray) -> np.ndarray:
        """BGR numpy crop → L2-normalized CLIP embedding (numpy 1D)"""
        if bgr_crop.size == 0:
            return None
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(rgb)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten()


# ── ByteTrack wrapper (ultralytics 내장) ─────────────────────────────────────

class Tracker:
    """ultralytics YOLO의 내장 ByteTrack을 프레임 단위로 사용"""
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def update(self, frame: np.ndarray):
        """
        Returns list of (track_id, x1, y1, x2, y2)
        person 클래스만 필터링
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            verbose=False,
        )
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                for tid, box in zip(boxes.id.int().tolist(),
                                    boxes.xyxy.int().tolist()):
                    x1, y1, x2, y2 = box
                    detections.append((tid, x1, y1, x2, y2))
        return detections


# ── Gallery (Camera A에서 추출한 임베딩 DB) ───────────────────────────────────

class Gallery:
    """
    Camera A의 local_id → CLIP embedding 저장
    Camera B 객체가 들어오면 가장 가까운 것과 매칭 → Global ID 부여
    """
    def __init__(self, threshold: float, max_age: int):
        self.threshold = threshold
        self.max_age   = max_age
        self.entries: dict = {}
        self._next_gid = 0

    def _new_gid(self):
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def update_cam_a(self, local_id: int, feat: np.ndarray, frame_idx: int):
        """Camera A 트랙 → gallery 등록/갱신"""
        if local_id not in self.entries:
            self.entries[local_id] = {
                'feat':      feat,
                'global_id': self._new_gid(),
                'last_seen': frame_idx,
                'cam':       'A',
            }
        else:
            old = self.entries[local_id]['feat']
            self.entries[local_id]['feat'] = 0.7 * old + 0.3 * feat
            self.entries[local_id]['feat'] /= np.linalg.norm(self.entries[local_id]['feat'])
            self.entries[local_id]['last_seen'] = frame_idx
        return self.entries[local_id]['global_id']

    def match_cam_b(self, local_id: int, feat: np.ndarray, frame_idx: int) -> int:
        b_key = f"B_{local_id}"
        if b_key in self.entries:
            self.entries[b_key]['last_seen'] = frame_idx
            return self.entries[b_key]['global_id']

        best_dist  = float('inf')
        best_entry = None
        for key, entry in self.entries.items():
            if entry['cam'] == 'A':
                dist = cosine(feat, entry['feat'])
                if dist < best_dist:
                    best_dist  = dist
                    best_entry = entry

        if best_entry is not None and best_dist < self.threshold:
            gid = best_entry['global_id']
        else:
            gid = self._new_gid()

        self.entries[b_key] = {
            'feat':      feat,
            'global_id': gid,
            'last_seen': frame_idx,
            'cam':       'B',
        }
        return gid

    def expire(self, frame_idx: int):
        to_del = [k for k, v in self.entries.items()
                  if frame_idx - v['last_seen'] > self.max_age]
        for k in to_del:
            del self.entries[k]

    def get_gid(self, local_id: int, cam: str):
        """
        CLIP을 돌리지 않는 skip 프레임에서
        ByteTrack local id로 기존 global id를 재사용
        """
        key = local_id if cam == "A" else f"B_{local_id}"
        entry = self.entries.get(key)

        if entry is None:
            return None

        return entry["global_id"]

# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, gid, local_id, cam_label):
    color = get_color(gid)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"[{cam_label}] LID:{local_id} GID:{gid}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(frame, label, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_cached(frame, cached_dets, cam_label, orig_w, orig_h):
    """stride로 건너뛴 프레임에 마지막 결과 재사용해서 그리기"""
    sx = DISPLAY_W / orig_w
    sy = DISPLAY_H / orig_h
    for (lid, x1, y1, x2, y2, gid) in cached_dets:
        draw_box(frame,
                 int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy),
                 gid, lid, cam_label)


def resize_frame(frame, w, h):
    return cv2.resize(frame, (w, h))


def make_display(frame_a, frame_b, frame_idx, stride, is_inference_frame, gid_map_a, gid_map_b):
    fa = resize_frame(frame_a, DISPLAY_W, DISPLAY_H)
    fb = resize_frame(frame_b, DISPLAY_W, DISPLAY_H)

    cv2.putText(fa, "Camera A", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(fb, "Camera B", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (240, 240, 240), 2, cv2.LINE_AA)

    # 왼쪽 하단: 프레임 번호 + stride 표시
    infer_tag = "INFER" if is_inference_frame else f"skip({stride})"
    cv2.putText(fa, f"frame {frame_idx}  [{infer_tag}]", (10, DISPLAY_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    matched = set(gid_map_a.values()) & set(gid_map_b.values())
    info = f"Matched IDs: {len(matched)}"
    cv2.putText(fb, info, (10, DISPLAY_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)

    divider = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)
    return np.hstack([fa, divider, fb])


# ── 디렉토리 유틸 ─────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

def collect_videos(directory: str) -> list:
    d = Path(directory)
    if not d.is_dir():
        sys.exit(f"[ERROR] 디렉토리가 없습니다: {directory}")
    videos = sorted(p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"[ERROR] 영상 파일이 없습니다: {directory}")
    return videos


def make_pairs(dir_a: str, dir_b: str):
    vids_a = collect_videos(dir_a)
    vids_b = collect_videos(dir_b)
    n = min(len(vids_a), len(vids_b))
    if len(vids_a) != len(vids_b):
        print(f"[WARN] 파일 수 불일치 (A:{len(vids_a)}, B:{len(vids_b)}) → 앞 {n}쌍만 처리")
    pairs = list(zip(vids_a[:n], vids_b[:n]))
    print(f"[INFO] 총 {n}쌍 처리 예정:")
    for i, (a, b) in enumerate(pairs, 1):
        print(f"  [{i:02d}] {a.name}  ←→  {b.name}")
    return pairs


# ── 메인 루프 ─────────────────────────────────────────────────────────────────

def run(video_a: str, video_b: str, save_path: str = None, stride: int = FRAME_STRIDE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")
    print(f"[INFO] Stride : {stride}  (1/{stride} 프레임만 YOLO+CLIP 실행)")

    print("[INFO] Loading YOLO + ByteTrack ...")
    tracker_a = Tracker(YOLO_MODEL)
    tracker_b = Tracker(YOLO_MODEL)

    print("[INFO] Loading CLIP ...")
    extractor = CLIPExtractor(CLIP_MODEL, device)

    gallery = Gallery(threshold=MATCH_THRESHOLD, max_age=GALLERY_MAX_AGE)

    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)

    if not cap_a.isOpened():
        sys.exit(f"[ERROR] Cannot open: {video_a}")
    if not cap_b.isOpened():
        sys.exit(f"[ERROR] Cannot open: {video_b}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, 20.0,
                                 (DISPLAY_W * 2 + 4, DISPLAY_H))
        print(f"[INFO] Saving to : {save_path}")

    frame_idx  = 0

    # stride 동안 재사용할 마지막 결과 캐시
    # 형식: list of (lid, x1, y1, x2, y2, gid)
    cache_a: list = []
    cache_b: list = []
    gid_map_a: dict = {}
    gid_map_b: dict = {}

    orig_h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("[INFO] Starting — press Q to quit")

    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        if not ret_a or not ret_b:
            print("[INFO] Video ended.")
            break

        vis_a = resize_frame(frame_a, DISPLAY_W, DISPLAY_H)
        vis_b = resize_frame(frame_b, DISPLAY_W, DISPLAY_H)

        #is_infer = (frame_idx % stride == 0)


        is_infer = (frame_idx % stride == 0)

        cache_a.clear()
        cache_b.clear()
        gid_map_a.clear()
        gid_map_b.clear()

        # ── Camera A: YOLO는 매 프레임 실행 ─────────────────────────
        dets_a = tracker_a.update(frame_a)

        for (lid, x1, y1, x2, y2) in dets_a:
            if is_infer:
                # stride 프레임: CLIP 실행
                crop = frame_a[max(y1, 0):y2, max(x1, 0):x2]
                feat = extractor.extract(crop)

                if feat is None:
                    continue

                gid = gallery.update_cam_a(lid, feat, frame_idx)
            else:
                # skip 프레임: CLIP 실행 안 함, 기존 GID만 재사용
                gid = gallery.get_gid(lid, "A")

                if gid is None:
                    continue

            gid_map_a[lid] = gid
            cache_a.append((lid, x1, y1, x2, y2, gid))


        # ── Camera B: YOLO는 매 프레임 실행 ─────────────────────────
        dets_b = tracker_b.update(frame_b)

        for (lid, x1, y1, x2, y2) in dets_b:
            if is_infer:
                # stride 프레임: CLIP 실행
                crop = frame_b[max(y1, 0):y2, max(x1, 0):x2]
                feat = extractor.extract(crop)

                if feat is None:
                    continue

                gid = gallery.match_cam_b(lid, feat, frame_idx)
            else:
                # skip 프레임: CLIP 실행 안 함, 기존 GID만 재사용
                gid = gallery.get_gid(lid, "B")

                if gid is None:
                    continue

            gid_map_b[lid] = gid
            cache_b.append((lid, x1, y1, x2, y2, gid))


        if is_infer:
            gallery.expire(frame_idx)

        # ── 캐시된 박스 그리기 ─────────────────────────────────────
        draw_cached(vis_a, cache_a, "A", orig_w_a, orig_h_a)
        draw_cached(vis_b, cache_b, "B", orig_w_b, orig_h_b)

        display = make_display(
            vis_a, vis_b,
            frame_idx,
            stride,
            is_infer,
            gid_map_a,
            gid_map_b
        )

        cv2.imshow("Cross-Camera ReID  |  Q: quit", display)
        if writer:
            writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        paused = False
        
        # q: 종료
        if key == ord('q'):
            print("[INFO] User quit.")
            break

        # p 또는 space: 일시정지 / 재개
        elif key == ord('p') or key == 32:
            paused = not paused
            print("[INFO] Paused" if paused else "[INFO] Resumed")

        # paused 상태면 여기서 멈춤
        while paused:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("[INFO] User quit.")
                cap_a.release()
                cap_b.release()
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                return

            elif key == ord('p') or key == 32:
                paused = False
                print("[INFO] Resumed")

        frame_idx += 1

    cap_a.release()
    cap_b.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frames.")


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Camera ReID Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 단일 파일 모드
    parser.add_argument("--video_a",  default="/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0002.mp4",
                        help="Camera A 영상 경로")
    parser.add_argument("--video_b",  default="/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0003.mp4",
                        help="Camera B 영상 경로")
    parser.add_argument("--save",     default="/home/knuvi/Multi-scene-detection/results",
                        help="결과 영상 저장 경로 (옵션)")

    # 디렉토리 모드
    parser.add_argument("--dir_a",    default=None, help="Camera A 영상 폴더")
    parser.add_argument("--dir_b",    default=None, help="Camera B 영상 폴더")
    parser.add_argument("--save_dir", default=None, help="결과 저장 폴더 (디렉토리 모드)")

    # 공통
    parser.add_argument("--stride",    type=int,   default=FRAME_STRIDE,
                        help=f"N프레임마다 1번 추론 (기본: {FRAME_STRIDE})\n"
                             f"  stride=1: 매 프레임 추론 (느리지만 정확)\n"
                             f"  stride=4: 4프레임당 1회 추론 (권장)\n"
                             f"  stride=8: 8프레임당 1회 추론 (빠르지만 부정확)")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                        help=f"매칭 cosine distance 임계값 (기본: {MATCH_THRESHOLD})")

    args = parser.parse_args()
    MATCH_THRESHOLD = args.threshold
    FRAME_STRIDE    = args.stride

    use_dir  = args.dir_a is not None or args.dir_b is not None
    use_file = args.video_a is not None

    if use_dir:
        if not args.dir_a or not args.dir_b:
            sys.exit("[ERROR] --dir_a 와 --dir_b 를 모두 지정해야 합니다.")
        pairs = make_pairs(args.dir_a, args.dir_b)
        save_dir = Path(args.save_dir) if args.save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        for idx, (path_a, path_b) in enumerate(pairs, 1):
            print(f"\n{'='*60}")
            print(f"[{idx:02d}/{len(pairs)}] {path_a.name}  ←→  {path_b.name}")
            print(f"{'='*60}")
            save_path = str(save_dir / f"result_{idx:02d}_{path_a.stem}_vs_{path_b.stem}.mp4") \
                        if save_dir else None
            run(str(path_a), str(path_b), save_path=save_path, stride=args.stride)
        print(f"\n[INFO] 전체 {len(pairs)}쌍 처리 완료.")
    else:
        run(args.video_a, args.video_b, save_path=args.save, stride=args.stride)