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
"""

import argparse
import sys
import cv2
import numpy as np
import torch
import clip
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from collections import defaultdict


# ── 설정 ──────────────────────────────────────────────────────────────────────

YOLO_MODEL      = "yolov8n.pt"      # nano: CPU도 충분
CLIP_MODEL      = "ViT-B/32"
PERSON_CLASS    = 0                  # YOLO COCO: 0 = person
MATCH_THRESHOLD = 0.35               # cosine distance 임계값 (낮을수록 엄격)
GALLERY_MAX_AGE = 300                # 몇 프레임 후 gallery에서 제거 (10fps 기준 ~30초)
DISPLAY_W       = 640                # 한 쪽 화면 너비
DISPLAY_H       = 480


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
        # local_id → {'feat': np.ndarray, 'global_id': int, 'age': int}
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
            # 지수이동평균으로 특징 업데이트 (외관 변화 반영)
            old = self.entries[local_id]['feat']
            self.entries[local_id]['feat'] = 0.7 * old + 0.3 * feat
            self.entries[local_id]['feat'] /= np.linalg.norm(self.entries[local_id]['feat'])
            self.entries[local_id]['last_seen'] = frame_idx
        return self.entries[local_id]['global_id']

    def match_cam_b(self, local_id: int, feat: np.ndarray, frame_idx: int) -> int:
        """
        Camera B 트랙 feat → Camera A gallery 중 가장 유사한 항목 매칭
        임계값 내 매칭 없으면 새 Global ID 발급
        """
        # Camera B 자체 트랙도 gallery에 있으면 바로 반환
        b_key = f"B_{local_id}"
        if b_key in self.entries:
            self.entries[b_key]['last_seen'] = frame_idx
            return self.entries[b_key]['global_id']

        # Camera A entries 중 유사도 검색
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
        """오래된 항목 제거"""
        to_del = [k for k, v in self.entries.items()
                  if frame_idx - v['last_seen'] > self.max_age]
        for k in to_del:
            del self.entries[k]


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


def resize_frame(frame, w, h):
    return cv2.resize(frame, (w, h))


def make_display(frame_a, frame_b, frame_idx, gid_map_a, gid_map_b):
    """두 프레임을 가로로 이어붙여 반환"""
    fa = resize_frame(frame_a, DISPLAY_W, DISPLAY_H)
    fb = resize_frame(frame_b, DISPLAY_W, DISPLAY_H)

    # 구분선 + 헤더
    cv2.putText(fa, "Camera A", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(fb, "Camera B", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(fa, f"frame {frame_idx}", (10, DISPLAY_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # 매칭 개수 표시
    matched = set(gid_map_a.values()) & set(gid_map_b.values())
    info = f"Matched IDs: {len(matched)}"
    cv2.putText(fb, info, (10, DISPLAY_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)

    divider = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)
    return np.hstack([fa, divider, fb])


# ── 메인 루프 ─────────────────────────────────────────────────────────────────

def run(video_a: str, video_b: str, save_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

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
        print(f"[INFO] Saving output to: {save_path}")

    frame_idx = 0
    print("[INFO] Starting — press Q to quit")

    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        # 한쪽이 끝나면 종료
        if not ret_a or not ret_b:
            print("[INFO] Video ended.")
            break

        vis_a = frame_a.copy()
        vis_b = frame_b.copy()

        gid_map_a: dict[int, int] = {}
        gid_map_b: dict[int, int] = {}

        # ── Camera A 처리 ──────────────────────────────────────────────
        dets_a = tracker_a.update(frame_a)
        for (lid, x1, y1, x2, y2) in dets_a:
            crop = frame_a[max(y1,0):y2, max(x1,0):x2]
            feat = extractor.extract(crop)
            if feat is None:
                continue
            gid = gallery.update_cam_a(lid, feat, frame_idx)
            gid_map_a[lid] = gid
            # 시각화용 좌표 스케일 조정
            sx = DISPLAY_W / frame_a.shape[1]
            sy = DISPLAY_H / frame_a.shape[0]
            draw_box(vis_a,
                     int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy),
                     gid, lid, "A")

        # ── Camera B 처리 ──────────────────────────────────────────────
        dets_b = tracker_b.update(frame_b)
        for (lid, x1, y1, x2, y2) in dets_b:
            crop = frame_b[max(y1,0):y2, max(x1,0):x2]
            feat = extractor.extract(crop)
            if feat is None:
                continue
            gid = gallery.match_cam_b(lid, feat, frame_idx)
            gid_map_b[lid] = gid
            sx = DISPLAY_W / frame_b.shape[1]
            sy = DISPLAY_H / frame_b.shape[0]
            draw_box(vis_b,
                     int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy),
                     gid, lid, "B")

        # ── gallery 정리 & 표시 ────────────────────────────────────────
        gallery.expire(frame_idx)
        display = make_display(vis_a, vis_b, frame_idx, gid_map_a, gid_map_b)

        cv2.imshow("Cross-Camera ReID  |  Q: quit", display)
        if writer:
            writer.write(display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User quit.")
            break

        frame_idx += 1

    cap_a.release()
    cap_b.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frames.")


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Camera ReID Pipeline")
    parser.add_argument("--video_a",  required=True,  help="Camera A 영상 경로")
    parser.add_argument("--video_b",  required=True,  help="Camera B 영상 경로")
    parser.add_argument("--save",     default=None,   help="결과 영상 저장 경로 (옵션)")
    parser.add_argument("--threshold",type=float, default=MATCH_THRESHOLD,
                        help=f"매칭 임계값 (기본: {MATCH_THRESHOLD})")
    args = parser.parse_args()

    MATCH_THRESHOLD = args.threshold
    run(args.video_a, args.video_b, save_path=args.save)
