# Cross-Camera ReID — 실행 가이드

## 한 줄 요약
YOLO(탐지) → ByteTrack(단일 카메라 추적) → CLIP(임베딩 추출) → Cosine 매칭(카메라 간 ReID)
**GPU 없이 CPU만으로 동작**

---

## 설치

```bash
pip install ultralytics opencv-python torch torchvision scipy numpy Pillow
pip install git+https://github.com/openai/CLIP.git
```

> 처음 실행 시 YOLOv8n (6MB), CLIP ViT-B/32 (~340MB) 자동 다운로드

---

## 실행

```bash
# 기본 실행
python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4

# 결과 영상 저장
python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4 --save output.mp4

# 매칭 민감도 조정 (기본 0.35 / 낮출수록 엄격)
python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4 --threshold 0.30
```

---

## 화면 설명

```
┌─────────────────────┬─┬─────────────────────┐
│    Camera A         │ │    Camera B          │
│  [A] LID:1 GID:0   │ │  [B] LID:3 GID:0    │  ← 같은 GID = 동일 인물
│  [A] LID:2 GID:1   │ │                      │
│  frame 142          │ │  Matched IDs: 1      │
└─────────────────────┴─┴─────────────────────┘
```

- `LID` = 해당 카메라 내 로컬 트랙 ID (ByteTrack 부여)
- `GID` = 카메라를 넘나드는 전역 ID (CLIP 매칭 결과)
- 같은 색 바운딩박스 = 동일 인물로 판단
- `Q` 키로 종료

---

## 파라미터 튜닝 가이드

| 상황 | 조치 |
|------|------|
| 오매칭(다른 사람을 같다고 함)이 많다 | `--threshold` 낮추기 (예: 0.25) |
| 같은 사람인데 못 찾는다 | `--threshold` 높이기 (예: 0.45) |
| 카메라 전환 간격이 길다 | 코드 내 `GALLERY_MAX_AGE` 늘리기 |
| CPU가 너무 느리다 | YOLO 처리를 N프레임마다 1번으로 줄이기 |

---

## 구조 설명

```
cross_camera_reid.py
├── CLIPExtractor     CLIP ViT-B/32으로 person crop → 512-dim 임베딩
├── Tracker           YOLOv8n + ByteTrack, 카메라별 독립 인스턴스
├── Gallery           Camera A 임베딩 DB + Camera B 매칭 로직
│   ├── update_cam_a  Camera A 트랙 등록 / 지수이동평균 업데이트
│   └── match_cam_b   Cosine distance로 가장 유사한 Camera A 트랙 검색
└── run()             메인 루프: 두 영상 동시 처리 → 시각화
```

---

## 직접 촬영 팁

스마트폰 2대로 촬영해서 테스트할 경우:
- 조명 비슷하게 맞추기 (실외 or 형광등 환경 통일)
- 사람이 카메라 A에서 나가고 5~10초 내 카메라 B에 등장하게 하면 매칭률이 높음
- 해상도 너무 낮으면 CLIP 임베딩 품질 하락 → 720p 이상 권장

---

## 한계 및 개선 방향

- 현재는 외관(appearance)만 사용 → 같은 옷 입은 다른 사람은 오매칭 가능
- 개선: 카메라 전환 시간 분포 + 진입 방향 정보 추가
- 더 강한 성능 필요하면: OSNet (torchreid) feature extractor로 교체
