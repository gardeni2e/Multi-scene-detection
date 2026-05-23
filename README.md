# Multi-Camera Pedestrian Re-Identification

본 프로젝트는 서로 다른 두 CCTV 영상에서 등장하는 보행자를 탐지·추적하고, 각 보행자 track의 appearance feature를 기반으로 동일 인물 후보를 연결하는 **다중 카메라 보행자 Re-Identification 시스템**입니다.

기본 입력은 두 개의 영상 `camA.mp4`, `camB.mp4`이며, 각 영상에서 사람을 검출하고 local tracking을 수행한 뒤, 서로 다른 카메라에 등장한 동일 인물에게 같은 Global ID를 부여하는 것을 목표로 합니다.

---

## 1. 프로젝트 목표

일반적인 CCTV 환경에서는 사람이 한 카메라의 시야에서 사라진 뒤 다른 카메라에 다시 등장할 수 있습니다. 이때 각 카메라의 local tracking 결과만으로는 같은 사람인지 알기 어렵습니다.

본 프로젝트는 다음과 같은 흐름을 통해 이를 해결하고자 합니다.

```text
Cam A 영상 / Cam B 영상 입력
→ YOLO11로 사람 detection
→ BoT-SORT로 카메라 내부 local tracking
→ OSNet으로 person crop feature 추출
→ tracklet-level feature aggregation
→ Cam A / Cam B track 간 similarity 계산
→ evidence 기반 Global ID matching
→ 결과 영상 저장
```

즉, 단일 이미지 한 장을 비교하는 방식이 아니라, 각 local track에 포함된 여러 crop feature를 누적하여 tracklet-level representation을 구성하고, 이를 기반으로 cross-camera Re-ID를 수행합니다.

---

## 2. 전체 코드 구조 설명

현재 최종 버전은 `v10 stable` 버전입니다.

v10은 이전 실험에서 가장 안정적이었던 `v7 archive` 구조를 기반으로 하며, local ID를 강제로 병합하는 방식은 제거하고, Global ID archive를 통해 끊긴 track이 기존 GID를 이어받을 수 있도록 구성했습니다.

전체 구성 요소는 다음과 같습니다.

---

### 2.1 YOLO11: Person Detection

각 프레임에서 사람 bounding box를 검출합니다.

```text
입력: 영상 프레임
출력: person bbox, confidence
```

v10에서는 YOLO11 계열 모델을 사용합니다.

추천 기본값은 다음과 같습니다.

```bash
--yolo_model yolo11m.pt
--imgsz 1280
--yolo_conf 0.18
--yolo_iou 0.80
--min_det_area 2200
```

`min_det_area`는 너무 멀리 있는 작은 사람까지 detection되는 문제를 줄이기 위해 사용합니다. bbox 면적이 이 값보다 작으면 시각화 및 Re-ID 대상에서 제외됩니다.

---

### 2.2 BoT-SORT: Local Tracking

YOLO가 매 프레임 검출한 bbox들을 시간 순서대로 연결하여 같은 카메라 내부에서 Local ID를 부여합니다.

```text
YOLO detection 결과
→ BoT-SORT tracking
→ Local ID 생성
```

실행 시 다음 옵션을 사용합니다.

```bash
--tracker botsort.yaml
```

BoT-SORT는 ByteTrack과 마찬가지로 tracking 알고리즘이며, YOLO 모델 자체가 아닙니다. YOLO가 검출한 bbox들을 프레임 간 연결하여 같은 카메라 안에서 동일 인물의 Local ID를 유지하는 역할을 합니다.

---

### 2.3 OSNet: Re-ID Feature Extraction

각 사람 crop 이미지를 OSNet에 입력하여 Re-ID feature vector를 추출합니다.

```text
person crop
→ OSNet
→ appearance feature vector
```

OSNet은 보행자 Re-Identification에 특화된 모델로, 사람의 옷 색, 실루엣, 가방, 신발 등의 appearance 정보를 feature로 표현합니다.

초기 구현에서는 CLIP feature를 사용했지만, CLIP은 범용 이미지 표현 모델이므로 사람 간 세밀한 외형 차이를 구분하는 Re-ID task에는 한계가 있었습니다. 따라서 v10에서는 OSNet을 사용합니다.

---

### 2.4 Feature Buffer

각 local track마다 여러 프레임에서 추출한 OSNet feature를 buffer에 저장합니다.

```text
Local Track 3
→ feature_1
→ feature_2
→ feature_3
→ ...
```

feature는 매 프레임 추출하지 않고, 일정 frame stride마다 추출합니다.

추천값:

```bash
--stride 10
```

즉, YOLO/BoT-SORT는 매 프레임 수행하지만, OSNet feature extraction은 10프레임마다 수행합니다.

각 local track의 feature buffer는 해당 사람의 tracklet-level representation을 만들기 위해 사용됩니다.

---

### 2.5 Mean Feature Similarity + Top-k Feature Set Similarity

v10에서는 두 track을 비교할 때 하나의 평균 feature만 사용하지 않습니다.

두 가지 similarity를 함께 사용합니다.

#### 1. Mean Feature Similarity

각 local track의 feature buffer를 평균내어 대표 feature를 만듭니다.

```text
A_mean = 평균(A track의 feature들)
B_mean = 평균(B track의 feature들)
```

그리고 두 평균 feature의 cosine similarity를 계산합니다.

이는 두 tracklet의 전체적인 appearance가 비슷한지 보는 기준입니다.

#### 2. Feature Set Top-k Similarity

tracklet 내부의 개별 feature들끼리도 비교합니다.

예를 들어:

```text
Cam A Track = [A1, A2, A3]
Cam B Track = [B1, B2, B3]
```

이면 모든 pair를 비교합니다.

```text
A1-B1, A1-B2, A1-B3
A2-B1, A2-B2, A2-B3
A3-B1, A3-B2, A3-B3
```

그중 similarity가 높은 상위 k개를 평균내어 top-k similarity로 사용합니다.

추천값:

```bash
--topk 5
--mean_weight 0.4
--set_weight 0.6
```

즉, 최종 similarity는 다음과 같이 계산됩니다.

```text
final similarity
= mean_weight × mean feature similarity
+ set_weight × top-k feature set similarity
```

이 방식은 평균 feature 하나만 비교할 때보다, 여러 프레임 중 일부라도 비슷한 모습이 존재할 경우 이를 matching에 반영할 수 있다는 장점이 있습니다.

---

### 2.6 Evidence 기반 Global ID Matching

v10에서는 한 번의 similarity 결과로 바로 GID를 확정하지 않습니다.

각 후보 GID에 대해 evidence score를 누적합니다.

```text
Cam B LID 5에 대한 후보 evidence:

GID 1: 0.4
GID 2: 2.1
GID 3: 0.7
```

가장 evidence가 높은 GID를 현재 best GID로 사용합니다.

관련 파라미터:

```bash
--threshold 0.55
--min_evidence 0.7
--evidence_margin 0.15
--distance_margin 0.01
--switch_margin 0.40
--evidence_decay 0.90
```

오래된 evidence가 계속 남아 잘못된 GID에 고정되는 것을 막기 위해 evidence decay를 적용합니다.

```text
old evidence × evidence_decay + new evidence
```

이를 통해 초기 오판이 있더라도, 이후 더 강한 evidence가 쌓이면 올바른 GID로 변경될 수 있습니다.

---

### 2.7 Global ID Archive

v10은 각 GID의 appearance memory를 archive 형태로 유지합니다.

사람이 detection miss, occlusion 등으로 잠깐 사라져 local track이 끊기더라도, 이전 GID의 feature memory는 일정 시간 유지됩니다.

이후 새 local track이 등장하면 바로 새 GID를 부여하지 않고, 기존 GID archive와 먼저 비교합니다.

```text
기존:
LID 4 / GID 2
→ detection miss
→ 새 LID 9
→ 새 GID 생성 가능

v10:
LID 4 / GID 2
→ detection miss
→ GID 2 archive 유지
→ 새 LID 9 등장
→ GID archive와 비교
→ 유사하면 GID 2 이어받음
```

관련 파라미터:

```bash
--memory_max_age 1200
--reassoc_margin 0.01
```

---

### 2.8 Occlusion-aware Update

사람끼리 겹치는 상황에서는 crop 안에 다른 사람의 정보가 섞여 Re-ID feature가 오염될 수 있습니다.

이를 줄이기 위해 같은 프레임 내 person bbox 간 IoU가 일정 값 이상이면, 해당 crop은 feature/evidence update에서 제외합니다.

```bash
--occ_iou 0.25
```

또한 프레임 가장자리에 걸린 crop은 사람이 잘렸을 가능성이 있으므로 feature update에서 제외합니다.

```bash
--edge_margin 0.02
```

---

## 3. Colab 실행 방법

Colab에서는 런타임이 끊기면 `/content` 내부 파일이 사라집니다.  
따라서 영상 파일은 Colab에 매번 직접 업로드하지 않고, Google Drive에 한 번 올려둔 뒤 실행할 때 `/content`로 복사해서 사용하는 방식을 권장합니다.

---

## 4. Google Drive 폴더 구조

Google Drive에 아래와 같은 폴더 구조를 만들어 주세요.

```text
내 드라이브/
└── dl_project/
    ├── videos/
    │   ├── camA.mp4
    │   └── camB.mp4
    └── outputs/
```

영상 파일 이름은 반드시 다음과 같이 맞춰 주세요.

```text
camA.mp4
camB.mp4
```

각 파일 위치:

```text
/content/drive/MyDrive/dl_project/videos/camA.mp4
/content/drive/MyDrive/dl_project/videos/camB.mp4
```

결과 영상은 다음 위치에 저장됩니다.

```text
/content/drive/MyDrive/dl_project/outputs/
```

---

## 5. Colab 실행 순서

### 5.1 Google Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 5.2 레포지토리 클론

```bash
%cd /content
!rm -rf Multi-scene-detection
!git clone https://github.com/Jehee-Kim/Multi-scene-detection.git
%cd /content/Multi-scene-detection
```

---

### 5.3 패키지 설치

```bash
!pip install -U ultralytics lap
!pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
!pip install huggingface_hub
```

---

### 5.4 Drive의 영상 파일을 `/content`로 복사

Drive에서 바로 영상을 읽을 수도 있지만, 실행 속도와 안정성을 위해 `/content`로 복사해서 사용하는 것을 권장합니다.

```bash
!cp "/content/drive/MyDrive/dl_project/videos/camA.mp4" /content/camA.mp4
!cp "/content/drive/MyDrive/dl_project/videos/camB.mp4" /content/camB.mp4
```

---

### 5.5 코드 실행

아래는 현재 추천 실행 조합입니다.

```bash
%cd /content/Multi-scene-detection

!python cross_camera_reid.py \
  --video_a /content/camA.mp4 \
  --video_b /content/camB.mp4 \
  --save "/content/drive/MyDrive/dl_project/outputs/output_osnet_v10_area2200_reid_loose.mp4" \
  --yolo_model yolo11m.pt \
  --tracker botsort.yaml \
  --imgsz 1280 \
  --yolo_conf 0.18 \
  --yolo_iou 0.80 \
  --min_det_area 2200 \
  --stride 10 \
  --threshold 0.55 \
  --confirm 2 \
  --topk 5 \
  --mean_weight 0.4 \
  --set_weight 0.6 \
  --evidence_decay 0.90 \
  --min_evidence 0.7 \
  --evidence_margin 0.15 \
  --switch_margin 0.40 \
  --distance_margin 0.01 \
  --occ_iou 0.25 \
  --edge_margin 0.02 \
  --memory_max_age 1200 \
  --reassoc_margin 0.01
```

실행이 완료되면 결과 영상은 Google Drive에 저장됩니다.

```text
/content/drive/MyDrive/dl_project/outputs/output_osnet_v10_area2200_reid_loose.mp4
```

---

## 6. 추천 실행 조합 설명

현재 추천 조합은 다음과 같은 의도로 설정되어 있습니다.

### Detection / Tracking

```bash
--yolo_model yolo11m.pt
--tracker botsort.yaml
--imgsz 1280
--yolo_conf 0.18
--yolo_iou 0.80
--min_det_area 2200
```

- `yolo11m.pt`: 속도와 정확도 균형을 고려한 YOLO11 모델
- `botsort.yaml`: occlusion 상황에서 local tracking 안정성을 높이기 위한 tracker
- `imgsz 1280`: 작은 사람이나 부분 가림 상황에서 detection 성능 향상
- `yolo_conf 0.18`: 낮은 confidence의 사람 후보도 어느 정도 유지
- `yolo_iou 0.80`: 겹친 bbox가 NMS로 과도하게 제거되는 것을 완화
- `min_det_area 2200`: 너무 멀리 있는 작은 사람 detection을 줄임

### Re-ID / Matching

```bash
--stride 10
--threshold 0.55
--topk 5
--mean_weight 0.4
--set_weight 0.6
```

- `stride 10`: 10프레임마다 OSNet feature 추출
- `threshold 0.55`: 같은 사람으로 인정하는 범위를 기본보다 약간 완화
- `topk 5`: feature set 간 상위 5개 pair similarity 사용
- `mean_weight 0.4`, `set_weight 0.6`: 평균 feature보다 feature set top-k similarity를 조금 더 반영

### Evidence / Archive

```bash
--min_evidence 0.7
--evidence_margin 0.15
--distance_margin 0.01
--memory_max_age 1200
--reassoc_margin 0.01
```

- `min_evidence 0.7`: GID가 조금 더 빠르게 붙도록 완화
- `evidence_margin 0.15`: 1등 후보가 2등보다 너무 압도적이지 않아도 선택 가능
- `distance_margin 0.01`: best/second-best 차이가 작아도 evidence를 쌓을 수 있음
- `memory_max_age 1200`: 사라진 GID를 더 오래 기억
- `reassoc_margin 0.01`: 끊겼다가 다시 등장한 track이 기존 GID를 더 쉽게 이어받도록 설정

---

## 7. 파라미터 조정 가이드

### 7.1 너무 멀리 있는 작은 사람이 많이 잡힐 때

```bash
--min_det_area 3000
```

또는

```bash
--yolo_conf 0.20
```

---

### 7.2 사람이 잘 detection되지 않을 때

```bash
--min_det_area 1200
--yolo_conf 0.15
```

더 강하게는:

```bash
--yolo_model yolo11l.pt
```

---

### 7.3 같은 사람인데 GID가 잘 안 붙을 때

```bash
--threshold 0.58
--min_evidence 0.5
--evidence_margin 0.10
--distance_margin 0.00
--reassoc_margin 0.00
```

단, 오매칭 가능성이 증가할 수 있습니다.

---

### 7.4 다른 사람을 같은 GID로 잘못 붙일 때

```bash
--threshold 0.50
--min_evidence 1.00
--evidence_margin 0.25
--distance_margin 0.03
```

---

### 7.5 GID가 너무 자주 바뀔 때

```bash
--switch_margin 0.60
--evidence_decay 0.95
```

---

## 8. 결과 출력

결과 영상에는 각 보행자의 bbox와 함께 Local ID, Global ID가 표시됩니다.

```text
LID:3 GID:2
```

- `LID`: 같은 카메라 내부 tracking ID
- `GID`: 카메라 간 동일 인물로 연결된 Global ID

Colab에서는 `cv2.imshow()`를 사용할 수 없기 때문에 실시간 창을 띄우지 않고, 결과 영상을 `--save` 경로에 저장합니다.

---

## 9. 현재 한계

본 프로젝트는 제한된 두 카메라 환경에서 tracklet-level Re-ID 가능성을 확인하는 시스템형 프로젝트입니다. 다음과 같은 한계가 존재합니다.

- 사람이 완전히 가려지는 경우 detection 자체가 불가능할 수 있음
- 비슷한 옷차림의 사람이 많으면 오매칭 가능
- crop 품질, blur, occlusion에 따라 Re-ID feature가 흔들릴 수 있음
- 카메라 구도 차이가 클수록 같은 사람 feature도 멀어질 수 있음
- 완전한 광역 CCTV 추적 시스템은 아님

---

## 10. 최종 요약

본 프로젝트는 YOLO11과 BoT-SORT를 이용해 각 카메라 내부의 보행자 local track을 생성하고, OSNet을 이용해 각 track의 appearance feature를 추출합니다. 이후 mean feature similarity와 feature set top-k similarity를 함께 사용하여 tracklet-level Re-ID matching을 수행하며, evidence score와 Global ID archive를 통해 초기 오판이나 detection miss 이후에도 Global ID를 안정적으로 유지하도록 설계했습니다.
