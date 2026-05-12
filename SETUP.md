# 콘다 환경 세팅 가이드

## 0. 사전 확인

```bash
conda --version   # 없으면 https://docs.conda.io/en/latest/miniconda.html
python --version  # 확인용 (conda 내부 python이 사용됨)
```

---

## 1. 환경 생성 (방법 A — environment.yml 한 방에 설치, 권장)

```bash
conda env create -f environment.yml
conda activate reid
pip install git+https://github.com/openai/CLIP.git
```

> CLIP은 PyPI에 없어서 git으로 따로 설치해야 합니다.

---

## 2. 환경 생성 (방법 B — 직접 하나씩 설치)

```bash
# 환경 생성
conda create -n reid python=3.10 -y
conda activate reid

# PyTorch 설치 — CPU 전용
conda install pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 cpuonly -c pytorch -y

# 나머지 패키지
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

---

## 3. GPU가 있는 경우 (CUDA 11.8 기준)

```bash
conda create -n reid python=3.10 -y
conda activate reid

# cpuonly 대신 cuda 버전으로 교체
conda install pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 \
    pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

> CUDA 12.1이면 `pytorch-cuda=12.1` 로 바꾸면 됩니다.
> 내 CUDA 버전 확인: `nvidia-smi`

---

## 4. 설치 확인

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import clip; print('CLIP OK')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

모두 에러 없이 출력되면 완료입니다.

---

## 5. 실행

```bash
conda activate reid

# 디렉토리 모드
python cross_camera_reid.py --dir_a ./cam_a --dir_b ./cam_b --save_dir ./out

# 단일 파일 모드
python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4
```

---

## 6. 자주 겪는 오류

| 오류 메시지 | 원인 | 해결 |
|---|---|---|
| `No module named 'clip'` | CLIP git 설치 안 됨 | `pip install git+https://github.com/openai/CLIP.git` |
| `No module named 'ultralytics'` | conda activate 안 함 | `conda activate reid` |
| `YOLO bytetrack.yaml not found` | ultralytics 버전 문제 | `pip install ultralytics==8.2.2 --upgrade` |
| `libGL.so.1: cannot open` | 리눅스 OpenCV headless 문제 | `pip install opencv-python-headless` 로 교체 |
| `Killed` (메모리 부족) | RAM 부족 | 영상 해상도 줄이거나 DISPLAY_W/H 낮추기 |

---

## 7. 환경 관리

```bash
# 환경 목록 확인
conda env list

# 환경 비활성화
conda deactivate

# 환경 삭제 (재설치 필요할 때)
conda env remove -n reid
```
