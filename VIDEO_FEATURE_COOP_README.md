# CoOp Prompt Learning on Pre-Extracted Video Features

이 구현은 사전 추출된 비디오 피처(.npy)를 사용하여 MobileCLIP S0 기반 CoOp 프롬프트 러닝을 수행합니다.

## 주요 특징

1. **MobileCLIP S0 텍스트 인코더**: OpenAI CLIP 대신 MobileCLIP S0의 텍스트 인코더 사용
2. **Strict Normal Snippet Filtering**: ETRIFeatureDataset 패턴을 따라 노이즈가 있는 post-event 윈도우 필터링
3. **프레임별 개별 처리**: [T, D] 비디오 피처를 T개의 독립 샘플로 분리
4. **비디오 레벨 평가**: 프레임 logit 평균을 통한 비디오 분류 정확도 측정

## 디렉토리 구조

```
{root}/
├── train/
│   ├── class_name_1/
│   │   ├── class_name_1_001_x264.npy   # shape: [T, D]
│   │   ├── class_name_1_002_x264.npy
│   │   └── ...
│   ├── class_name_2/
│   │   └── ...
│   └── ...
├── val/
│   └── (동일 구조)
└── test/
    └── (동일 구조)
```

**피처 파일 형식**:
- `.npy` 파일, shape: `[T, D]` (T: 초 단위 프레임 수, D: 피처 차원)
- MobileCLIP S0의 경우 D=512
- 각 row는 1초당 1프레임을 나타냄

## 설치

### 필수 패키지

```bash
pip install torch torchvision torchaudio
pip install open_clip_torch
pip install numpy pandas tqdm tensorboard scikit-learn
```

### MobileCLIP S0 가중치

```bash
# Apple ml-mobileclip 저장소에서 다운로드
# 또는 open_clip을 통해 자동 다운로드
```

## 사용법

### 1. 기본 학습

```bash
python train_video_feature_coop.py \
    --feature-dir /path/to/features/train \
    --val-feature-dir /path/to/features/val \
    --mobileclip-path /path/to/mobileclip_s0.pt \
    --n-ctx 16 \
    --lr 0.002 \
    --epochs 50 \
    --batch-size 32
```

### 2. Annotation 기반 Strict Filtering 사용

```bash
python train_video_feature_coop.py \
    --feature-dir /path/to/features/train \
    --val-feature-dir /path/to/features/val \
    --annotation-dir /path/to/annotations \
    --val-annotation-dir /path/to/val_annotations \
    --mobileclip-path /path/to/mobileclip_s0.pt \
    --strict-normal-sampling \
    --epochs 50
```

**Annotation CSV 형식**:
- 파일명: `{class_name}_timestamp.csv`
- 컬럼: `file_name`, `start_time`, `end_time`
- `start_time`, `end_time`: HH:MM:SS 또는 MM:SS 또는 초 단위

### 3. Context 초기화 사용

```bash
python train_video_feature_coop.py \
    --feature-dir /path/to/features/train \
    --val-feature-dir /path/to/features/val \
    --mobileclip-path /path/to/mobileclip_s0.pt \
    --ctx-init "a video of a" \
    --class-token-position end \
    --epochs 50
```

### 4. Class-Specific Context (CSC) 사용

```bash
python train_video_feature_coop.py \
    --feature-dir /path/to/features/train \
    --val-feature-dir /path/to/features/val \
    --mobileclip-path /path/to/mobileclip_s0.pt \
    --csc \
    --n-ctx 4 \
    --epochs 50
```

### 5. Shell Script 사용

```bash
# scripts/video_feature_coop/train.sh 편집하여 경로 설정
chmod +x scripts/video_feature_coop/train.sh
./scripts/video_feature_coop/train.sh
```

### 6. 평가 전용 모드

```bash
python train_video_feature_coop.py \
    --feature-dir /path/to/features/test \
    --val-feature-dir /path/to/features/test \
    --mobileclip-path /path/to/mobileclip_s0.pt \
    --resume-ckpt ./output/video_feature_coop/video_feature_coop_best.pth \
    --eval-only
```

## 주요 파라미터

### CoOp 관련
- `--n-ctx`: 학습 가능한 컨텍스트 토큰 수 (기본값: 16)
- `--ctx-init`: 컨텍스트 초기화 단어 (예: "a video of a")
- `--csc`: Class-Specific Context 사용 여부
- `--class-token-position`: 클래스 토큰 위치 (`end`, `middle`, `front`)

### 데이터 관련
- `--unit-duration`: 윈도우 크기 (초 단위, 기본값: 1)
- `--overlap-ratio`: Sliding window overlap 비율 (0.0~1.0, 기본값: 0.0)
- `--strict-normal-sampling`: Strict normal filtering 적용 여부 (기본값: True)
- `--normal-class`: Normal 클래스 디렉토리 이름 (기본값: "normal")

### 학습 관련
- `--lr`: Learning rate (기본값: 0.002)
- `--epochs`: 에폭 수 (기본값: 50)
- `--batch-size`: 배치 크기 (기본값: 32)
- `--warmup-epochs`: Warmup 에폭 수 (기본값: 1)
- `--patience`: Early stopping patience (기본값: 10)

### 시간 집계
- `--temporal-agg`: 시간축 집계 방법 (`mean`, `max`, 기본값: `mean`)

## 출력 구조

```
./output/video_feature_coop/
├── checkpoints/
│   ├── video_feature_coop_best.pth      # 최고 검증 정확도 모델
│   ├── video_feature_coop_ep10.pth      # 10 에폭 체크포인트
│   ├── video_feature_coop_ep20.pth
│   └── video_feature_coop_final.pth     # 최종 모델
└── tensorboard/
    └── video_feature_coop_YYMMDDHHMM/
        └── events.out.tfevents.*
```

## TensorBoard 사용

```bash
tensorboard --logdir ./output/video_feature_coop/tensorboard
# 브라우저에서 http://localhost:6006 접속
```

로그 내용:
- Loss (train/val)
- Accuracy (train/val, frame-level)
- Learning Rate
- Per-Class Accuracy
- Video-Level Accuracy (평가 시)

## 구현 세부사항

### 1. Dataset: `datasets/video_features.py`

ETRIFeatureDataset 패턴을 다중 클래스 분류에 맞게 적응:
- Memory-mapped .npy 로딩 (`mmap_mode='r'`)
- Sliding window 생성 (overlap 지원)
- Strict normal sampling: 비정상 클래스 비디오에서 이벤트 이후 라벨=0 윈도우 제거
- Annotation 기반 라벨링

### 2. Model: `trainers/video_feature_coop.py`

- `PromptLearner`: 학습 가능한 컨텍스트 벡터 (trainers/coop.py 기반)
- `TextEncoder`: MobileCLIP 텍스트 트랜스포머 래핑
- `VideoFeatureCLIP`: 이미지 인코더 없이 피처 직접 사용
  - Temporal aggregation: [B, T, D] → [B, D]
  - L2 normalization
  - Cosine similarity with temperature scaling

### 3. Training: `train_video_feature_coop.py`

train_resmlp.py 패턴 따름:
- `train_one_epoch()`: tqdm 진행률, CrossEntropyLoss
- `validate()`: Frame-level accuracy + per-class accuracy
- `validate_video_level()`: Video-level accuracy (프레임 logit 평균)
- SGD optimizer (prompt_learner만 학습)
- Warmup + Cosine annealing scheduler
- EarlyStopping, TensorBoard 로깅

## Strict Normal Snippet Filtering

비정상/액션 클래스 비디오에서:
1. Annotation에서 이벤트 시작/종료 시간 로드
2. 각 sliding window가 이벤트와 겹치는지 확인
3. 겹치지 않는(label=0) 윈도우 중:
   - 가장 이른 이벤트 시작 시간 **이전**에 끝나는 윈도우 → 유지
   - 이벤트 시작 시간 **이후**에 끝나는 윈도우 → **제거** (노이즈 방지)

이를 통해 액션 후 잔여 신호가 있는 애매한 프레임을 학습에서 제외.

## 참조 코드

- Base CoOp: `trainers/coop.py`
- Reference training loop: `ExploreVAD/vanilla_spotting/clip/npy_training/train_resmlp.py`
- Reference dataset: `ExploreVAD/vanilla_spotting/clip/dataset.py` (ETRIFeatureDataset)

## 문제 해결

### MobileCLIP 로딩 오류

```python
# open_clip을 통한 자동 다운로드
import open_clip
model, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S0', pretrained='datacomp_xl_s13b_b90k')
```

### Annotation 파일 형식 오류

CSV 파일이 다음 컬럼을 포함하는지 확인:
- `file_name`: 비디오 파일명 (확장자 포함 또는 제외)
- `start_time`: 이벤트 시작 (HH:MM:SS, MM:SS, 또는 초)
- `end_time`: 이벤트 종료

### 메모리 부족

- `--batch-size` 감소 (예: 16 또는 8)
- `--num-workers` 감소 (예: 2 또는 0)
- `--unit-duration` 감소 (프레임 수 줄이기)

### CUDA Out of Memory

```bash
# Gradient accumulation 사용
# 또는 배치 크기 감소
python train_video_feature_coop.py \
    --batch-size 8 \
    --epochs 50
```

## 성능 팁

1. **학습률**: SGD는 lr=0.002가 기본, 더 큰 데이터셋은 0.005 시도
2. **Warmup**: 안정적인 학습을 위해 1-2 에폭 warmup 권장
3. **Context 수**: 작은 데이터셋은 n_ctx=4, 큰 데이터셋은 16 시도
4. **Temporal aggregation**: `mean`이 일반적으로 안정적, `max`는 이벤트 감지에 유리할 수 있음
5. **Overlap**: 학습 데이터 증강을 위해 overlap_ratio=0.5 시도

## 인용

CoOp 논문:
```
@inproceedings{zhou2022learning,
  title={Learning to Prompt for Vision-Language Models},
  author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  booktitle={CVPR},
  year={2022}
}
```

MobileCLIP:
```
@article{vasu2024mobileclip,
  title={MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
  author={Vasu, Pavan Kumar Anasosalu and others},
  journal={CVPR},
  year={2024}
}
```
