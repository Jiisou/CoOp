# CoOp 비디오 피처 프롬프트 러닝 - 완전 가이드

## 📋 개요

이 프로젝트는 사전 추출된 비디오 피처와 MobileCLIP S0 텍스트 인코더를 활용한 **CoOp (Context Optimization)** 기반 프롬프트 러닝 구현입니다.

**주요 특징:**
- ✅ 사전 추출된 비디오 피처 (.npy 형식) 지원
- ✅ 클래스별 커스텀 초기 프롬프트 지원
- ✅ MobileCLIP S0 v1/v2 모두 지원
- ✅ 프레임 레벨 및 비디오 레벨 평가
- ✅ Temporal Ground Truth 기반 평가
- ✅ TensorBoard 통합

---

## 🗂️ 프로젝트 구조

```
CoOp/
├── datasets/
│   └── video_features.py              # 비디오 피처 데이터셋 로더
├── trainers/
│   └── video_feature_coop.py          # CoOp 모델 (MobileCLIP S0 기반)
├── scripts/
│   └── evaluate_coop_model.sh         # 평가 스크립트
├── train_video_feature_coop.py        # 메인 학습 스크립트
├── custom_prompt_training.py          # 커스텀 프롬프트 학습
├── evaluate_video_feature_coop.py     # 표준 평가
├── evaluate_with_temporal_gt.py       # Temporal GT 평가
├── example_custom_prompt_workflow.py  # 커스텀 프롬프트 예제
├── custom_prompts_example.json        # 프롬프트 예시
└── output/
    └── video_feature_coop/            # 학습 결과
        ├── checkpoints/
        ├── tensorboard/
        └── metrics.json
```

---

## 🚀 빠른 시작

### 1단계: 기본 학습

```bash
python train_video_feature_coop.py \
    --feature-dir /path/to/train/features \
    --val-feature-dir /path/to/val/features \
    --epochs 50 \
    --output-dir ./output/video_feature_coop
```

### 2단계: 커스텀 프롬프트로 학습 (선택)

```bash
python custom_prompt_training.py \
    --feature-dir /path/to/train/features \
    --val-feature-dir /path/to/val/features \
    --initial-prompts-file ./custom_prompts_example.json \
    --epochs 50 \
    --output-dir ./output/custom_prompts_v1
```

### 3단계: 평가

```bash
# 표준 평가
python evaluate_video_feature_coop.py \
    --test-feature-dir /path/to/test/features \
    --checkpoint-path ./output/video_feature_coop/video_feature_coop_best.pth \
    --csc \
    --output-dir ./output/evaluation

# Temporal GT 평가 (어노테이션 있을 때)
python evaluate_with_temporal_gt.py \
    --test-feature-dir /path/to/test/features \
    --annotation-file ./annotation/Temporal_Anomaly_Annotation.txt \
    --checkpoint-path ./output/video_feature_coop/video_feature_coop_best.pth \
    --output-dir ./output/evaluation_temporal
```

---

## 📚 상세 가이드

### 학습

#### 기본 CoOp 학습
👉 [train_video_feature_coop.py](./train_video_feature_coop.py) 직접 실행

**주요 하이퍼파라미터:**
- `--epochs`: 학습 에포크 수 (기본: 50)
- `--lr`: 학습률 (기본: 0.002)
- `--n-ctx`: 컨텍스트 토큰 개수 (기본: 16)
- `--batch-size`: 배치 크기 (기본: 32)
- `--csc`: Class-Specific Context 사용 (기본: True)

**출력:**
- `output/video_feature_coop/video_feature_coop_best.pth`: 최고 성능 모델
- `output/video_feature_coop/tensorboard/`: TensorBoard 로그

#### 커스텀 프롬프트로 학습
👉 [CUSTOM_PROMPT_TRAINING.md](./CUSTOM_PROMPT_TRAINING.md) 참고

**워크플로우:**
1. 프롬프트 설계 → `custom_prompts.json`
2. 학습 → `custom_prompt_training.py`
3. 분석 → `learned_prompts.json` 검토

**예제:**
```bash
python example_custom_prompt_workflow.py \
    --feature-dir /path/to/train \
    --val-feature-dir /path/to/val \
    --epochs 50
```

---

### 평가

#### 표준 평가 (Frame/Video level)
👉 [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) 참고

**출력 메트릭:**
- Frame-level Accuracy
- Video-level Accuracy
- Per-class Precision, Recall, F1
- Confusion Matrix
- ROC-AUC

#### Temporal Ground Truth 평가
👉 [TEMPORAL_GT_EVALUATION.md](./TEMPORAL_GT_EVALUATION.md) 참고

**평가 레벨:**
1. **Multi-class**: 14개 클래스 분류 성능
2. **Binary**: 정상(0) vs 이상(1) 이진 분류
3. **Anomaly-only**: 이상 샘플에 대한 다중 클래스 AUC

**어노테이션 형식:**
```
Video_Name          Class        Event1_Start  Event1_End  Event2_Start  Event2_End
Abuse028_x264.mp4   Abuse        165           240         -1            -1
Arson011_x264.mp4   Arson        150           420         680           1267
```

#### 빠른 평가
👉 [QUICKSTART_EVALUATION.md](./QUICKSTART_EVALUATION.md) 참고

기본 명령어와 결과 해석 방법

---

## 🔧 핵심 컴포넌트

### 1. datasets/video_features.py

**VideoFeatureDataset 클래스**

```python
dataset = VideoFeatureDataset(
    feature_dir="/path/to/features",
    annotation_dir="/path/to/annotations",
    unit_duration=1,              # 슬라이딩 윈도우 크기 (초)
    overlap_ratio=0.0,            # 윈도우 오버랩 비율
    strict_normal_sampling=True,  # 비정상 비디오의 이벤트 후 정상 샘플 제거
    use_video_level_pooling=False # True: [T,D]→[D] mean pooling, False: 슬라이딩 윈도우
)

# Returns: (feature_tensor [unit_duration, D], label_int, video_id)
feature, label, video_id = dataset[0]
```

**주요 기능:**
- 클래스 디렉토리 자동 스캔
- Annotation 기반 이벤트 시간 구간 처리
- Strict normal filtering: 이상 이벤트 후 label=0 스니펫 제거
- Video-level mean pooling 지원

### 2. trainers/video_feature_coop.py

**VideoFeatureCLIP 모델 구조**

```
Input: features [B, T, D]
  ↓
Temporal Aggregation (mean)
  ↓ [B, D]
L2 Normalize
  ↓ [B, D]
PromptLearner()
  ↓ prompts [n_cls, seq_len, ctx_dim]
TextEncoder()
  ↓ text_features [n_cls, D]
L2 Normalize
  ↓
Cosine Similarity + logit_scale
  ↓ logits [B, n_cls]
```

**핵심 수정사항:**
- **Mean Pooling 대신 EOT 추출**: 텍스트 인코더 출력의 평균을 사용 (EOT 위치 collapse 해결)
- **Device 일관성**: tokenized_prompts를 buffer로 등록하여 자동 GPU 이동
- **MobileCLIP v1/v2 지원**: Safe attribute extraction로 nested CustomTextCLIP 구조 처리

### 3. train_video_feature_coop.py

**학습 루프**

```
1. 데이터 로드
2. 모델 초기화
3. Optimizer 설정 (prompt_learner만 학습)
4. LR Scheduler (Warmup + Cosine Annealing)
5. 반복:
   - Forward pass
   - CrossEntropyLoss 계산
   - Backward pass (prompt_learner만)
   - 모델 체크포인트 저장
6. TensorBoard 로그
```

**저장 파일:**
- `checkpoints/best_model.pth`: 최고 검증 정확도
- `checkpoints/video_feature_coop_final.pth`: 최종 모델
- `checkpoints/video_feature_coop_ep*.pth`: 주기적 체크포인트
- `tensorboard/events.out.tfevents.*`: TensorBoard 로그

### 4. custom_prompt_training.py

**커스텀 프롬프트 지원**

```python
# 방법 1: JSON 파일
python custom_prompt_training.py \
    --initial-prompts-file ./my_prompts.json

# 방법 2: 명령어 라인
python custom_prompt_training.py \
    --custom-prompts Normal "a normal scene" Abuse "an attack"

# 방법 3: 기본값 (자동 생성)
python custom_prompt_training.py
```

**출력:**
- `initial_prompts.json`: 입력 프롬프트
- `learned_prompts.json`: 학습된 컨텍스트 벡터 및 임베딩

---

## 📊 학습 결과 해석

### TensorBoard 확인

```bash
tensorboard --logdir=./output/video_feature_coop/tensorboard
```

**모니터링 항목:**
- Training Loss: 감소 추이 확인
- Validation Accuracy: 수렴 확인
- Learning Rate: 스케줄러 동작 확인

### 메트릭 분석

#### 표준 평가 결과 (metrics.json)

```json
{
  "frame_level": {
    "accuracy": 0.8234,
    "macro_f1": 0.7891,
    "per_class": {
      "Normal": {"precision": 0.92, "recall": 0.88, "f1": 0.90},
      "Abuse": {"precision": 0.85, "recall": 0.82, "f1": 0.835}
    }
  },
  "video_level": {
    "accuracy": 0.8901
  }
}
```

**해석:**
- **Accuracy > 0.80**: 우수
- **Accuracy > 0.70**: 양호
- **Accuracy < 0.60**: 재학습 권장

#### Temporal GT 평가 결과

```json
{
  "multi_class": {
    "accuracy": 0.8234
  },
  "binary_anomaly_detection": {
    "accuracy": 0.8901,
    "auc_roc": 0.9234,
    "auc_pr": 0.9156
  },
  "anomaly_only": {
    "auc_roc": 0.8934
  }
}
```

**배포 기준:**
- AUC-ROC > 0.90
- Recall > 0.85 (이상 놓치지 않음)
- Precision > 0.80 (오경보 최소화)
- Accuracy > 0.85

---

## 🐛 문제 해결

### 학습이 수렴하지 않음 (Loss 고정, Accuracy 1%)

**원인 분석:**
- EOT 위치 collapse (이미 해결됨 - mean pooling으로 변경)
- Device mismatch (이미 해결됨 - buffer 등록)
- Learning rate 너무 높음

**해결책:**
```bash
# Learning rate 감소
python train_video_feature_coop.py ... --lr 0.0005

# 더 많은 에포크
python train_video_feature_coop.py ... --epochs 100

# 더 많은 context tokens
python train_video_feature_coop.py ... --n-ctx 32
```

### 메모리 부족 (OOM)

```bash
# 배치 크기 감소
python train_video_feature_coop.py ... --batch-size 16

# 워커 감소
python train_video_feature_coop.py ... --num-workers 2
```

### 평가 시 Video ID 매칭 실패

**원인:** 데이터셋 파일명과 어노테이션 파일명 불일치

**해결책:**
- 어노테이션 파일의 비디오 이름 확인
- `_x264`, `.mp4` 접미사 확인
- 평가 스크립트가 자동으로 정규화함

---

## 📈 권장 워크플로우

### 1. 기본 모델 학습

```bash
# 기본 설정으로 학습
python train_video_feature_coop.py \
    --feature-dir /path/to/train \
    --val-feature-dir /path/to/val \
    --epochs 50 \
    --output-dir ./output/baseline
```

### 2. 성능 평가

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir /path/to/test \
    --checkpoint-path ./output/baseline/video_feature_coop_best.pth \
    --output-dir ./output/eval_baseline
```

### 3. 커스텀 프롬프트 실험

```bash
# 프롬프트 설계
nano custom_prompts_v1.json

# 학습
python custom_prompt_training.py \
    --feature-dir /path/to/train \
    --val-feature-dir /path/to/val \
    --initial-prompts-file ./custom_prompts_v1.json \
    --output-dir ./output/custom_v1

# 평가
python evaluate_video_feature_coop.py \
    --test-feature-dir /path/to/test \
    --checkpoint-path ./output/custom_v1/checkpoints/best_model.pth \
    --output-dir ./output/eval_custom_v1
```

### 4. 결과 비교

```python
import json

# Baseline 결과
with open('./output/eval_baseline/metrics.json') as f:
    baseline = json.load(f)

# Custom 결과
with open('./output/eval_custom_v1/metrics.json') as f:
    custom = json.load(f)

# 비교
print(f"Baseline Acc: {baseline['frame_level']['accuracy']:.4f}")
print(f"Custom Acc: {custom['frame_level']['accuracy']:.4f}")
print(f"Improvement: {custom['frame_level']['accuracy'] - baseline['frame_level']['accuracy']:+.4f}")
```

### 5. 하이퍼파라미터 최적화 (선택)

성능이 부족하면:

```bash
# Context tokens 증가
python custom_prompt_training.py ... --n-ctx 32

# Learning rate 조정
python custom_prompt_training.py ... --lr 0.001

# 더 긴 학습
python custom_prompt_training.py ... --epochs 100
```

---

## 🔍 고급 사용법

### 비디오 레벨 Mean Pooling

```bash
# 학습: [T, D] → [D]로 집계
python train_video_feature_coop.py \
    --feature-dir /path/to/train \
    --val-feature-dir /path/to/val \
    --use-video-level-pooling \
    --output-dir ./output/video_pooling

# 평가: 동일하게 지정
python evaluate_video_feature_coop.py \
    --test-feature-dir /path/to/test \
    --checkpoint-path ./output/video_pooling/video_feature_coop_best.pth \
    --use-video-level-pooling
```

### Temporal Ground Truth 기반 평가

```bash
python evaluate_with_temporal_gt.py \
    --test-feature-dir /path/to/test \
    --annotation-file ./annotation/Temporal_Anomaly_Annotation.txt \
    --checkpoint-path ./output/baseline/video_feature_coop_best.pth \
    --fps 25 \
    --output-dir ./output/eval_temporal
```

### 학습된 프롬프트 분석

```python
import json
import numpy as np

with open('./output/custom_v1/learned_prompts.json') as f:
    learned = json.load(f)

for classname, data in learned.items():
    ctx = np.array(data['context_vector'])
    print(f"{classname}:")
    print(f"  Shape: {data['context_shape']}")
    print(f"  L2 norm: {np.linalg.norm(ctx):.4f}")
```

---

## 📋 체크리스트

### 배포 전 확인사항

- [ ] **학습 완료**
  - [ ] Loss 감소 확인
  - [ ] Validation accuracy 수렴
  - [ ] 체크포인트 저장됨

- [ ] **평가 결과**
  - [ ] Frame-level accuracy > 0.70
  - [ ] Video-level accuracy > 0.80
  - [ ] Per-class F1 모두 > 0.60

- [ ] **Temporal GT 평가** (해당 시 필수)
  - [ ] Multi-class accuracy > 0.80
  - [ ] Binary AUC-ROC > 0.90
  - [ ] Recall > 0.85

- [ ] **혼동행렬 분석**
  - [ ] 비정상적 패턴 없음
  - [ ] 특정 클래스 편향 없음

---

## 📚 추가 리소스

### 문서
- [CUSTOM_PROMPT_TRAINING.md](./CUSTOM_PROMPT_TRAINING.md) - 커스텀 프롬프트 상세 가이드
- [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) - 표준 평가 가이드
- [TEMPORAL_GT_EVALUATION.md](./TEMPORAL_GT_EVALUATION.md) - Temporal GT 평가 가이드
- [QUICKSTART_EVALUATION.md](./QUICKSTART_EVALUATION.md) - 빠른 시작

### 예제
- [custom_prompts_example.json](./custom_prompts_example.json) - 프롬프트 예시
- [example_custom_prompt_workflow.py](./example_custom_prompt_workflow.py) - 전체 워크플로우 예제

### 코드
- [datasets/video_features.py](./datasets/video_features.py) - 데이터셋
- [trainers/video_feature_coop.py](./trainers/video_feature_coop.py) - 모델
- [train_video_feature_coop.py](./train_video_feature_coop.py) - 학습
- [custom_prompt_training.py](./custom_prompt_training.py) - 커스텀 프롬프트
- [evaluate_video_feature_coop.py](./evaluate_video_feature_coop.py) - 평가
- [evaluate_with_temporal_gt.py](./evaluate_with_temporal_gt.py) - Temporal GT 평가

---

## 📞 자주 묻는 질문

**Q: 기본 학습과 커스텀 프롬프트 학습의 차이는?**

A: 기본 학습은 자동 생성 프롬프트("a video with {class}")로 시작, 커스텀 학습은 사용자가 정의한 구체적 프롬프트로 시작. 커스텀 프롬프트가 좋으면 수렴이 빠르고 성능이 향상될 수 있음.

**Q: V1과 V2 MobileCLIP의 차이는?**

A: v1은 mobileclip 패키지, v2는 open_clip에서 로드. 코드에서 자동으로 detection하고 지원.

**Q: 평가할 때 어떤 메트릭을 봐야 하나?**

A:
- 일반적: Video-level Accuracy
- 이상 탐지: Binary AUC-ROC
- 클래스 불균형: Anomaly-only AUC
- 상세 분석: Confusion matrix

**Q: 커스텀 프롬프트를 어떻게 쓰나요?**

A:
1. JSON 파일 준비 (custom_prompts.json)
2. custom_prompt_training.py 실행
3. learned_prompts.json에서 최종 벡터 확인

---

## 라이선스

MIT License

## 기여

이슈 및 PR 환영합니다!
