# CoOp Video Feature 모델 평가 가이드

## 개요

학습된 CoOp 모델을 테스트셋 비디오 피처와 어노테이션으로 평가합니다.

지원하는 평가 방식:
- **Frame-level 평가**: 개별 프레임 샘플별 정확도
- **Video-level 평가**: 비디오 단위 예측 집계 (max/mean pooling)
- **Per-class 메트릭**: 클래스별 Precision, Recall, F1
- **혼동행렬 (Confusion Matrix)**: 클래스 간 예측 오류 분석

---

## 필수 준비사항

### 1. 테스트셋 데이터
```
/path/to/test/features/
├── Abuse/
│   ├── Abuse_1_x264.npy
│   ├── Abuse_2_x264.npy
│   └── ...
├── Normal/
├── Fighting/
└── ...
```

### 2. 어노테이션 (선택사항)
```
/path/to/annotations/
├── Abuse.csv
├── Normal.csv
├── Fighting.csv
└── ...

# CSV 형식: file_name, start_time, end_time
Abuse_1_x264, 5.0, 15.0
Abuse_1_x264, 20.0, 30.0
```

### 3. 학습된 모델 체크포인트
학습 완료 후:
```
./output/video_feature_coop/video_feature_coop_best.pth
```

---

## 사용 방법

### 방법 1: Python 스크립트 직접 실행

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir /path/to/test/features \
    --test-annotation-dir /path/to/annotations \
    --checkpoint-path ./output/video_feature_coop/video_feature_coop_best.pth \
    --mobileclip-model mobileclip2_s0 \
    --n-ctx 16 \
    --csc \
    --batch-size 32 \
    --output-dir ./output/evaluation
```

### 방법 2: Shell 스크립트 사용

```bash
# 스크립트 편집 (경로 설정)
nano scripts/evaluate_coop_model.sh

# 스크립트 실행
bash scripts/evaluate_coop_model.sh
```

---

## 주요 인자 설명

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--test-feature-dir` | 테스트 비디오 피처 디렉토리 | 필수 |
| `--checkpoint-path` | 학습된 모델 체크포인트 경로 | 필수 |
| `--test-annotation-dir` | 어노테이션 디렉토리 | None |
| `--mobileclip-model` | MobileCLIP 모델 종류 | mobileclip2_s0 |
| `--n-ctx` | Context 토큰 개수 (학습시와 동일) | 16 |
| `--csc` | Class-specific context 사용 (학습시와 동일) | False |
| `--use-video-level-pooling` | 비디오 레벨 mean pooling 사용 | False |
| `--batch-size` | 배치 크기 | 32 |
| `--output-dir` | 결과 저장 디렉토리 | ./output/evaluation |
| `--device` | 사용할 디바이스 (cuda/cpu) | auto-detect |

---

## 출력 결과

평가 완료 후 `--output-dir`에 다음 파일들이 생성됩니다:

### 1. `metrics.json`
전체 메트릭 요약
```json
{
  "frame_level": {
    "accuracy": 0.8234,
    "macro_f1": 0.7891,
    "weighted_f1": 0.8156,
    "roc_auc_ovr": 0.9234,
    "per_class": {
      "Abuse": {"precision": 0.85, "recall": 0.82, "f1": 0.835},
      "Normal": {"precision": 0.92, "recall": 0.88, "f1": 0.90},
      ...
    }
  },
  "video_level": { ... }
}
```

### 2. `classification_report_frame.json` / `classification_report_video.json`
scikit-learn 형식의 상세 메트릭

### 3. `confusion_matrix_frame.png` / `confusion_matrix_video.png`
혼동행렬 시각화 이미지

### 4. `video_predictions.json` (video-level 평가 시)
비디오별 예측 결과
```json
{
  "predictions": [0, 1, 2, 0, ...],  # 예측 클래스
  "labels": [0, 1, 2, 0, ...],       # 정답
  "video_ids": ["Abuse_1", "Normal_5", ...]
}
```

---

## 결과 해석

### Frame-level vs Video-level

- **Frame-level Accuracy**: 개별 프레임 단위의 정확도
  - 더 어려운 과제 (노이즈 많음)
  - 정밀한 모델 성능 평가

- **Video-level Accuracy**: 비디오 단위 집계 후 정확도
  - 더 쉬운 과제 (노이즈 감소)
  - 실제 어플리케이션에 가까움

### Per-class 메트릭

- **Precision**: "이 클래스로 예측한 것 중 몇 %가 맞는가?"
- **Recall**: "이 클래스 샘플 중 몇 %를 맞췄는가?"
- **F1**: Precision과 Recall의 조화평균

소수 클래스(Assault, Explosion 등)에서 Recall이 낮으면 모델이 이들을 놓치는 경향이 있음.

### 혼동행렬

- 대각선: 올바른 예측
- 비대각선: 오류
  - 어떤 클래스를 다른 클래스로 자주 착각하는지 파악

---

## 예제: 완전한 평가 파이프라인

```bash
# 1. 학습 완료 후 최상 체크포인트 확인
ls -lh ./output/video_feature_coop/

# 2. 테스트셋에서 평가
python evaluate_video_feature_coop.py \
    --test-feature-dir /mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/test \
    --test-annotation-dir /mnt/c/JJS/UCF_Crimes/Annotations \
    --checkpoint-path ./output/video_feature_coop/video_feature_coop_best.pth \
    --mobileclip-model mobileclip2_s0 \
    --n-ctx 16 \
    --csc \
    --output-dir ./output/evaluation/final_test

# 3. 결과 확인
cat ./output/evaluation/final_test/metrics.json | python -m json.tool

# 4. 혼동행렬 이미지 확인
# 이미지 뷰어로 confusion_matrix_frame.png, confusion_matrix_video.png 확인
```

---

## 문제 해결

### 1. "Checkpoint not found"
- 체크포인트 경로 확인
- 학습이 완료되었는지 확인

### 2. "Module not found"
```bash
# CoOp 디렉토리에서 실행하는지 확인
cd /path/to/CoOp
python evaluate_video_feature_coop.py ...
```

### 3. 메모리 부족 (OOM)
```bash
# batch-size 감소
python evaluate_video_feature_coop.py ... --batch-size 16
```

### 4. 느린 속도
```bash
# num-workers 증가 (GPU 가용성에 따라)
python evaluate_video_feature_coop.py ... --num-workers 8
```

---

## 고급 사용법

### Video-level Mean Pooling 사용

학습시 `--use-video-level-pooling`을 사용했다면 평가시도 사용:

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir ... \
    --checkpoint-path ... \
    --use-video-level-pooling \
    ...
```

### CPU에서 평가

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir ... \
    --checkpoint-path ... \
    --device cpu \
    ...
```

### 커스텀 결과 분석

생성된 `metrics.json`과 `video_predictions.json`을 이용하여 추가 분석:

```python
import json

# 메트릭 로드
with open('./output/evaluation/metrics.json') as f:
    metrics = json.load(f)

# Video-level 결과 로드
with open('./output/evaluation/video_predictions.json') as f:
    predictions = json.load(f)

# 오분류된 비디오 분석
for pred, label, vid in zip(predictions['predictions'], predictions['labels'], predictions['video_ids']):
    if pred != label:
        print(f"Misclassified: {vid} (predicted: {pred}, true: {label})")
```

---

## 참고

- 평가 결과는 모델의 최종 성능을 나타냅니다
- Frame-level 정확도가 낮으면 모델 재학습이 필요할 수 있습니다
- Per-class 메트릭을 확인하여 특정 클래스의 성능 개선이 필요한지 판단하세요
- 혼동행렬에서 패턴을 찾아 모델 개선 방향을 결정하세요
