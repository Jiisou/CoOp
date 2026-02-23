# Temporal Ground Truth를 이용한 CoOp 모델 평가

## 개요

Temporal_Anomaly_Annotation.txt의 Ground Truth 정보를 사용하여 모델을 평가합니다.

### 평가 방식

1. **Multi-class 평가**: 14개 클래스 분류 성능
2. **Binary 평가**: 정상(Normal) vs 이상(Anomaly) 이진 분류
   - Normal 클래스: 이상 없음
   - Anomaly: Abuse, Arrest, Arson 등 모든 이상 이벤트

### 지원 메트릭

- **Multi-class**: Accuracy, Precision, Recall, F1
- **Binary Anomaly Detection**:
  - Accuracy, Precision, Recall, F1
  - **AUC-ROC**: 이진 분류 성능
  - **AUC-PR**: 클래스 불균형이 있을 때 유용

---

## Annotation 파일 형식

```
Video_Name          Class        Event1_Start  Event1_End  Event2_Start  Event2_End
Abuse028_x264.mp4   Abuse        165           240         -1            -1
Arson011_x264.mp4   Arson        150           420         680           1267
```

**의미:**
- **Video_Name**: 비디오 파일명 (mp4)
- **Class**: 비디오의 주요 클래스
- **Event1_Start ~ Event2_End**: 이상 이벤트의 시간 범위 (프레임 단위)
  - -1: 해당 이벤트 없음
  - 최대 2개 이벤트 지원

**프레임 변환:**
- Frame Index = Time(초) × FPS
- 기본 FPS: 25

---

## 사용 방법

### 기본 명령어

```bash
python evaluate_with_temporal_gt.py \
    --test-feature-dir /mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/test \
    --annotation-file ./annotation/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt \
    --checkpoint-path ./output/video_feature_coop/video_feature_coop_best.pth \
    --n-ctx 16 \
    --csc \
    --output-dir ./output/evaluation_temporal
```

### 주요 인자

| 인자 | 설명 | 필수 |
|------|------|------|
| `--test-feature-dir` | 테스트 비디오 피처 디렉토리 | ✓ |
| `--annotation-file` | Temporal_Anomaly_Annotation.txt 경로 | ✓ |
| `--checkpoint-path` | 모델 체크포인트 | ✓ |
| `--n-ctx` | Context 토큰 개수 (학습시와 동일) | - |
| `--csc` | CSC 모드 사용 (학습시와 동일) | - |
| `--fps` | 비디오 FPS (프레임 단위 시간 변환) | 25 |
| `--output-dir` | 결과 저장 디렉토리 | - |

---

## 출력 결과

### 파일 생성

```
output/evaluation_temporal/
├── evaluation_results.json       # 평가 메트릭
├── roc_curve.png               # ROC 곡선
├── pr_curve.png                # Precision-Recall 곡선
└── confusion_matrix_binary.png  # 혼동행렬 (정상 vs 이상)
```

### evaluation_results.json

```json
{
  "multi_class": {
    "accuracy": 0.8234,
    "weighted_precision": 0.8156,
    "weighted_recall": 0.8123,
    "weighted_f1": 0.8139
  },
  "binary_anomaly_detection": {
    "accuracy": 0.8901,
    "precision": 0.8723,
    "recall": 0.8456,
    "f1": 0.8588,
    "auc_roc": 0.9234,
    "auc_pr": 0.9156
  },
  "statistics": {
    "total_samples": 5000,
    "samples_with_gt": 4800,
    "anomaly_samples": 2340,
    "normal_samples": 2460
  }
}
```

---

## 결과 해석

### Multi-class 평가
14개 클래스를 모두 분류하는 성능을 평가합니다.

- **Accuracy > 0.80**: 우수
- **Accuracy > 0.70**: 양호
- **Accuracy < 0.60**: 재학습 권장

### Binary Anomaly Detection

#### AUC-ROC 해석
- **AUC-ROC > 0.90**: 매우 우수한 이상 탐지 성능
- **AUC-ROC > 0.80**: 우수
- **AUC-ROC > 0.70**: 양호
- **AUC-ROC < 0.60**: 개선 필요

#### Precision vs Recall 트레이드오프

```
높은 Precision, 낮은 Recall
→ 이상을 매우 신중하게 예측 (False Positive 적음)
→ 하지만 이상을 놓치기 쉬움 (False Negative 많음)

낮은 Precision, 높은 Recall
→ 이상을 적극적으로 예측 (False Negative 적음)
→ 하지만 오경보가 많음 (False Positive 많음)
```

**추천:**
- 보안 시스템: Recall > Precision (이상 놓치면 안됨)
- 경보 시스템: Precision > Recall (오경보 줄여야 함)

### ROC & PR 곡선

**ROC 곡선 (Receiver Operating Characteristic)**
- 좌상단에 가까울수록 좋음
- 대각선: 무작위 분류기 (AUC=0.5)

**PR 곡선 (Precision-Recall)**
- 우상단에 가까울수록 좋음
- 클래스 불균형이 심할 때 ROC보다 유용

---

## 실제 예제

### 예제 1: 높은 성능 모델

```json
{
  "binary_anomaly_detection": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.90,
    "auc_roc": 0.96,
    "auc_pr": 0.95
  }
}
```

→ **배포 준비 완료** ✓

### 예제 2: 개선 필요

```json
{
  "binary_anomaly_detection": {
    "accuracy": 0.65,
    "precision": 0.58,
    "recall": 0.60,
    "auc_roc": 0.72,
    "auc_pr": 0.68
  }
}
```

→ **재학습 필요**

**개선 방안:**
- Learning rate 조정
- Context 토큰 개수 증가
- 에포크 수 증가
- 모델 구조 수정

---

## 고급 분석

### 클래스별 성능 확인

다시 실행하여 클래스별 메트릭 추가:

```python
# evaluate_with_temporal_gt.py 수정
# Per-class metrics for multi-class evaluation 추가
from sklearn.metrics import classification_report

report = classification_report(
    all_labels, class_preds,
    target_names=classnames
)
print(report)
```

### 특정 클래스의 이상 탐지 성능

```python
# 특정 클래스에 대한 binary 평가
class_idx = classnames.index("Abuse")
is_abuse = (all_labels == class_idx)

auc = roc_auc_score(is_abuse, all_logits[:, class_idx])
print(f"Abuse vs Others AUC: {auc:.4f}")
```

---

## 문제 해결

### 1. "samples_with_gt" 가 total_samples보다 훨씬 적음

**원인:** 일부 비디오가 annotation에 없음

**해결:**
```bash
# 로그에서 확인
# "Videos with ground truth: 4800 / 5000"

# annotation 파일에서 비디오 이름 확인
grep "video_name" ./annotation/.../Temporal_Anomaly_Annotation.txt
```

### 2. AUC-ROC 계산 불가 ("Not enough samples")

**원인:** 정상 또는 이상 샘플이 너무 적음

**해결:**
- 더 많은 테스트 데이터 사용
- 클래스 불균형 확인

### 3. 메모리 부족 (OOM)

```bash
python evaluate_with_temporal_gt.py ... --batch-size 16
```

---

## 성능 목표 설정

### 실제 보안 시스템 기준

| 메트릭 | 목표 | 설명 |
|--------|------|------|
| AUC-ROC | > 0.90 | 이상 vs 정상 분류 성능 |
| Recall | > 0.85 | 85% 이상의 이상 포착 |
| Precision | > 0.80 | 경보의 80% 이상이 실제 이상 |
| Accuracy | > 0.85 | 전체 85% 정확 |

---

## 배포 체크리스트

평가 결과가 다음을 만족하면 배포 가능:

- [ ] AUC-ROC > 0.90
- [ ] Recall > 0.85 (이상 놓치지 않음)
- [ ] Precision > 0.80 (오경보 최소화)
- [ ] Multi-class Accuracy > 0.80
- [ ] Confusion matrix에서 특이한 패턴 없음

---

## 다음 단계

1. **모델 평가** (이 스크립트)
2. **결과 분석** (메트릭 검토)
3. **필요시 재학습** (성능 개선)
4. **배포** (체크리스트 확인 후)
5. **모니터링** (실제 운영 중 성능 추적)

