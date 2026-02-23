# CoOp 평가 빠른 시작

## 기본 평가 명령어

학습이 완료된 후 다음 명령어로 테스트셋 평가를 실행합니다:

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir /mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/test \
    --test-annotation-dir /mnt/c/JJS/UCF_Crimes/Annotations \
    --checkpoint-path ./output/video_feature_coop/video_feature_coop_best.pth \
    --mobileclip-model mobileclip2_s0 \
    --n-ctx 16 \
    --csc \
    --output-dir ./output/evaluation
```

---

## 단계별 진행

### 1단계: 학습 완료 확인

```bash
# 체크포인트 파일 확인
ls -lh ./output/video_feature_coop/video_feature_coop_best.pth
```

예상 출력:
```
-rw-r--r-- 1 user user 1.2M ... video_feature_coop_best.pth
```

### 2단계: 평가 실행

```bash
# 기본 평가 (GPU 자동 감지)
python evaluate_video_feature_coop.py \
    --test-feature-dir /mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/test \
    --checkpoint-path ./output/video_feature_coop/video_feature_coop_best.pth \
    --csc  # CSC를 사용한 경우
```

### 3단계: 결과 확인

```bash
# 메트릭 출력
cat ./output/evaluation/metrics.json | python -m json.tool

# 결과 파일 목록
ls -lh ./output/evaluation/
```

예상 출력:
```
metrics.json
classification_report_frame.json
classification_report_video.json
confusion_matrix_frame.png
confusion_matrix_video.png
video_predictions.json
```

---

## 주요 결과 읽기

### Accuracy 해석

```
Frame-level Accuracy: 0.8234  (82.34%)
Video-level Accuracy: 0.8901  (89.01%)
```

- **Frame-level**: 개별 프레임의 정확도 (더 어려움)
- **Video-level**: 비디오 단위 예측의 정확도 (더 쉬움)

### Per-class Performance

```json
{
  "Abuse": {
    "precision": 0.85,  // "Abuse로 예측한 것 중 85% 정확"
    "recall": 0.82,     // "실제 Abuse의 82% 맞춤"
    "f1": 0.835         // precision과 recall의 조화평균
  },
  "Normal": {
    "precision": 0.92,
    "recall": 0.88,
    "f1": 0.90
  }
}
```

낮은 recall → 해당 클래스를 자주 놓침
낮은 precision → 해당 클래스로 잘못 예측함

### Confusion Matrix 분석

이미지 파일 `confusion_matrix_frame.png`에서:
- 대각선: 정확한 예측
- 비대각선: 오분류
  - 어느 클래스로 자주 헷갈리는지 확인 가능

---

## 학습 설정에 따른 평가

### 1. CSC (Class-Specific Context) 사용

학습 시 `--csc` 사용:
```bash
python evaluate_video_feature_coop.py ... --csc
```

학습 시 CSC 미사용:
```bash
python evaluate_video_feature_coop.py ...  # --csc 생략
```

### 2. Video-level Pooling 사용

학습 시 `--use-video-level-pooling` 사용:
```bash
python evaluate_video_feature_coop.py ... --use-video-level-pooling
```

이 경우:
- Frame-level 평가만 수행 (각 샘플이 비디오 단위)
- Video-level 평가 생략

### 3. Context 토큰 개수

학습시 사용한 `--n-ctx` 값 확인 후 동일하게 설정:
```bash
python evaluate_video_feature_coop.py ... --n-ctx 16
```

---

## 성능 개선 팁

### 낮은 Accuracy의 경우

1. **Per-class 성능 확인**
   - 특정 클래스가 자주 틀리는지 확인
   - 데이터 불균형 확인

2. **Confusion Matrix 분석**
   - 어떤 클래스들이 자주 헷갈리는지 확인
   - 클래스명 유사성, 특징 중복성 검토

3. **모델 재학습**
   ```bash
   # Learning rate 조정
   python train_video_feature_coop.py ... --lr 0.001

   # Context 토큰 개수 증가
   python train_video_feature_coop.py ... --n-ctx 32

   # 더 많은 에포크 학습
   python train_video_feature_coop.py ... --epochs 100
   ```

### 높은 Frame-level과 낮은 Video-level의 경우

→ 단일 프레임은 맞추지만 비디오 단위 집계에서 오류 발생

해결책:
```bash
# 다른 집계 방식 시도 (코드 수정 필요)
# 현재: max pooling
# 시도: mean pooling, voting
```

---

## 추가 명령어

### 세부 분석 결과 저장

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir ... \
    --checkpoint-path ... \
    --output-dir ./output/evaluation_detailed \
    --batch-size 16 \
    --num-workers 8
```

### CPU에서 평가 (GPU 없을 때)

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir ... \
    --checkpoint-path ... \
    --device cpu \
    --batch-size 8  # 작은 배치 크기
```

### 어노테이션 없이 평가

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir ... \
    --checkpoint-path ... \
    # --test-annotation-dir 생략
```

---

## 결과 예시

### 좋은 결과 (모델이 잘 학습됨)

```
Frame-level Accuracy: 0.82
Video-level Accuracy: 0.91
Macro F1: 0.85

Per-class (예시):
  Abuse:      Precision=0.88, Recall=0.85, F1=0.865
  Arrest:     Precision=0.90, Recall=0.88, F1=0.890
  Normal:     Precision=0.95, Recall=0.92, F1=0.935
```

### 개선 필요 (모델 재학습)

```
Frame-level Accuracy: 0.45
Video-level Accuracy: 0.52
Macro F1: 0.42

Per-class (예시):
  Assault:    Precision=0.30, Recall=0.25, F1=0.275  ← 너무 낮음
  Explosion:  Precision=0.35, Recall=0.28, F1=0.313  ← 너무 낮음
```

→ 모델 재학습, 데이터 확인, 피처 품질 검토 필요

---

## 자주 묻는 질문

**Q: Frame-level과 Video-level 중 어느 것을 봐야 하나?**
A: 용도에 따라 다릅니다.
- 프레임 단위 분류 필요 → Frame-level
- 비디오 단위 분류 필요 → Video-level
- 일반적으로 Video-level이 실제 성능에 더 가까움

**Q: Per-class F1이 낮은 클래스는?**
A: 해당 클래스가 부족하거나 모델이 잘 못 배웠을 가능성.
- 데이터 추가
- 가중치 조정
- 모델 구조 개선

**Q: Confusion Matrix에서 패턴이 보이면?**
A: 클래스 간 유사성이 있을 수 있음.
- 클래스 정의 재검토
- 피처 품질 확인
- 더 많은 context tokens 사용

---

## 다음 단계

결과를 바탕으로:

1. **성능이 좋으면**: 모델 배포, 실제 운영 환경에서 테스트
2. **성능이 부족하면**:
   - 모델 파인튜닝
   - 하이퍼파라미터 조정
   - 데이터 품질 개선
3. **특정 클래스가 문제**: 해당 클래스 데이터 분석 및 추가

