# CoOp 프롬프트 러닝 - 문서 색인

이 문서는 CoOp 비디오 피처 프롬프트 러닝 프로젝트의 모든 문서와 리소스를 색인화합니다.

## 📖 문서 가이드맵

### 🚀 처음 시작하는 사용자
1. **[README_COMPLETE_PIPELINE.md](./README_COMPLETE_PIPELINE.md)** ← 여기서 시작
   - 전체 프로젝트 개요
   - 빠른 시작 (3단계)
   - 핵심 컴포넌트 설명
   - 권장 워크플로우

2. **[QUICKSTART_EVALUATION.md](./QUICKSTART_EVALUATION.md)**
   - 기본 평가 명령어
   - 결과 읽기 방법
   - 성능 개선 팁

### 📚 상세 주제별 가이드

#### 🎓 학습 (Training)

- **[train_video_feature_coop.py](./train_video_feature_coop.py)**
  - 기본 CoOp 학습
  - 모든 클래스 자동 프롬프트 생성
  - TensorBoard 로깅

  ```bash
  python train_video_feature_coop.py \
      --feature-dir /path/to/train \
      --val-feature-dir /path/to/val \
      --epochs 50
  ```

- **[CUSTOM_PROMPT_TRAINING.md](./CUSTOM_PROMPT_TRAINING.md)** (필독!)
  - 클래스별 커스텀 초기 프롬프트 지정
  - 프롬프트 엔지니어링 팁
  - 학습된 프롬프트 벡터 추출
  - 성능 최적화 전략

  ```bash
  python custom_prompt_training.py \
      --initial-prompts-file ./custom_prompts.json \
      --epochs 50
  ```

#### 📊 평가 (Evaluation)

세 가지 평가 방식 지원:

1. **표준 평가** (모든 경우)
   - [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md)
   - Frame-level & Video-level accuracy
   - Per-class metrics (Precision, Recall, F1)
   - Confusion Matrix

   ```bash
   python evaluate_video_feature_coop.py \
       --test-feature-dir /path/to/test \
       --checkpoint-path ./best_model.pth
   ```

2. **Temporal Ground Truth 기반 평가** (어노테이션 있을 때)
   - [TEMPORAL_GT_EVALUATION.md](./TEMPORAL_GT_EVALUATION.md)
   - Multi-class (14개 클래스)
   - Binary (정상 vs 이상)
   - Anomaly-only (이상 샘플 다중 클래스 AUC)

   ```bash
   python evaluate_with_temporal_gt.py \
       --annotation-file ./Temporal_Anomaly_Annotation.txt \
       --checkpoint-path ./best_model.pth
   ```

3. **빠른 평가** (기본 명령어만 알고 싶을 때)
   - [QUICKSTART_EVALUATION.md](./QUICKSTART_EVALUATION.md)

#### 💻 코드 레벨

- **[datasets/video_features.py](./datasets/video_features.py)**
  - VideoFeatureDataset 클래스
  - 슬라이딩 윈도우 & 비디오 레벨 평균화 지원
  - 어노테이션 기반 이벤트 처리

- **[trainers/video_feature_coop.py](./trainers/video_feature_coop.py)**
  - VideoFeatureCLIP 모델
  - PromptLearner (학습 가능한 프롬프트)
  - TextEncoder (MobileCLIP S0)
  - MobileCLIP v1/v2 지원

---

## 🔗 파일별 설명

### 📋 주요 스크립트

| 파일 | 목적 | 사용 시기 |
|------|------|---------|
| [train_video_feature_coop.py](./train_video_feature_coop.py) | 기본 CoOp 모델 학습 | 프로젝트 시작 |
| [custom_prompt_training.py](./custom_prompt_training.py) | 커스텀 프롬프트로 학습 | 기본 모델 후 성능 개선 |
| [evaluate_video_feature_coop.py](./evaluate_video_feature_coop.py) | 표준 평가 | 항상 (필수) |
| [evaluate_with_temporal_gt.py](./evaluate_with_temporal_gt.py) | Temporal GT 평가 | 어노테이션 있을 때 |
| [example_custom_prompt_workflow.py](./example_custom_prompt_workflow.py) | 커스텀 프롬프트 전체 워크플로우 | 참고용 예제 |

### 📄 설정 & 데이터

| 파일 | 내용 |
|------|------|
| [custom_prompts_example.json](./custom_prompts_example.json) | 커스텀 프롬프트 예시 (바로 사용 가능) |
| [scripts/evaluate_coop_model.sh](./scripts/evaluate_coop_model.sh) | 평가 스크립트 (경로 설정 후 실행) |

### 📚 문서

| 파일 | 대상 | 길이 |
|------|------|------|
| [README_COMPLETE_PIPELINE.md](./README_COMPLETE_PIPELINE.md) | 전체 프로젝트 | 장문 |
| [CUSTOM_PROMPT_TRAINING.md](./CUSTOM_PROMPT_TRAINING.md) | 커스텀 프롬프트 | 중문 |
| [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) | 표준 평가 | 장문 |
| [TEMPORAL_GT_EVALUATION.md](./TEMPORAL_GT_EVALUATION.md) | Temporal GT 평가 | 중문 |
| [QUICKSTART_EVALUATION.md](./QUICKSTART_EVALUATION.md) | 빠른 시작 | 단문 |

---

## 🎯 사용 시나리오별 가이드

### 시나리오 1: 빠르게 기본 모델 학습하고 평가하기

1. README_COMPLETE_PIPELINE.md의 "빠른 시작" 섹션
2. train_video_feature_coop.py 실행
3. evaluate_video_feature_coop.py 실행
4. QUICKSTART_EVALUATION.md에서 결과 해석

**소요 시간:** 문서 읽기 10분 + 실행

### 시나리오 2: 커스텀 프롬프트로 성능 개선하기

1. CUSTOM_PROMPT_TRAINING.md 읽기
2. custom_prompts_example.json 참고하여 프롬프트 작성
3. custom_prompt_training.py 실행
4. learned_prompts.json 분석

**소요 시간:** 문서 읽기 30분 + 프롬프트 설계 + 실행

### 시나리오 3: Temporal 어노테이션으로 평가하기

1. TEMPORAL_GT_EVALUATION.md 읽기
2. Temporal_Anomaly_Annotation.txt 파일 준비
3. evaluate_with_temporal_gt.py 실행
4. evaluation_results.json 분석

**소요 시간:** 문서 읽기 20분 + 실행

### 시나리오 4: 완전한 실험 파이프라인 구축하기

1. README_COMPLETE_PIPELINE.md 읽기
2. example_custom_prompt_workflow.py 참고
3. 자신의 워크플로우 구현
4. 결과 비교 분석

**소요 시간:** 문서 읽기 1시간 + 구현

---

## 🔍 주요 개념 설명

### CoOp (Context Optimization)
- CLIP의 텍스트 인코더를 고정하고
- 프롬프트의 일부를 학습 가능한 벡터로 치환
- 해당 벡터를 데이터에 맞게 최적화

**장점:**
- 매우 적은 파라미터만 학습
- 빠른 수렴
- 좋은 일반화 성능

### PromptLearner
```
입력 프롬프트: "a video showing crime"
↓
토크나이제이션: [BOS] a video showing crime [EOT]
↓
Embedding: [embed_a, embed_video, ..., embed_crime, embed_EOT]
↓
컨텍스트 벡터 삽입: [embed_a, ctx_0, ctx_1, ..., ctx_15, embed_crime, embed_EOT]
↓
학습: ctx_i만 역전파로 업데이트
```

### Strict Normal Filtering
비정상 비디오 (예: Abuse 클래스)에서:
- 이벤트 구간: 레이블 = Abuse
- 이벤트 후: 레이블 = Normal (제거됨 - 노이즈)
- 이벤트 전: 레이블 = Normal (유지됨 - 정상 배경)

목적: 비정상 비디오의 혼합된 정상 샘플로 인한 혼동 방지

### Video-level Pooling
- Sliding window: 각 프레임을 독립 샘플로 처리
  - [T, D] → [1, D], [1, D], ..., [1, D] (T개)
  - 더 많은 샘플 → 더 정확한 학습

- Video-level mean: 전체 비디오를 집계
  - [T, D] → mean pooling → [D] (1개)
  - 계산 효율성 향상

---

## ✅ 체크리스트

### 설치 & 준비
- [ ] 프로젝트 클론/다운로드
- [ ] Python 3.8+ 설치
- [ ] 필요 라이브러리 설치 (`pip install -r requirements.txt`)
- [ ] 비디오 피처 파일 준비 (`.npy` 형식)

### 첫 실행
- [ ] README_COMPLETE_PIPELINE.md 읽기
- [ ] train_video_feature_coop.py 실행
- [ ] 학습 완료 확인 (체크포인트 생성)
- [ ] evaluate_video_feature_coop.py 실행
- [ ] 메트릭 확인

### 성능 개선
- [ ] CUSTOM_PROMPT_TRAINING.md 읽기
- [ ] 프롬프트 설계
- [ ] custom_prompt_training.py 실행
- [ ] 결과 비교

### 배포 전
- [ ] Temporal GT 평가 수행 (해당 시)
- [ ] 모든 메트릭 확인
- [ ] 혼동행렬 분석
- [ ] 배포 기준 충족 확인

---

## 📞 문서 선택 가이드

| 목표 | 읽을 문서 |
|------|---------|
| 전체 이해 | README_COMPLETE_PIPELINE.md |
| 빨리 시작 | QUICKSTART_EVALUATION.md |
| 커스텀 프롬프트 | CUSTOM_PROMPT_TRAINING.md |
| 상세 평가 | EVALUATION_GUIDE.md |
| Temporal 평가 | TEMPORAL_GT_EVALUATION.md |
| 코드 학습 | 소스 코드 + 주석 |
| 예제 보기 | example_custom_prompt_workflow.py |

---

## 🚨 일반적인 문제

### "어디서부터 시작해야 하나요?"
→ [README_COMPLETE_PIPELINE.md](./README_COMPLETE_PIPELINE.md)의 "빠른 시작" 섹션

### "커스텀 프롬프트를 어떻게 쓰나요?"
→ [CUSTOM_PROMPT_TRAINING.md](./CUSTOM_PROMPT_TRAINING.md)

### "평가 결과를 어떻게 해석하나요?"
→ [QUICKSTART_EVALUATION.md](./QUICKSTART_EVALUATION.md) 또는 [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md)

### "Temporal 어노테이션이 있는데?"
→ [TEMPORAL_GT_EVALUATION.md](./TEMPORAL_GT_EVALUATION.md)

### "코드를 이해하고 싶어요"
→ [README_COMPLETE_PIPELINE.md](./README_COMPLETE_PIPELINE.md)의 "핵심 컴포넌트" 섹션

---

## 📚 추가 리소스

### 외부 링크
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [CoOp Paper](https://arxiv.org/abs/2109.01134)
- [MobileCLIP](https://github.com/openai/CLIP)
- [open_clip](https://github.com/mlfoundations/open_clip)

### 관련 코드
- `/datasets/` - 데이터 로딩
- `/trainers/` - 모델 구현
- `/scripts/` - 유틸리티 스크립트

---

**마지막 업데이트:** 2026-02-23
**문서 버전:** 1.0
