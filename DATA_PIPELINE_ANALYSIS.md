# Custom Prompt Training - 데이터 파이프라인 분석

## 📊 전체 데이터 흐름 (Data Flow)

```
Dataset 생성
    ↓
VideoFeatureDataset (datasets/video_features.py)
    ├─ 클래스별 디렉토리 스캔
    └─ .npy 파일 로드
        ↓
    use_video_level_pooling=False (기본값)
    ├─ 각 비디오를 슬라이딩 윈도우로 분할
    ├─ 각 윈도우: [unit_duration, D]
    └─ Strict normal filtering 적용
        ↓
DataLoader
    └─ Batch 생성
        ↓
train_one_epoch()
    └─ features [B, T, D]
        ↓
VideoFeatureCLIP.forward()
    └─ 임베딩 및 분류
        ↓
CrossEntropyLoss
    └─ Backprop
```

---

## 🔍 1. 데이터셋 로드 단계 (custom_prompt_training.py)

### 코드 위치
**custom_prompt_training.py:161-182**

```python
train_dataset = VideoFeatureDataset(
    feature_dir=args.feature_dir,                    # 학습 피처 디렉토리
    normal_class="Normal",                           # "Normal" 클래스명
    unit_duration=1,                                 # 슬라이딩 윈도우 크기: 1초 (1프레임)
    overlap_ratio=0.0,                               # 윈도우 오버랩 없음
    strict_normal_sampling=True,                     # 엄격한 정상 필터링 적용
    use_video_level_pooling=False,                   # 슬라이딩 윈도우 사용
    verbose=True,
    seed=42,
)

classnames = train_dataset.classnames                # ['Normal', 'Abuse', 'Arrest', ...]
```

### 중요 파라미터 설명

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `unit_duration` | 1 | 각 윈도우가 1프레임만 포함 |
| `overlap_ratio` | 0.0 | 윈도우 간 오버랩 없음 → stride=1 |
| `strict_normal_sampling` | True | 비정상 비디오에서 이벤트 후 정상 레이블 윈도우 제거 |
| `use_video_level_pooling` | False | 비디오 전체 평균화 안 함 → 프레임별 샘플 생성 |

**결과:** 각 프레임이 독립적인 샘플 → 데이터 양 많음 → 계산량 많음

---

## 📝 2. 샘플 생성 단계 (VideoFeatureDataset._build_samples)

### 슬라이딩 윈도우 방식 (use_video_level_pooling=False)

**데이터셋 위치:** `datasets/video_features.py:213-232`

```python
# 각 클래스별로 처리
for class_dir, label in self.class_to_label.items():  # e.g., "Abuse" → label=1
    class_path = os.path.join(self.feature_dir, class_dir)

    # 클래스 디렉토리의 모든 .npy 파일
    npy_files = sorted([f for f in os.listdir(class_path) if f.endswith(".npy")])
    # 예: ["Abuse_1_x264.npy", "Abuse_2_x264.npy", ...]

    for npy_file in npy_files:
        npy_path = os.path.join(class_path, npy_file)
        self._process_npy_feature(
            npy_path, label, class_dir, is_normal_class, stride
        )
```

### 윈도우 분할 로직

**데이터셋 위치:** `datasets/video_features.py:263-294`

```python
def _process_npy_feature(npy_path, label, class_dir, is_normal_class, stride):
    feat = np.load(npy_path, mmap_mode="r")  # [T, D], T = 비디오 길이(초)
    total_seconds = feat.shape[0]            # T

    # 슬라이딩 윈도우 생성
    stride = max(1, int(unit_duration * (1.0 - overlap_ratio)))  # = 1
    num_windows = (total_seconds - unit_duration) // stride + 1
    # 예: T=1000, unit_duration=1, stride=1 → 1000개 윈도우

    for i in range(num_windows):
        start_sec = i * stride           # 0, 1, 2, ..., 999
        end_sec = start_sec + 1          # 1, 2, 3, ..., 1000

        # Strict normal filtering 적용
        if has_annotations and not is_normal_class and events:
            if not overlaps_event:
                if strict_normal_sampling and end_sec > earliest_event_start:
                    continue  # ← 이벤트 후 정상 윈도우 제거

        # 샘플 추가
        self.samples.append({
            "npy_path": npy_path,      # Abuse_1_x264.npy 경로
            "start_sec": start_sec,     # 0, 1, 2, ...
            "end_sec": end_sec,         # 1, 2, 3, ...
            "label": label,             # 1 (Abuse)
            "video_id": "Abuse_1_x264", # 파일명 (확장자 제외)
        })
```

**결과:** 각 비디오에서 T개 샘플 생성

**예시:**
```
Abuse_1_x264.npy [1000, 512]
  → [0:1], [1:2], [2:3], ..., [999:1000] (1000개 샘플)

Normal_1_x264.npy [2000, 512]
  → [0:1], [1:2], ..., [1999:2000] (2000개 샘플)

...

총 샘플 수 = 모든 비디오의 프레임 합
```

---

## 🔄 3. 배치 생성 단계 (DataLoader)

**custom_prompt_training.py:187-193**

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,           # 한 번에 32개 샘플
    shuffle=True,            # 매 에포크마다 섞음
    num_workers=4,           # 4개 병렬 처리
    pin_memory=True,         # GPU 전송 최적화
)
```

**DataLoader의 동작:**
1. 32개의 인덱스 선택 (무작위)
2. 각 인덱스에 대해 `VideoFeatureDataset.__getitem__(idx)` 호출
3. 32개 샘플을 Tensor로 변환 및 배치화

---

## 📦 4. 데이터 반환 (VideoFeatureDataset.__getitem__)

**데이터셋 위치:** `datasets/video_features.py:325-339`

```python
def __getitem__(self, idx: int):
    sample = self.samples[idx]  # 샘플 메타정보

    # .npy 파일 로드 (메모리 맵 사용)
    feat = np.load(sample["npy_path"], mmap_mode="r")  # [T, D]

    if sample.get("pool_video", False):
        # 비디오 레벨 평균화 (use_video_level_pooling=True일 때)
        feature_vector = np.mean(feat, axis=0)         # [D]
        feature_tensor = torch.from_numpy(feature_vector).float()
    else:
        # 슬라이딩 윈도우 (use_video_level_pooling=False일 때) ← 현재
        window = feat[sample["start_sec"]:sample["end_sec"]]  # [1, 512]
        feature_tensor = torch.from_numpy(np.array(window)).float()

    # 반환: (features [unit_duration, D], label, video_id)
    return feature_tensor, sample["label"], sample["video_id"]
    # 예: (torch[1, 512], 1, "Abuse_1_x264")
```

**반환값:**
- `feature_tensor`: Shape [1, 512] (unit_duration=1, D=512)
- `label`: int (0-13, 14개 클래스)
- `video_id`: str ("Abuse_1_x264" 등)

---

## 🧠 5. 학습 단계 (train_one_epoch)

**위치:** `train_video_feature_coop.py:171-211`

```python
def train_one_epoch(model, data_loader, optimizer, device, ...):
    for features, labels, _ in data_loader:  # ← 3개 값 언팩 (수정됨)
        features = features.to(device)       # [B, 1, 512] → GPU
        labels = labels.to(device)           # [B] → GPU

        # Forward pass
        logits = model(features)             # [B, 14]
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 메트릭 업데이트
        _, predicted = logits.max(1)         # [B]
        correct += predicted.eq(labels).sum().item()
```

---

## 🎯 6. 모델 처리 (VideoFeatureCLIP.forward)

**위치:** `trainers/video_feature_coop.py:446-475`

```python
def forward(self, features):  # features [B, 1, 512]
    # Temporal aggregation
    if features.dim() == 3:
        if self.temporal_agg == "mean":
            image_features = features.mean(dim=1)  # [B, 512]
        # ...

    # Generate text features
    prompts = self.prompt_learner()        # [14, seq_len, 512]
    tokenized_prompts = self.prompt_learner.tokenized_prompts
    text_features = self.text_encoder(prompts, tokenized_prompts)  # [14, 512]

    # L2 normalize
    image_features = image_features / image_features.norm(...)  # [B, 512]
    text_features = text_features / text_features.norm(...)    # [14, 512]

    # Cosine similarity
    logit_scale = self.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()  # [B, 14]

    return logits
```

**처리 과정:**
1. `[B, 1, 512]` → Temporal mean → `[B, 512]`
2. 각 클래스별 텍스트 임베딩 생성 → `[14, 512]`
3. Cosine similarity 계산 → `[B, 14]` logits

---

## 📊 데이터 크기 예시

### 입력 구조
```
학습 셋 구성:
├─ Normal/
│  ├─ Normal_1_x264.npy [2000초, 512차원]
│  ├─ Normal_2_x264.npy [1500초, 512차원]
│  └─ ... (20개 비디오)
├─ Abuse/
│  ├─ Abuse_1_x264.npy [1000초, 512차원]
│  └─ ... (20개 비디오)
└─ ... (12개 클래스)

총 비디오: ~280개 (14클래스 × 20)
```

### 샘플 생성
```
VideoFeatureDataset (slding window, unit_duration=1)
  Normal: 2000 + 1500 + ... = ~40,000 샘플
  Abuse:  1000 + 900 + ... = ~18,000 샘플
  Arrest: ~25,000 샘플
  ...

총 샘플 수: ~400,000+ (모든 비디오의 초 합산)
```

### 배치 처리
```
DataLoader batch_size=32
  Iteration 1: 32개 샘플
  Iteration 2: 32개 샘플
  ...
  Epoch = 400,000 / 32 ≈ 12,500 iterations
```

---

## 🚨 현재 문제: Loss가 변하지 않는 이유

### 가능성 1: 프롬프트 초기화 문제

**custom_prompt_training.py:256-258**
```python
ctx_init_str = " ".join(initial_prompts.get(cls, f"{cls}") for cls in classnames)
# 예: "a video showing physical abuse a video of police making arrest ..."

model = VideoFeatureCLIP(
    ...
    ctx_init=ctx_init_str,  # ← 문제: 모든 클래스 프롬프트 연결
    ...
)
```

**문제:**
- `ctx_init_str`이 너무 길어짐 (모든 클래스 프롬프트 합침)
- PromptLearner가 이를 **공유 컨텍스트**로 사용
- 결과: 모든 클래스가 비슷한 초기 프롬프트로 시작
- 따라서 text_features도 비슷함
- Loss 개선 불가능

### 가능성 2: 데이터 분포 문제

**정상 필터링이 과도하게 적용될 수 있음:**
```
strict_normal_sampling=True
  + 비정상 비디오에서 이벤트 후 정상 샘플 제거
  = 비정상 클래스의 샘플 수 급감
  = 클래스 불균형 심화
```

---

## ✅ 체크리스트

```
[ ] 1. 데이터가 제대로 로드되나?
    python -c "from datasets.video_features import VideoFeatureDataset; \
    ds = VideoFeatureDataset('...'); print(f'Samples: {len(ds)}')"

[ ] 2. 배치 형태가 올바른가?
    python -c "from torch.utils.data import DataLoader; \
    loader = DataLoader(ds, batch_size=32); \
    batch = next(iter(loader)); print(batch[0].shape, batch[1].shape)"

[ ] 3. 프롬프트가 올바르게 초기화되나?
    → debug_custom_prompt_init.py 실행

[ ] 4. 기울기가 흐르고 있나?
    → debug_gradient_flow.py 실행

[ ] 5. 텍스트 피처가 다양한가?
    → 14개 클래스의 text_features가 서로 다른지 확인
```

---

## 🔧 다음 단계

1. **프롬프트 초기화 문제 확인**
   - `ctx_init_str` 길이 확인
   - Text features 차이 확인

2. **데이터 분포 확인**
   - 클래스별 샘플 수 출력
   - Strict filtering 효과 측정

3. **Gradient flow 확인**
   - Loss 값 변화
   - Prompt learner 업데이트 확인
