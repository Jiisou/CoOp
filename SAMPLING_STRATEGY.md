# Custom Prompt Training - 샘플링 전략

## 📊 현재 구현 (strict_normal_sampling=True)

### Normal 샘플
```
Data source: Normal 폴더
├─ Normal_1_x264.npy [1000초]
├─ Normal_2_x264.npy [1500초]
└─ ...

처리:
  - 모든 프레임 → sliding window → Normal 레이블
  - 결과: ~2500 Normal 샘플
```

### Abnormal 샘플 (Abuse, Arrest, ...)
```
Data source: 각 Abnormal 클래스 폴더
├─ Abuse_1_x264.npy [1000초]
│   ├─ [0:100s]    ← 이벤트 전
│   ├─ [100:300s]  ← 이벤트 발생
│   └─ [300:1000s] ← 이벤트 후
├─ Abuse_2_x264.npy
└─ ...

처리 (strict_normal_sampling=True):
  ✓ 이벤트 전 ([0:100s])
    → Abuse 레이블 (같은 비디오의 맥락, pre-event context)

  ✓ 이벤트 구간 ([100:300s])
    → Abuse 레이블 (실제 이벤트)

  ✗ 이벤트 후 ([300:1000s])
    → 제외됨 (너무 노이즈, 실제 이벤트 아님)

결과: ~1800 Abnormal 샘플 (정제됨)
```

---

## 📈 샘플 구성

### 학습셋 (strict_normal_sampling=True)
```
Normal:   ~40,000 샘플 (Normal 폴더만)
Abuse:    ~1,800 샘플 (이벤트 구간 + 이벤트 전)
Arrest:   ~2,200 샘플
...
Total:    ~400,000+ 샘플 (정제됨)

특징:
- Normal 클래스가 과대 대표 (정상이 비정상보다 훨씬 많음)
- 각 Abnormal 클래스는 실제 이벤트만 포함 (노이즈 최소화)
```

### 검증셋 (strict_normal_sampling=False)
```
모든 샘플 포함 (이벤트 후 포함)
- 평가시 실제 분포를 반영
- 노이즈가 있어도 포함 (현실적 평가)
```

---

## 🔍 Normal 샘플만 사용하는 방법

### 현재 구현
```python
# Normal 폴더의 모든 프레임 → Normal 레이블 ✓
if is_normal_class:  # "Normal" 폴더
    label = normal_label  # 0
    # 모든 프레임 사용

# Abnormal 폴더의 normal 부분 → 제외 ✓
else:  # "Abuse", "Arrest" 등
    if strict_normal_sampling and post_event:
        continue  # 제외
```

**결과:**
- Normal 레이블 = Normal 폴더만 ✓
- Abnormal 레이블 = Abnormal 폴더의 이벤트 구간 + 이벤트 전 ✓

---

## 📋 구체적 예시

### 데이터 구조
```
train/
├─ Normal/
│  ├─ Normal_001_x264.npy [2000초]
│  └─ Normal_002_x264.npy [1500초]
│
├─ Abuse/
│  ├─ Abuse_001_x264.npy [1000초]
│  │   ├─ [0:100s]     ← 이벤트 전
│  │   ├─ [100:300s]   ← 이벤트
│  │   └─ [300:1000s]  ← 이벤트 후
│  └─ Abuse_002_x264.npy [800초]
│      ├─ [0:50s]      ← 이벤트 전
│      ├─ [50:200s]    ← 이벤트
│      └─ [200:800s]   ← 이벤트 후
│
└─ Arrest/
   └─ ...
```

### 처리 결과

#### Normal 샘플
```
Normal_001_x264.npy [2000초]
  → [0:1], [1:2], [2:3], ..., [1999:2000] (2000개)
  → 모두 Normal 레이블

Normal_002_x264.npy [1500초]
  → [0:1], [1:2], ..., [1499:1500] (1500개)
  → 모두 Normal 레이블

총: 3500개 Normal 샘플 (Normal 폴더만!)
```

#### Abuse 샘플 (strict_normal_sampling=True)
```
Abuse_001_x264.npy:
  [0:100s]      → 100개 Abuse 샘플 (이벤트 전, 포함)
  [100:300s]    → 200개 Abuse 샘플 (이벤트, 포함)
  [300:1000s]   → 700개 샘플 제외 (이벤트 후, 제거)
  결과: 300개 Abuse 샘플

Abuse_002_x264.npy:
  [0:50s]       → 50개 Abuse 샘플 (이벤트 전, 포함)
  [50:200s]     → 150개 Abuse 샘플 (이벤트, 포함)
  [200:800s]    → 600개 샘플 제외 (이벤트 후, 제거)
  결과: 200개 Abuse 샘플

총: 500개 Abuse 샘플 (정제됨, 노이즈 제거)
```

---

## ✅ 정리

| 항목 | 구현 | 상태 |
|------|------|------|
| **Normal 샘플** | Normal 폴더만 사용 | ✅ 구현됨 |
| **Abnormal 샘플** | 이벤트 구간 + 이벤트 전 | ✅ 구현됨 |
| **포스트 이벤트** | strict_normal_sampling=True일 때 제거 | ✅ 구현됨 |
| **학습셋** | strict_normal_sampling=True (정제) | ✅ 적용됨 |
| **검증셋** | strict_normal_sampling=False (현실적) | ✅ 적용됨 |

---

## 🎯 추가 옵션 (필요시)

만약 더 엄격하게 하고 싶다면:

### 옵션 1: 이벤트 전 제거 (이벤트만 사용)
```python
# 현재: 이벤트 전 + 이벤트 구간
# 변경: 이벤트 구간만

if not overlaps_event:
    continue  # 이벤트 전도 제외
```

### 옵션 2: Normal 클래스만 추가 필터링
```python
# 추가 기준:
# - video_length < 30초: 제외 (너무 짧음)
# - suspicious_scene: 제외 (경계 부분)
```

---

## 📝 권장사항

### 현재 설정 (권장)
```python
# 학습
strict_normal_sampling=True
# Normal: Normal 폴더만
# Abnormal: 이벤트 구간 + 이벤트 전 (노이즈 최소화)

# 검증
strict_normal_sampling=False
# 모든 샘플 포함 (현실적 평가)
```

**이유:**
- Normal 샘플이 충분함 (Normal 폴더만 해도 ~40K)
- Abnormal 샘플을 정제 (노이즈 제거)
- 클래스 불균형이 있지만 현실적 (정상이 비정상보다 많음)
- 검증시 현실적 평가 (노이즈 포함)

