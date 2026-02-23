# CoOp 커스텀 초기 프롬프트 학습 가이드

## 개요

기본 CoOp 모델은 모든 클래스에 대해 동일한 초기 프롬프트로 시작합니다. 하지만 각 클래스의 특성을 반영한 **커스텀 초기 프롬프트**로 시작하면 학습 수렴이 빨라지고 성능이 향상될 수 있습니다.

이 가이드는 다음을 다룹니다:
- 커스텀 프롬프트 준비 (JSON 파일, 명령어 라인, 기본값)
- 커스텀 프롬프트를 이용한 모델 학습
- 학습된 최종 프롬프트 벡터 추출 및 분석

---

## 빠른 시작

### 방법 1: 기본 프롬프트로 학습 (권장 시작점)

```bash
python custom_prompt_training.py \
    --feature-dir /path/to/train/features \
    --val-feature-dir /path/to/val/features \
    --epochs 50 \
    --output-dir ./output/custom_prompts_default
```

자동으로 생성된 기본 프롬프트:
```json
{
  "Normal": "a normal scene without any anomaly",
  "Abuse": "a video with abuse event",
  "Arrest": "a video with arrest event",
  ...
}
```

### 방법 2: JSON 파일로 커스텀 프롬프트 지정

#### Step 1: 커스텀 프롬프트 JSON 파일 준비

`custom_prompts.json` 파일 생성:

```json
{
  "Normal": "a normal video without any criminal activity or anomaly",
  "Abuse": "a person being abused or assaulted physically",
  "Arrest": "police officers making an arrest",
  "Arson": "fire or building burning",
  "Assault": "people fighting or assaulting each other",
  "Burglary": "someone breaking into a building or stealing items",
  "Explosion": "an explosion or bomb detonation",
  "Fighting": "violent fight between multiple people",
  "RoadAccidents": "car crash or traffic accident",
  "Robbery": "armed robbery or mugging",
  "Shooting": "gunshot or shooting incident",
  "Shoplifting": "person stealing items from a store",
  "Stealing": "person stealing property",
  "Vandalism": "property damage or vandalism"
}
```

#### Step 2: JSON 파일을 이용해 학습

```bash
python custom_prompt_training.py \
    --feature-dir /path/to/train/features \
    --val-feature-dir /path/to/val/features \
    --initial-prompts-file ./custom_prompts.json \
    --epochs 50 \
    --output-dir ./output/custom_prompts_v1
```

### 방법 3: 명령어 라인으로 프롬프트 지정

```bash
python custom_prompt_training.py \
    --feature-dir /path/to/train/features \
    --val-feature-dir /path/to/val/features \
    --custom-prompts \
        Normal "a normal scene" \
        Abuse "a physical attack" \
        Arrest "police making an arrest" \
    --epochs 50 \
    --output-dir ./output/custom_prompts_cli
```

---

## 프롬프트 엔지니어링 팁

### 프롬프트 작성 원칙

1. **구체적이고 설명적**: 추상적이 아닌 구체적인 설명 사용
   ```
   ❌ 나쁜 예: "abuse"
   ✓ 좋은 예: "a person being abused or assaulted physically"
   ```

2. **시각적 특징 포함**: 모델이 시각적으로 인식할 수 있는 요소
   ```
   ❌ 나쁜 예: "an emergency situation"
   ✓ 좋은 예: "emergency vehicles and flashing lights at accident scene"
   ```

3. **비디오 컨텍스트**: 이미지가 아닌 비디오임을 명시
   ```
   ❌ "a person being abused"
   ✓ "a video showing a person being abused"
   ```

4. **일관된 문법**: 모든 프롬프트를 비슷한 구조로 작성
   ```
   ✓ 좋은 예:
   - "a video of a normal scene without anomaly"
   - "a video of people fighting violently"
   - "a video of a robbery or theft"
   ```

### 클래스별 프롬프트 예시

```json
{
  "Normal": "a normal video without any anomaly, crime or unusual activity",

  "Abuse": "a video showing physical abuse, hitting, punching or violence toward a person",

  "Arrest": "a video of police officers arresting or apprehending a person",

  "Arson": "a video showing fire, flames or a building burning",

  "Assault": "a video of people attacking each other with weapons or physical force",

  "Burglary": "a video of someone breaking into a building, stealing or ransacking property",

  "Explosion": "a video showing an explosion, bomb detonation or blast",

  "Fighting": "a video of multiple people engaged in violent physical combat",

  "RoadAccidents": "a video of a car crash, traffic accident or vehicle collision",

  "Robbery": "a video of armed robbery, mugging or violent theft",

  "Shooting": "a video of a person shooting a gun or gunshot being fired",

  "Shoplifting": "a video of someone stealing items from a store or retail location",

  "Stealing": "a video of property theft, pickpocketing or stealing",

  "Vandalism": "a video showing property damage, graffiti or destruction of property"
}
```

---

## 실행 및 출력

### 전체 실행 예시

```bash
python custom_prompt_training.py \
    --feature-dir /mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/train \
    --val-feature-dir /mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/val \
    --initial-prompts-file ./custom_prompts.json \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.002 \
    --n-ctx 16 \
    --csc \
    --output-dir ./output/custom_prompts_v1 \
    --save-prompts
```

### 출력 파일 구조

```
./output/custom_prompts_v1/
├── initial_prompts.json          # 입력된 초기 프롬프트
├── learned_prompts.json          # 학습된 최종 프롬프트 벡터
├── checkpoints/
│   └── best_model.pth            # 최고 성능 모델 체크포인트
└── tensorboard/
    └── events.out.tfevents.*     # TensorBoard 로그
```

### initial_prompts.json

초기 프롬프트를 저장합니다:

```json
{
  "Normal": "a normal video without any anomaly, crime or unusual activity",
  "Abuse": "a video showing physical abuse, hitting, punching or violence toward a person",
  ...
}
```

### learned_prompts.json

학습된 최종 프롬프트 벡터를 저장합니다:

```json
{
  "Normal": {
    "context_vector": [
      [-0.0234, 0.0567, -0.0123, ...],  // [n_ctx, ctx_dim]
      [0.0145, -0.0234, 0.0567, ...],
      ...
    ],
    "context_shape": [16, 512],
    "full_prompt_embedding": [
      [0.0123, -0.0456, 0.0789, ...],   // [prefix + ctx + suffix, embed_dim]
      [0.0234, 0.0567, -0.0123, ...],
      ...
    ],
    "full_prompt_shape": [77, 512]
  },
  "Abuse": {
    ...
  }
}
```

**설명:**
- `context_vector`: 학습 가능한 컨텍스트 벡터 (shape: [n_ctx, ctx_dim])
  - n_ctx: 컨텍스트 토큰 개수 (기본값: 16)
  - ctx_dim: 텍스트 임베딩 차원 (MobileCLIP S0: 512)

- `full_prompt_embedding`: 완전한 프롬프트 임베딩
  - [prefix (1개) + context (n_ctx) + suffix (suffix_len)] × embedding_dim
  - 텍스트 인코더 출력값

- `full_prompt_shape`: [total_seq_len, embed_dim]

---

## 주요 인자 설명

| 인자 | 설명 | 기본값 | 필수 |
|------|------|--------|------|
| `--feature-dir` | 학습 비디오 피처 디렉토리 | - | ✓ |
| `--val-feature-dir` | 검증 비디오 피처 디렉토리 | - | ✓ |
| `--initial-prompts-file` | JSON 파일의 초기 프롬프트 | None | |
| `--custom-prompts` | 명령어 라인 프롬프트 | None | |
| `--save-initial-prompts` | 초기 프롬프트 저장 경로 | `{output_dir}/initial_prompts.json` | |
| `--mobileclip-model` | MobileCLIP 모델 | mobileclip2_s0 | |
| `--mobileclip-path` | MobileCLIP 경로 (선택) | None | |
| `--n-ctx` | 컨텍스트 토큰 개수 | 16 | |
| `--csc` | Class-specific context 사용 | True | |
| `--epochs` | 학습 에포크 수 | 50 | |
| `--batch-size` | 배치 크기 | 32 | |
| `--lr` | 학습률 | 0.002 | |
| `--num-workers` | 데이터 로더 워커 | 4 | |
| `--output-dir` | 출력 디렉토리 | ./output/custom_prompts | |
| `--save-prompts` | 최종 프롬프트 저장 | True | |

---

## 워크플로우

### 1단계: 프롬프트 설계

데이터셋의 클래스를 분석하고 각 클래스에 맞는 설명적 프롬프트 작성:

```json
{
  "Normal": "...",
  "Abuse": "...",
  ...
}
```

### 2단계: 모델 학습

커스텀 프롬프트로 CoOp 모델 학습:

```bash
python custom_prompt_training.py \
    --feature-dir ... \
    --val-feature-dir ... \
    --initial-prompts-file ./my_prompts.json \
    --epochs 50 \
    --output-dir ./output/my_training
```

**출력:**
- `initial_prompts.json`: 입력한 초기 프롬프트
- `learned_prompts.json`: 학습된 최종 프롬프트
- `checkpoints/best_model.pth`: 최고 성능 모델

### 3단계: 학습된 프롬프트 분석

생성된 `learned_prompts.json`에서:

```python
import json
import numpy as np

with open('./output/my_training/learned_prompts.json') as f:
    learned_prompts = json.load(f)

for classname, data in learned_prompts.items():
    ctx_shape = data['context_shape']  # [n_ctx, ctx_dim]
    ctx_vector = np.array(data['context_vector'])
    print(f"{classname}: context_vector shape = {ctx_shape}")
    print(f"  Mean magnitude: {np.linalg.norm(ctx_vector, axis=1).mean():.4f}")
```

### 4단계: 평가

학습된 모델을 테스트셋에서 평가:

```bash
python evaluate_video_feature_coop.py \
    --test-feature-dir /path/to/test \
    --checkpoint-path ./output/my_training/checkpoints/best_model.pth \
    --n-ctx 16 \
    --csc \
    --output-dir ./output/my_evaluation
```

---

## 성능 최적화

### 프롬프트 개선

초기 학습 후 결과 분석:

1. **낮은 성능을 보인 클래스** 확인
2. 해당 클래스 프롬프트 개선
3. 새 프롬프트로 재학습

```bash
# v1: 초기 시도
python custom_prompt_training.py ... --output-dir ./output/v1

# v1 결과 분석 후 프롬프트 개선
nano custom_prompts_v2.json

# v2: 개선된 프롬프트
python custom_prompt_training.py \
    ... \
    --initial-prompts-file ./custom_prompts_v2.json \
    --output-dir ./output/v2
```

### 하이퍼파라미터 조정

```bash
# Learning rate 증가 (빠른 수렴)
python custom_prompt_training.py ... --lr 0.005

# 더 많은 컨텍스트 토큰 (표현력 증가)
python custom_prompt_training.py ... --n-ctx 32

# 더 긴 학습 (수렴 개선)
python custom_prompt_training.py ... --epochs 100
```

---

## 예제: 커스텀 vs 기본 프롬프트 비교

### 기본 프롬프트로 학습

```bash
python custom_prompt_training.py \
    --feature-dir /path/to/train \
    --val-feature-dir /path/to/val \
    --epochs 50 \
    --output-dir ./output/baseline
```

**결과:** `./output/baseline/initial_prompts.json`
```json
{
  "Normal": "a normal scene without any anomaly",
  "Abuse": "a video with abuse event",
  ...
}
```

### 커스텀 프롬프트로 학습

```bash
python custom_prompt_training.py \
    --feature-dir /path/to/train \
    --val-feature-dir /path/to/val \
    --initial-prompts-file ./detailed_prompts.json \
    --epochs 50 \
    --output-dir ./output/custom
```

**결과:** `./output/custom/initial_prompts.json`
```json
{
  "Normal": "a normal video without any anomaly, crime or unusual activity",
  "Abuse": "a video showing physical abuse, hitting, punching or violence toward a person",
  ...
}
```

### 성능 비교

```python
import json

# Baseline 학습된 프롬프트
with open('./output/baseline/learned_prompts.json') as f:
    baseline = json.load(f)

# Custom 학습된 프롬프트
with open('./output/custom/learned_prompts.json') as f:
    custom = json.load(f)

# 초기 프롬프트 길이 비교
for cls in ['Normal', 'Abuse']:
    baseline_len = len(baseline[cls]['context_vector'])
    custom_len = len(custom[cls]['context_vector'])
    print(f"{cls}: baseline ctx={baseline_len}, custom ctx={custom_len}")
```

---

## 문제 해결

### 1. "프롬프트가 모든 클래스를 포함하지 않음"

```bash
python custom_prompt_training.py \
    --initial-prompts-file ./incomplete_prompts.json \
    ...
```

**해결:**
```
⚠ Warning: Missing prompts for classes: ['Arrest', 'Arson']
Adding default prompts...
```

누락된 클래스는 자동으로 기본 프롬프트가 추가됩니다.

### 2. "메모리 부족 (OOM)"

```bash
# 배치 크기 감소
python custom_prompt_training.py ... --batch-size 16
```

### 3. "학습이 수렴하지 않음"

```bash
# Learning rate 감소
python custom_prompt_training.py ... --lr 0.001

# 더 많은 에포크
python custom_prompt_training.py ... --epochs 100
```

---

## 다음 단계

1. **프롬프트 준비** (이 가이드의 팁 참조)
2. **모델 학습** (커스텀 프롬프트 사용)
3. **학습된 프롬프트 분석** (learned_prompts.json 검토)
4. **모델 평가** (evaluate_video_feature_coop.py 또는 evaluate_with_temporal_gt.py)
5. **결과 기반 반복** (필요시 프롬프트 개선)

---

## 참고

- **Prompt Learning**: CoOp (Context Optimization) 방식으로 학습 가능한 컨텍스트 벡터를 사전 학습된 CLIP 텍스트 인코더의 임베딩 공간에서 최적화합니다.
- **초기 프롬프트의 역할**: 좋은 초기 프롬프트는 수렴을 빠르게 하고 최종 성능을 향상시킵니다.
- **Context Tokens**: 학습 가능한 토큰 개수 (n_ctx)는 모델의 표현력과 계산량의 트레이드오프입니다.
  - n_ctx=16 (기본): 빠른 학습, 일반적 성능
  - n_ctx=32: 더 나은 표현력, 더 많은 계산량
