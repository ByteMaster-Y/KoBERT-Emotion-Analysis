# KoBERT 영화 리뷰 감정 분석 프로젝트

한국어 영화 리뷰 데이터를 기반으로 KoBERT 모델을 활용한 감정 분석 프로젝트입니다.  
긍정/부정 리뷰를 자동으로 분류하며, 학습 과정과 예측 테스트를 포함합니다.

## 데이터셋 출처
본 프로젝트에서 사용한 데이터셋은 [e9t/nsmc GitHub 저장소](https://github.com/e9t/nsmc)에서 제공하는 NSMC(Naver Sentiment Movie Corpus)입니다. 이 데이터셋은 네이버 영화 리뷰를 기반으로 한 감정 분석용 한국어 데이터셋으로, `ratings_train.txt`와 `ratings_test.txt` 파일을 포함하고 있습니다.

---

## 프로젝트 개요

- **목표**: 영화 리뷰 데이터를 분석하여 감정을 분류하고, KoBERT 모델을 활용한 한국어 NLP 경험 습득
- **주요 기능**:
  - 한국어 리뷰 전처리 및 정제
  - KoBERT 기반 이진 분류 모델 학습
  - 학습된 모델로 입력 문장 감정 예측

---

## 데이터

- **데이터 출처**: NSMC(Naver Sentiment Movie Corpus)  
- **파일**:
  - `ratings_train.txt` (학습 데이터, 150,001개)
  - `ratings_test.txt` (테스트 데이터, 50,001개)
- **열(column)**: `document` (리뷰 내용), `label` (0=부정, 1=긍정)

## 데이터 로드 및 전처리

### 데이터셋 로드
- `datasets` 라이브러리의 `load_dataset` 사용
- 로컬 파일(`ratings_train.txt`, `ratings_test.txt`) 불러오기
- 파일은 탭(`\t`)으로 구분, 열 이름: `document`, `label`

```python
from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files={
        "train": "ratings_train.txt",
        "test": "ratings_test.txt"
    },
    delimiter="\t",
    column_names=["document", "label"]
)

```

---

### 데이터 전처리  

KoBERT 모델 학습을 위해 원본 데이터를 **클린업(정제) → 라벨 처리 → 토크나이징** 순서로 전처리했습니다.  

---

### 1. 텍스트 정제 (Cleaning)  
- `document` 값이 `None`이거나 문자열이 아닐 경우 빈 문자열(`""`)로 변환  
- 모든 입력을 문자열(`str`) 타입으로 통일  

```python
def clean_text(example):
    text = example["document"]
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    example["document"] = text
    return example

```

### 2. 라벨 정제 (Fixing Labels)

- 라벨(label)을 정수형(int)으로 변환
- 리스트나 문자열 형태로 들어온 경우에도 처리
- 변환이 불가능할 경우 기본값 0으로 설정

```python
def fix_labels(batch):
    labels = batch["label"]
    new_labels = []
    for l in labels:
        if isinstance(l, list):
            new_labels.append(int(l[0]))
        else:
            try:
                new_labels.append(int(l))
            except:
                new_labels.append(0)
    batch["label"] = new_labels
    return batch

```

### 3. 토크나이징

- **Hugging Face KoBERT 토크나이저** 사용  
- 입력 문장을 모델이 처리할 수 있도록 **토큰 단위로 변환**  
- 길이 통일 및 잘림 방지를 위해 다음 옵션 적용:  
  - `padding="max_length"` → 모든 입력을 동일 길이로 패딩  
  - `truncation=True` → 문장이 길면 잘라냄  
  - `max_length=64` → 학습 효율성을 고려한 입력 길이  

```python
def tokenize_function(batch):
    return tokenizer(
        batch["document"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

```

----

## 모델

- **모델명**: `skt/kobert-base-v1` (KoBERT)
- **분류 목적**: 긍정 / 부정
- **토크나이징**:
  - `max_length=64`, `padding=max_length`, `truncation=True`
- **데이터 Collator**:
  - `DataCollatorWithPadding` 사용 → 배치 내 문장 길이 다를 때 패딩 처리

---

## 학습 환경 및 설정

- **학습 환경**: MacBook M1 (MPS GPU), PyTorch
- **배치 사이즈**: 8
- **Epoch**: 2
- **Optimizer**: AdamW, Learning rate=5e-5
- **Weight Decay**: 0.01
- **FP16 사용 여부**: False (MPS는 FP16 미지원)
- **로그**:
  - 학습 진행 시 Loss, Gradient Norm, Learning Rate, Epoch 표시

---

## 평가 지표

- **Accuracy**: 정확도
- **F1 score**: 불균형 데이터셋에서 정밀도와 재현율 균형 평가
- **평가 과정**: Trainer의 `compute_metrics`를 사용하여 검증 데이터셋 평가

---

## Training Log Visualization

<img width="1200" height="688" alt="Image" src="https://github.com/user-attachments/assets/c1f63d11-381b-4f30-b43c-d57c61e3b9dd" />


---

## 최종 평가 지표

```
eval_loss: 0.6922
eval_accuracy: 0.5394
eval_f1: 0.6409
```

<img width="979" height="580" alt="Image" src="https://github.com/user-attachments/assets/86b57296-538b-4bb6-bf82-a26297f1e6b6" />

---

## 사용 예시
<img width="1489" height="63" alt="Image" src="https://github.com/user-attachments/assets/521b3386-f6c2-418f-84b6-32b11028834d" />

### 학습

```zsh
python koBERT_example.py
