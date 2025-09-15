from load_dataset import dataset  
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import matplotlib.pyplot as plt
import pandas as pd
torch.mps.empty_cache()

# 1. GPU(MPS) 사용 확인
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# 2. 모델과 토크나이저 로드
model_name = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 3. 데이터 전처리
def clean_text(example):
    text = example["document"]
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    example["document"] = text
    return example

def fix_labels(batch):
    # batch 단위 처리 (batched=True)
    labels = batch["label"]
    new_labels = []
    for l in labels:
        if isinstance(l, list):
            new_labels.append(int(l[0]))
        else:
            try:
                new_labels.append(int(l))
            except:
                # 혹시 문자열이면 0으로 처리
                new_labels.append(0)
    batch["label"] = new_labels
    return batch


dataset = dataset.map(clean_text)
dataset = dataset.map(fix_labels, batched=True)  # batched=True 필수!

# 4. 토크나이징
def tokenize_function(batch):
    return tokenizer(
        batch["document"],
        padding="max_length",  # max_length로 패딩
        truncation=True,      # 길이 자르기
        max_length=64
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# 5. 평가 지표 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1) # axis=-1 은 마지막 축(여기서는 클래스 축)을 의미
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# 6. 학습 설정 (MPS는 fp16 지원 안 됨)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="no",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=False,  # MPS는 fp16 미지원
)

# 7. DataCollator 추가, 배치 내 문장 길이가 다를 때 패딩(Padding) 처리
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 8. Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.shuffle(seed=42).select(range(20000)),
    eval_dataset=test_dataset.select(range(5000)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# # 9. 학습
# trainer.train()

# # 10. 예측 테스트
# sample_text = "이 영화 진짜 재미있고 웃기면서 감동적이기 까지 하더라"
# inputs = tokenizer(
#     sample_text,
#     return_tensors="pt",
#     truncation=True,
#     padding="max_length",
#     max_length=64
# ).to(device)

# outputs = model(**inputs) # ** → 딕셔너리 풀기 → 키=값 형태로 함수에 전달
# pred = outputs.logits.argmax(-1).item()
# print("입력 문장:", sample_text)
# print("예측 결과:", "긍정" if pred == 1 else "부정")


# 9. 학습
trainer.train()

# 9-1. 학습 끝난 후 최종 성능 평가 & 그래프 저장
eval_results = trainer.evaluate()
print("최종 평가 결과:", eval_results)

log_history = trainer.state.log_history
df = pd.DataFrame(log_history)

plt.figure(figsize=(10,6))
if "loss" in df.columns:
    plt.plot(df["step"], df["loss"], label="Train Loss")
if "eval_loss" in df.columns:
    plt.plot(df["step"], df["eval_loss"], label="Eval Loss")
if "eval_accuracy" in df.columns:
    plt.plot(df["step"], df["eval_accuracy"], label="Eval Accuracy")
if "eval_f1" in df.columns:
    plt.plot(df["step"], df["eval_f1"], label="Eval F1")

plt.xlabel("Steps")
plt.ylabel("Metric Value")
plt.title("Training & Evaluation Metrics")
plt.legend()
plt.grid(True)
plt.savefig("training_results.png")
plt.show()

# 10. 예측 테스트
sample_text = "이 영화 진짜 재미있고 웃기면서 감동적이기 까지 하더라"
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=64
).to(device)

outputs = model(**inputs) # ** → 딕셔너리 풀기 → 키=값 형태로 함수에 전달
pred = outputs.logits.argmax(-1).item()
print("입력 문장:", sample_text)
print("예측 결과:", "긍정" if pred == 1 else "부정")
