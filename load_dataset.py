from datasets import load_dataset

# 로컬 CSV/TSV 파일 불러오기
dataset = load_dataset(
    "csv",
    data_files={
        "train": "ratings_train.txt",
        "test": "ratings_test.txt"
    },
    delimiter="\t",                # NSMC 파일은 탭으로 구분됨, 텍스트 데이터를 분리하는 구분자
    column_names=["document", "label"]  # 열 이름 지정
)

print(dataset)
