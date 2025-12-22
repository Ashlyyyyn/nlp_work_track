# Chinese Genre Classification (Course Project)

Compare:
- TF-IDF (char n-gram) + Logistic Regression
- Word2Vec (char) mean pooling + Logistic Regression

## Setup
pip install -r requirements.txt

## Data
Put `data/dataset.csv` with columns: text,label

## Run
python src/train_tfidf_lr.py
python src/interpret_tfidf.py

python src/train_w2v_lr.py
python src/interpret_w2v.py

python src/train_tfidf_svm.py
python src/analyze_dataset.py
python src/error_analysis_tfidf.py
python src/compare_results.py
python src/train_tfidf_nb.py
python src/eval_confusion.py --model models/tfidf_lr.joblib
python src/predict_texts.py --model models/tfidf_lr.joblib --text "示例文本"
python src/kfold_tfidf_lr.py --k 5
python src/tfidf_top_features.py --model models/tfidf_lr.joblib --topn 30
python src/length_bucket_analysis.py --model models/tfidf_lr.joblib

Outputs:
- models/...
- results/...
