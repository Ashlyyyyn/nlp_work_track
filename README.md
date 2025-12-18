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

Outputs:
- models/...
- results/...