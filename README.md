# Archaeological Period Classifier

Predicts the cultural period (**Ancient** / **Medieval** / **Modern**) of archaeological artifacts from Metropolitan Museum of Art (Met) metadata using machine learning.

## Overview

Manual classification of archaeological artifacts is time-consuming and prone to expert subjectivity. This project applies supervised machine learning to digital heritage metadata to automate period classification while flagging low-confidence predictions for expert review.

## Dataset

- **Full dataset**: https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv (448,203 rows x 54 columns, ~300 MB - not included in this repo)
- **Sample**: data/sample_metobjects.csv - first 500 rows, for pipeline testing

Download the full dataset and place it at data/MetObjects.csv before running the full notebook.

## Methodology

1. **Data Cleaning** - Select 6 key columns; lowercase; fill missing with Unknown.
2. **Temporal Mapping** - Convert Object Date to a target class: Ancient (year < 500), Medieval (500 <= year < 1500), Modern (year >= 1500).
3. **Feature Engineering** - Concatenate text columns; TF-IDF (500 features).
4. **Dimensionality Reduction** - PCA (50 components).
5. **Modeling** - Random Forest, SVM, KNN, Logistic Regression + Stacking Ensemble.
6. **Evaluation** - Cross-validation, Precision-Recall curves, decision boundaries.
7. **Unique Contribution** - Confidence-based classification: High confidence (> 90%) auto-catalog; Low confidence (< 70%) flag for expert review.

## Results

| Metric | Value |
|---|---|
| Best model accuracy | ~95% |
| Cross-validation average | ~94.7% |
| High-confidence coverage | ~91% of artifacts |
| Accuracy on high-confidence | ~97% |

## Reproduce

    pip install -r requirements.txt
    jupyter notebook notebooks/archaeological-period-classifier.ipynb

## Citation

N. Potdukhe, I. Bhagat, K. Agre, P. Meshram, N. Akmar, "Predicting cultural period of archaeological artefacts from digital heritage metadata by automated machine learning approach", Department of AI and ML, Government Polytechnic, Nagpur.

## License

MIT
