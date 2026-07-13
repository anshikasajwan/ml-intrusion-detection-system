# ML-Based Intrusion Detection System (IDS)

A machine learning system that analyzes network traffic patterns to detect
intrusions — including previously unseen (zero-day) attacks — using the
NSL-KDD benchmark dataset.

## Overview

This project explores two complementary ML approaches to network intrusion
detection:

- **Supervised classification** (Decision Tree) — trained on labeled traffic
  to recognize known attack categories (DoS, Probe, R2L, U2R).
- **Unsupervised anomaly detection** (K-Means) — trained only on normal
  traffic, flagging statistical outliers as potential unknown/zero-day
  attacks.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — an improved version of
the KDD Cup 1999 dataset with duplicate records removed to avoid biased
results. Each record has 41 network traffic features (duration, protocol
type, byte counts, error rates, etc.) and a label.

| | Normal | DoS | Probe | R2L | U2R |
|---|---|---|---|---|---|
| Train | 67,343 | 45,927 | 11,656 | 995 | 52 |
| Test | 9,711 | 7,458 | 2,421 | 2,754 | 200 |

Download: `KDDTrain+.txt` and `KDDTest+.txt` — see [Setup](#setup) below.

## Project structure

```
├── data/                  # KDDTrain+.txt, KDDTest+.txt (not committed — see .gitignore)
├── ids_starter.py         # main script: preprocessing, training, evaluation
├── requirements.txt
└── README.md
```

## Setup

```bash
# clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# install dependencies
pip install -r requirements.txt

# download the dataset into data/
curl -L -o data/KDDTrain+.txt "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt"
curl -L -o data/KDDTest+.txt "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt"

# run
python ids_starter.py
```

## Methodology

1. **Preprocessing** — categorical features (`protocol_type`, `service`,
   `flag`) label-encoded; numeric features standardized.
2. **Supervised model** — Decision Tree classifies traffic into
   normal / DoS / Probe / R2L / U2R.
3. **Unsupervised model** — K-Means fitted on normal traffic only;
   distance-to-nearest-centroid above a threshold flags anomalies.
4. **Evaluation** — precision, recall, F1, and confusion matrices per
   category (accuracy alone is misleading on this imbalanced dataset).



## Future work

- Compare against Random Forest, Isolation Forest, One-Class SVM
- Try a neural network baseline in TensorFlow
- Feature importance analysis
- ROC / precision-recall curves

## References

- M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, "A Detailed Analysis of the
  KDD CUP 99 Data Set," IEEE CISDA, 2009.
