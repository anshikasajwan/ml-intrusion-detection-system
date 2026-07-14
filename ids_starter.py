"""
ML-Based Intrusion Detection System — Starter Script
Dataset: NSL-KDD (download KDDTrain+.txt and KDDTest+.txt first)
  Source: https://www.unb.ca/cic/datasets/nsl.html  (or search "NSL-KDD" on Kaggle)

This script:
  1. Loads and preprocesses NSL-KDD
  2. Trains a supervised Decision Tree to classify known attack types
  3. Trains an unsupervised K-Means model on "normal" traffic only,
     to flag anomalies (a proxy for catching unknown/zero-day attacks)
  4. Evaluates both with proper metrics (not just accuracy)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------------
# 1. Column names (NSL-KDD has no header row in the raw files)
# ---------------------------------------------------------------------
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Map fine-grained attack names to the 4 standard NSL-KDD categories
ATTACK_CATEGORIES = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe",
    "portsweep": "probe", "mscan": "probe", "saint": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l",
    "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
    "warezclient": "r2l", "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "snmpguess": "r2l", "snmpgetattack": "r2l", "httptunnel": "r2l",
    "sendmail": "r2l", "named": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}


def load_data(path):
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    # strip trailing "." some versions of the file have on labels
    df["label"] = df["label"].str.strip(".")
    df["category"] = df["label"].map(ATTACK_CATEGORIES).fillna("unknown")
    return df


def preprocess(train_df, test_df):
    """Encode categoricals and scale numerics. Fit only on train, apply to both."""
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train_df[col])
        # handle unseen categories in test set gracefully
        train_df[col] = le.transform(train_df[col])
        test_df[col] = test_df[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    feature_cols = [c for c in COLUMNS if c not in ("label", "difficulty")]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    return X_train, X_test, feature_cols, scaler


def run_supervised(X_train, y_train, X_test, y_test):
    print("\n=== Supervised: Decision Tree (known attack classification) ===")
    clf = DecisionTreeClassifier(max_depth=12, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report (per category):")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    return clf


def run_anomaly_detection(X_train, train_category, X_test, y_test_binary):
    """
    Train K-Means ONLY on normal traffic. At inference, measure distance
    to nearest cluster centroid; large distance = anomaly = potential
    unknown/zero-day attack.
    """
    print("\n=== Unsupervised: K-Means anomaly detection (zero-day proxy) ===")
    normal_mask = train_category == "normal"
    X_normal = X_train[normal_mask]

    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(X_normal)

    # distance of each test point to its nearest centroid
    distances = np.min(kmeans.transform(X_test), axis=1)

    # threshold = e.g. 95th percentile of distances seen on normal training data
    train_distances = np.min(kmeans.transform(X_normal), axis=1)
    threshold = np.percentile(train_distances, 95)

    preds_anomaly = (distances > threshold).astype(int)  # 1 = flagged as attack

    print(f"Distance threshold (95th pct of normal): {threshold:.3f}")
    print("Precision:", precision_score(y_test_binary, preds_anomaly))
    print("Recall:", recall_score(y_test_binary, preds_anomaly))
    print("F1:", f1_score(y_test_binary, preds_anomaly))
    print("Confusion matrix:\n", confusion_matrix(y_test_binary, preds_anomaly))
    return kmeans, threshold


if __name__ == "__main__":
    # Update these paths to wherever you downloaded the NSL-KDD files
    train_df = load_data("KDDTrain+.txt")
    test_df = load_data("KDDTest+.txt")

    X_train, X_test, feature_cols, scaler = preprocess(train_df, test_df)

    # --- Supervised task: predict attack category ---
    y_train_cat = train_df["category"]
    y_test_cat = test_df["category"]
    run_supervised(X_train, y_train_cat, X_test, y_test_cat)

    # --- Unsupervised task: normal vs anomaly (binary) ---
    y_test_binary = (test_df["category"] != "normal").astype(int)
    run_anomaly_detection(X_train, y_train_cat, X_test, y_test_binary)