import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="data/training_data.json"):
    with open(path, "r") as f:
        return json.load(f)

def extract_all_tags(data):
    tags = set()
    for u in data:
        tags.update(u["tag_submission_count"].keys())
        tags.update(u["tag_ac_count"].keys())
    return sorted(tags)

def create_user_profile(u, all_tags):
    total_subs = u["total_submissions"]
    ac_subs    = u["ac_submissions"]
    history    = u.get("rating_history", [])
    contest_ct = len(history)
    rating_min   = min(history) if history else 0
    rating_max   = max(history) if history else 0
    rating_first = history[0] if history else 0
    rating_last  = history[-1] if history else 0

    profile = {
        "handle": u["handle"],
        "total_submissions": total_subs,
        "ac_submissions": ac_subs,
        "ac_ratio": ac_subs / total_subs if total_subs else 0,
        "current_rating": u["current_rating"],
        "contest_count": contest_ct,
        "rating_range": rating_max - rating_min,
        "rating_trend": rating_last - rating_first,
        "rating_mean": np.mean(history) if history else 0,
        "rating_std":  np.std(history)  if history else 0,
        "avg_subs_per_contest": total_subs / contest_ct if contest_ct else 0,
        "avg_ac_per_contest":   ac_subs    / contest_ct if contest_ct else 0,
        "tag_diversity":         len(u["tag_submission_count"]),
        "tag_success_diversity": len([t for t,v in u["tag_ac_count"].items() if v>0]),
        "top_tag_sub_ratio": (
            max(u["tag_submission_count"].values(), default=0) / total_subs
            if total_subs else 0
        ),
    }

    for i in range(30):
        profile[f"rating_t{i+1}"] = history[i] if i < len(history) else 0

    for tag in all_tags:
        profile[f"{tag}_subs"] = u["tag_submission_count"].get(tag, 0)
        profile[f"{tag}_ac"]   = u["tag_ac_count"].get(tag, 0)

    return profile

def main():
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    data = load_data()
    tags = extract_all_tags(data)
    profiles = [create_user_profile(u, tags) for u in data]
    df = pd.DataFrame(profiles)

    X = df.drop(columns=["handle", "current_rating"])
    y = df["current_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

    X_train_scaled.to_csv("data/X_train.csv", index=False)
    X_test_scaled .to_csv("data/X_test.csv",  index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv",  index=False)

    joblib.dump(scaler, "models/scaler.pkl")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
