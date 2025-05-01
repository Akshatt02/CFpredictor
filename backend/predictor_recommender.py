import json
import requests
import joblib
import numpy as np
import pandas as pd
import random
from collections import Counter

BASE_URL = "https://codeforces.com/api"

scaler = joblib.load("models/scaler.pkl")
model  = joblib.load("models/rating_predictor.pkl")

def fetch_user_data(handle, max_submissions=1000):
    r = requests.get(f"{BASE_URL}/user.rating?handle={handle}")
    js = r.json()
    if js.get("status") != "OK":
        raise ValueError("Could not fetch ratings.")
    history = [c["newRating"] for c in js["result"]][-30:]

    r2 = requests.get(f"{BASE_URL}/user.status?handle={handle}&from=1&count={max_submissions}")
    js2 = r2.json()
    if js2.get("status") != "OK":
        raise ValueError("Could not fetch submissions.")
    subs = js2["result"]

    total_subs = len(subs)
    ac_subs = sum(1 for s in subs if s.get("verdict") == "OK")

    tag_sub_count = Counter()
    tag_ac_count  = Counter()
    for s in subs:
        tags = s.get("problem", {}).get("tags", [])
        if not tags:
            continue
        for t in tags:
            tag_sub_count[t] += 1
            if s.get("verdict") == "OK":
                tag_ac_count[t] += 1

    return {
        "handle": handle,
        "total_submissions": total_subs,
        "ac_submissions": ac_subs,
        "ac_ratio": ac_subs / total_subs if total_subs else 0,
        "current_rating": history[-1] if history else 0,
        "rating_history": history,
        "tag_submission_count": dict(tag_sub_count),
        "tag_ac_count": dict(tag_ac_count),
        "submissions": subs
    }

def extract_all_tags(data):
    tags = set()
    for u in data:
        tags.update(u["tag_submission_count"].keys())
        tags.update(u["tag_ac_count"].keys())
    return sorted(tags)

def create_user_profile(u, all_tags):
    hist = u.get("rating_history", [])
    ct = len(hist)

    profile = {
        "total_submissions": u["total_submissions"],
        "ac_submissions": u["ac_submissions"],
        "ac_ratio": u["ac_ratio"],
        "contest_count": ct,
        "rating_range": (max(hist) - min(hist)) if ct > 1 else 0,
        "rating_trend": (hist[-1] - hist[0]) if ct > 1 else 0,
        "rating_mean": np.mean(hist) if ct else 0,
        "rating_std": np.std(hist) if ct else 0,
        "avg_subs_per_contest": u["total_submissions"] / ct if ct else 0,
        "avg_ac_per_contest": u["ac_submissions"] / ct if ct else 0,
        "tag_diversity": len(u["tag_submission_count"]),
        "tag_success_diversity": len([t for t, v in u["tag_ac_count"].items() if v > 0]),
        "top_tag_sub_ratio": (max(u["tag_submission_count"].values()) / u["total_submissions"]) if u["total_submissions"] else 0
    }

    for i in range(30):
        profile[f"rating_t{i+1}"] = hist[i] if i < ct else 0

    for tag in all_tags:
        profile[f"{tag}_subs"] = u["tag_submission_count"].get(tag, 0)
        profile[f"{tag}_ac"] = u["tag_ac_count"].get(tag, 0)

    return profile

def predict_from_profile(user):
    with open("data/training_data.json", "r") as f:
        train_data = json.load(f)
    all_tags = extract_all_tags(train_data)

    profile = create_user_profile(user, all_tags)
    df = pd.DataFrame([profile])

    X_scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    pred = model.predict(X_scaled_df)[0]

    rating_change = round(pred - user["current_rating"], 2)

    return {
        "predicted_rating": round(pred),
        "rating_change": round(rating_change),
    }

def recommend_problems(user_data, predicted_rating, count=10):
    weak_tags = sorted(user_data['tag_ac_count'],key=lambda tag: user_data['tag_ac_count'][tag] / user_data['tag_submission_count'].get(tag, 1))[:3]

    ignored_tags = {"special problem", "interactive", "implementation"}

    solved_set = set()
    for s in user_data.get("submissions", []):
        if s.get("verdict") == "OK":
            p = s["problem"]
            solved_set.add(f"{p['contestId']}-{p['index']}")

    problems = []
    for tag in weak_tags:
        if tag in ignored_tags:
            continue
        try:
            res = requests.get(f"{BASE_URL}/problemset.problems?tags={tag}").json()
            if res['status'] != 'OK':
                continue

            for problem, stats in zip(res['result']['problems'], res['result']['problemStatistics']):
                prob_id = f"{problem['contestId']}-{problem['index']}"
                if prob_id in solved_set:
                    continue
                if any(t in ignored_tags for t in problem['tags']):
                    continue

                rating = problem.get('rating', None)
                if not rating:
                    continue

                if predicted_rating >= 3000:
                    if rating >= 3000:
                        problems.append(make_problem_dict(problem))
                else:
                    if 0 <= rating - predicted_rating <= 200:
                        problems.append(make_problem_dict(problem))
        except:
            continue

    random.shuffle(problems)
    return problems[:count]

def make_problem_dict(problem):
    return {
        "name": problem['name'],
        "contestId": problem['contestId'],
        "index": problem['index'],
        "rating": problem.get('rating', '?'),
        "tags": problem['tags'],
        "url": f"https://codeforces.com/contest/{problem['contestId']}/problem/{problem['index']}"
    }