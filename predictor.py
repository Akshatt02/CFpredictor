import sys
import requests
import joblib
import pandas as pd
import numpy as np

def fetch_user_data(handle, max_submissions=1000, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(f"https://codeforces.com/api/user.rating?handle={handle}", timeout=10)
            r.raise_for_status()
            js = r.json()
            if js["status"] != "OK" or not js["result"]:
                raise ValueError(f"Could not fetch rating history for '{handle}'.")

            history = [entry["newRating"] for entry in js["result"]]

            r2 = requests.get(f"https://codeforces.com/api/user.status?handle={handle}&count={max_submissions}", timeout=10)
            r2.raise_for_status()
            js2 = r2.json()
            if js2["status"] != "OK":
                raise ValueError("Could not fetch submission history.")

            subs = js2["result"]

            total_subs, ac_subs, tag_sub, tag_ac = 0, 0, {}, {}
            for s in subs:
                tags = s.get("problem", {}).get("tags", [])
                verdict = s.get("verdict", "")
                if not tags:
                    continue
                total_subs += 1
                for t in tags:
                    tag_sub[t] = tag_sub.get(t, 0) + 1
                    if verdict == "OK":
                        tag_ac[t] = tag_ac.get(t, 0) + 1
                        ac_subs += 1

            ac_ratio = ac_subs / total_subs if total_subs else 0

            return {
                "handle": handle,
                "total_submissions": total_subs,
                "ac_submissions": ac_subs,
                "ac_ratio": ac_ratio,
                "current_rating": history[-1] if history else 0,
                "rating_history": history,
                "tag_submission_count": tag_sub,
                "tag_ac_count": tag_ac
            }

        except (requests.RequestException, ValueError) as e:
            print(f"Attempt {attempt+1}: Error fetching data - {e}")
            if attempt == retries - 1:
                print("Exceeded retry limit. Exiting...")
                sys.exit(1)
            handle = input("Try again with a valid Codeforces handle: ").strip()

def create_user_profile(user_data):
    history = user_data.get("rating_history", [])
    contest_count = len(history)

    rating_mean = np.mean(history) if contest_count else 0
    rating_std = np.std(history) if contest_count > 1 else 0
    rating_trend = (history[-1] - history[0]) if contest_count > 1 else 0

    avg_subs_per_contest = user_data["total_submissions"] / contest_count if contest_count else 0
    avg_ac_per_contest = user_data["ac_submissions"] / contest_count if contest_count else 0

    tag_diversity = len(user_data.get("tag_submission_count", {}))
    tag_success_diversity = len([t for t, v in user_data.get("tag_ac_count", {}).items() if v > 0])

    profile = {
        "total_submissions": user_data["total_submissions"],
        "ac_submissions": user_data["ac_submissions"],
        "ac_ratio": user_data["ac_ratio"],
        "contest_count": contest_count,
        "rating_mean": rating_mean,
        "rating_std": rating_std,
        "rating_trend": rating_trend,
        "avg_subs_per_contest": avg_subs_per_contest,
        "avg_ac_per_contest": avg_ac_per_contest,
        "tag_diversity": tag_diversity,
        "tag_success_diversity": tag_success_diversity
    }

    for i in range(30):
        profile[f"rating_t{i+1}"] = history[i] if i < len(history) else 0

    return profile

def main():
    handle = input("Enter Codeforces handle: ").strip()
    user_data = fetch_user_data(handle)

    profile = create_user_profile(user_data)
    df = pd.DataFrame([profile])

    try:
        scaler = joblib.load("models/scaler.pkl")
        model = joblib.load("models/rating_predictor.pkl")
        expected_features = scaler.feature_names_in_
        df = df.reindex(columns=expected_features, fill_value=0)

        X_scaled = pd.DataFrame(scaler.transform(df), columns=expected_features)

        predicted_rating = model.predict(X_scaled)[0]

        print(f"\nPredicted rating for {handle} after 5 more contests: {predicted_rating:.0f}")
        print(f"Current rating: {user_data['current_rating']}")
        print(f"Expected change: {predicted_rating - user_data['current_rating']:.0f}")

    except FileNotFoundError as e:
        print(f"Error loading model or scaler: {e}")
        print("Ensure models/scaler.pkl and models/rating_predictor.pkl exist.")
        sys.exit(1)

if __name__ == "__main__":
    main()