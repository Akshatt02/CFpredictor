import requests
import time
import json
import random
import os
from tqdm import tqdm
from collections import defaultdict

BASE_URL = "https://codeforces.com/api"

def fetch_info(handle):
    try:
        subs_res = requests.get(f"{BASE_URL}/user.status?handle={handle}&from=1&count=800").json()
        if subs_res['status'] != 'OK':
            return None
        time.sleep(2)

        submissions = subs_res['result']
        ac_count = sum(1 for s in submissions if s.get('verdict') == 'OK')

        submission_tags = []
        tag_submission_count = defaultdict(int)
        tag_ac_count = defaultdict(int)

        for submission in submissions:
            tags = submission.get('problem', {}).get('tags', [])
            for tag in tags:
                tag_submission_count[tag] += 1
                if submission.get('verdict') == 'OK':
                    tag_ac_count[tag] += 1
            submission_tags.append(tags)

        ratings_res = requests.get(f"{BASE_URL}/user.rating?handle={handle}").json()
        if ratings_res['status'] == 'OK':
            ratings = [c['newRating'] for c in ratings_res['result'][-30:]]

        return {
            "handle": handle,
            "total_submissions": len(submissions),
            "ac_submissions": ac_count,
            "ac_ratio": round(ac_count / len(submissions), 3) if submissions else 0,
            "current_rating": ratings[len(ratings) - 1] if ratings else None,
            "rating_history": ratings,
            "tag_submission_count": dict(tag_submission_count),
            "tag_ac_count": dict(tag_ac_count)
        }
    except Exception:
        return None

def fetch_users(low, high, limit=100):
    users = []
    offset = 1
    while len(users) < limit:
        url = f"{BASE_URL}/user.ratedList?activeOnly=true&from={offset}&count=100"
        res = requests.get(url).json()
        if res['status'] != 'OK':
            raise Exception(f"Error fetching users in range {low}-{high}")
        for user in res['result']:
            if low <= user['rating'] < high:
                users.append(user['handle'])
        if len(users) >= limit:
            break
        offset += 100
        time.sleep(2)
    return users[:limit]

def scrape_users(save_path="data/training_data.json"):
    all_data = []
    seen_handles = set()
    
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                all_data = json.load(f)
                seen_handles = {entry["handle"] for entry in all_data}
                print(f"Loaded {len(all_data)} existing users.")
            except Exception:
                print("Warning: Could not load existing data, starting fresh.")

    gm_handles = fetch_users(2400, 4000)
    cm_handles = fetch_users(1900, 2400)
    spec_handles = fetch_users(1500, 1900)
    pupil_handles = fetch_users(1200, 1500)
    newbie_handles = fetch_users(800, 1200)

    categories = {
        'gm': gm_handles,
        'cm': cm_handles,
        'spec': spec_handles,
        'pupil': pupil_handles,
        'newbie': newbie_handles
    }

    extracted_users = []
    for key, users_list in categories.items():
        for handle in users_list:
            if handle not in seen_handles:
                extracted_users.append(handle)

    random.shuffle(extracted_users)

    for idx, handle in enumerate(tqdm(extracted_users, desc="Scraping users", unit="user")):
        try:
            info = fetch_info(handle)
            if info and info['current_rating'] is not None:
                all_data.append(info)
                seen_handles.add(handle)
        except Exception:
            continue

        time.sleep(2)

        if len(all_data) % 10 == 0:
            with open(save_path, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"Progress saved after {len(all_data)} users.")

    with open(save_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Final data saved to {save_path}")


if __name__ == "__main__":
    scrape_users()