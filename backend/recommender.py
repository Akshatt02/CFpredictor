import requests
import random

def recommend_problems(user_data, predicted_rating, count=10):
    weak_tags = sorted(user_data['tag_ac_count'], key=lambda tag: user_data['tag_ac_count'][tag] / user_data['tag_submission_count'].get(tag, 1))[:3]

    problems = []
    for tag in weak_tags:
        try:
            res = requests.get(f"https://codeforces.com/api/problemset.problems?tags={tag}").json()
            if res['status'] != 'OK':
                continue

            for problem, stats in zip(res['result']['problems'], res['result']['problemStatistics']):
                if 'rating' in problem and problem['rating'] - predicted_rating <= 200 and problem['rating'] - predicted_rating >= 0:
                    problems.append({
                        "name": problem['name'],
                        "contestId": problem['contestId'],
                        "index": problem['index'],
                        "rating": problem.get('rating', '?'),
                        "tags": problem['tags'],
                        "url": f"https://codeforces.com/contest/{problem['contestId']}/problem/{problem['index']}"
                    })
        except:
            continue

    random.shuffle(problems)
    return problems[:count]