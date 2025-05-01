from flask import Flask, request, jsonify
from predictor import fetch_user_data, predict_from_profile
from recommender import recommend_problems
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["POST"])
def handle_request():
    try:
        user_data = request.get_json()
        if not user_data or user_data.get("handle") is None:
            return jsonify({"error": "No handle provided"}), 400

        handle = user_data["handle"]
        user_profile = fetch_user_data(handle)

        result = predict_from_profile(user_profile)
        recommendations = recommend_problems(user_profile, result["predicted_rating"])

        return jsonify({
            "predicted_rating": result["predicted_rating"],
            "rating_change": result["rating_change"],
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)