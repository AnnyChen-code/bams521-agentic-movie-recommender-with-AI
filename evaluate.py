from __future__ import annotations

from flask import Flask, jsonify, request

from llm import get_recommendation

app = Flask(__name__)


@app.get("/")
def healthcheck():
    return jsonify({"status": "ok", "message": "BAMS 521 movie recommender is running."})


@app.post("/recommend")
def recommend():
    payload = request.get_json(silent=True) or {}

    user_id = payload.get("user_id")
    preferences = str(payload.get("preferences", "")).strip()
    history = payload.get("history", []) or []

    history_titles: list[str] = []
    history_ids: list[int] = []

    for item in history:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        tmdb_id = item.get("tmdb_id")

        if name:
            history_titles.append(name)

        try:
            if tmdb_id is not None:
                history_ids.append(int(tmdb_id))
        except (TypeError, ValueError):
            pass

    if not preferences:
        return jsonify({"error": "preferences is required"}), 400

    recommendation = get_recommendation(preferences, history_titles, history_ids)
    response = {"user_id": user_id, **recommendation} if user_id is not None else recommendation
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
