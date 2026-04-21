from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass

from llm import DESCRIPTION_LIMIT, get_recommendation, load_movies, movie_lookup, normalize_text, tokenize, watched_movie_ids


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    preferences: str
    history: list[str]
    history_ids: list[int]
    target_genres: tuple[str, ...]
    target_tokens: tuple[str, ...]


BENCHMARKS = [
    BenchmarkCase(
        name="superhero_fan",
        preferences="I want a superhero movie with a big team-up feel, sharp banter, and a sense of momentum.",
        history=["The Avengers", "Iron Man 3"],
        history_ids=[24428, 68721],
        target_genres=("action", "adventure", "science fiction"),
        target_tokens=("superhero", "team", "banter", "marvel"),
    ),
    BenchmarkCase(
        name="horror_fan",
        preferences="Give me something creepy and intense with dread instead of cheap jump scares.",
        history=["The Conjuring", "Get Out"],
        history_ids=[],
        target_genres=("horror", "thriller", "mystery"),
        target_tokens=("creepy", "dread", "psychological", "tension"),
    ),
    BenchmarkCase(
        name="romcom_viewer",
        preferences="I'm in the mood for a charming romantic comedy with great chemistry and actual laughs.",
        history=["Crazy Rich Asians", "10 Things I Hate About You"],
        history_ids=[],
        target_genres=("romance", "comedy"),
        target_tokens=("romance", "chemistry", "funny", "charming"),
    ),
    BenchmarkCase(
        name="slow_burn_drama",
        preferences="I want a thoughtful slow-burn drama with strong performances and emotional payoff.",
        history=["Marriage Story", "Manchester by the Sea"],
        history_ids=[],
        target_genres=("drama",),
        target_tokens=("slow", "character", "emotional", "performance"),
    ),
    BenchmarkCase(
        name="family_night",
        preferences="We need a family movie night pick that works for adults and kids and still feels smart.",
        history=["Finding Nemo", "Toy Story 3"],
        history_ids=[],
        target_genres=("family", "animation", "adventure"),
        target_tokens=("family", "heartwarming", "fun", "adventure"),
    ),
    BenchmarkCase(
        name="twist_endings",
        preferences="I love thrillers with twist endings, unreliable perspectives, and a lot to unpack afterward.",
        history=["Gone Girl", "Shutter Island"],
        history_ids=[],
        target_genres=("thriller", "mystery"),
        target_tokens=("twist", "mystery", "psychological", "unreliable"),
    ),
    BenchmarkCase(
        name="narrow_preferences",
        preferences="Please no horror, no bleak endings, and no super long movies. I want a warm funny sci-fi adventure.",
        history=["Interstellar"],
        history_ids=[157336],
        target_genres=("science fiction", "adventure", "comedy"),
        target_tokens=("warm", "funny", "adventure", "sci"),
    ),
    BenchmarkCase(
        name="vague_user",
        preferences="I just want something really good tonight.",
        history=["The Dark Knight", "Inception", "Interstellar"],
        history_ids=[],
        target_genres=("action", "science fiction", "thriller", "drama"),
        target_tokens=("great", "smart", "big", "entertaining"),
    ),
    BenchmarkCase(
        name="history_driven",
        preferences="Something exciting but not mindless.",
        history=["Whiplash", "Black Swan", "Nightcrawler"],
        history_ids=[],
        target_genres=("thriller", "drama"),
        target_tokens=("intense", "driven", "smart", "character"),
    ),
    BenchmarkCase(
        name="obvious_seen_already",
        preferences="I want an emotional animated movie with a lot of heart.",
        history=["Inside Out", "Coco", "Toy Story 3"],
        history_ids=[150540],
        target_genres=("animation", "family", "drama"),
        target_tokens=("heart", "emotional", "family", "animation"),
    ),
]


def quality_heuristic(case: BenchmarkCase, recommended_id: int) -> float:
    movie = movie_lookup()[recommended_id]
    genres = {normalize_text(genre) for genre in movie.genres}
    genre_hits = sum(1 for genre in case.target_genres if genre in genres)
    token_hits = sum(1 for token in case.target_tokens if token in movie.searchable_blob or token in movie.token_set)
    return genre_hits * 2.0 + token_hits * 0.75 + movie.quality_score


def run_case(case: BenchmarkCase) -> dict[str, object]:
    start = time.monotonic()
    result = get_recommendation(case.preferences, case.history, case.history_ids)
    elapsed = time.monotonic() - start
    watched_ids = watched_movie_ids(case.history, case.history_ids)

    errors: list[str] = []
    if not isinstance(result, dict):
        errors.append("result_not_dict")
        return {"name": case.name, "errors": errors}

    movie_id = result.get("tmdb_id")
    description = result.get("description", "")
    if movie_id not in movie_lookup():
        errors.append("invalid_tmdb_id")
    if movie_id in watched_ids:
        errors.append("watched_leak")
    if not isinstance(description, str) or not description.strip():
        errors.append("empty_description")
    if isinstance(description, str) and len(description) > DESCRIPTION_LIMIT:
        errors.append("description_too_long")
    if elapsed > 20:
        errors.append("timeout")

    score = quality_heuristic(case, int(movie_id)) if not errors else 0.0
    return {
        "name": case.name,
        "tmdb_id": movie_id,
        "title": movie_lookup()[movie_id].title if movie_id in movie_lookup() else None,
        "runtime_seconds": round(elapsed, 3),
        "description_length": len(description) if isinstance(description, str) else None,
        "quality_score": round(score, 3),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local benchmark cases for the movie recommender.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text summary.")
    args = parser.parse_args()

    results = [run_case(case) for case in BENCHMARKS]
    failing = [result for result in results if result["errors"]]
    quality_scores = [result["quality_score"] for result in results if not result["errors"]]
    summary = {
        "benchmark_count": len(results),
        "pass_count": len(results) - len(failing),
        "fail_count": len(failing),
        "avg_runtime_seconds": round(statistics.mean(result["runtime_seconds"] for result in results), 3),
        "avg_quality_score": round(statistics.mean(quality_scores), 3) if quality_scores else 0.0,
    }

    if args.json:
        print(json.dumps({"summary": summary, "results": results}, indent=2))
        return

    print("Local benchmark summary")
    print(json.dumps(summary, indent=2))
    print("\nPer-case results")
    for result in results:
        marker = "PASS" if not result["errors"] else "FAIL"
        print(
            f"- {marker} {result['name']}: {result['title']} "
            f"(id={result['tmdb_id']}, runtime={result['runtime_seconds']}s, "
            f"desc_len={result['description_length']}, quality={result['quality_score']}, "
            f"errors={result['errors']})"
        )


if __name__ == "__main__":
    main()
        target_tokens=("creepy", "dread", "psychological", "tension"),
    ),
    BenchmarkCase(
        name="romcom_viewer",
        preferences="I'm in the mood for a charming romantic comedy with great chemistry and actual laughs.",
        history=["Crazy Rich Asians", "10 Things I Hate About You"],
        history_ids=[],
        target_genres=("romance", "comedy"),
        target_tokens=("romance", "chemistry", "funny", "charming"),
    ),
    BenchmarkCase(
        name="slow_burn_drama",
        preferences="I want a thoughtful slow-burn drama with strong performances and emotional payoff.",
        history=["Marriage Story", "Manchester by the Sea"],
        history_ids=[],
        target_genres=("drama",),
        target_tokens=("slow", "character", "emotional", "performance"),
    ),
    BenchmarkCase(
        name="family_night",
        preferences="We need a family movie night pick that works for adults and kids and still feels smart.",
        history=["Finding Nemo", "Toy Story 3"],
        history_ids=[],
        target_genres=("family", "animation", "adventure"),
        target_tokens=("family", "heartwarming", "fun", "adventure"),
    ),
    BenchmarkCase(
        name="twist_endings",
        preferences="I love thrillers with twist endings, unreliable perspectives, and a lot to unpack afterward.",
        history=["Gone Girl", "Shutter Island"],
        history_ids=[],
        target_genres=("thriller", "mystery"),
        target_tokens=("twist", "mystery", "psychological", "unreliable"),
    ),
    BenchmarkCase(
        name="narrow_preferences",
        preferences="Please no horror, no bleak endings, and no super long movies. I want a warm funny sci-fi adventure.",
        history=["Interstellar"],
        history_ids=[157336],
        target_genres=("science fiction", "adventure", "comedy"),
        target_tokens=("warm", "funny", "adventure", "sci"),
    ),
    BenchmarkCase(
        name="vague_user",
        preferences="I just want something really good tonight.",
        history=["The Dark Knight", "Inception", "Interstellar"],
        history_ids=[],
        target_genres=("action", "science fiction", "thriller", "drama"),
        target_tokens=("great", "smart", "big", "entertaining"),
    ),
    BenchmarkCase(
        name="history_driven",
        preferences="Something exciting but not mindless.",
        history=["Whiplash", "Black Swan", "Nightcrawler"],
        history_ids=[],
        target_genres=("thriller", "drama"),
        target_tokens=("intense", "driven", "smart", "character"),
    ),
    BenchmarkCase(
        name="obvious_seen_already",
        preferences="I want an emotional animated movie with a lot of heart.",
        history=["Inside Out", "Coco", "Toy Story 3"],
        history_ids=[150540],
        target_genres=("animation", "family", "drama"),
        target_tokens=("heart", "emotional", "family", "animation"),
    ),
]


def quality_heuristic(case: BenchmarkCase, recommended_id: int) -> float:
    movie = movie_lookup()[recommended_id]
    genres = {normalize_text(genre) for genre in movie.genres}
    genre_hits = sum(1 for genre in case.target_genres if genre in genres)
    token_hits = sum(1 for token in case.target_tokens if token in movie.searchable_blob or token in movie.token_set)
    return genre_hits * 2.0 + token_hits * 0.75 + movie.quality_score


def run_case(case: BenchmarkCase) -> dict[str, object]:
    start = time.monotonic()
    result = get_recommendation(case.preferences, case.history, case.history_ids)
    elapsed = time.monotonic() - start
    watched_ids = watched_movie_ids(case.history, case.history_ids)

    errors: list[str] = []
    if not isinstance(result, dict):
        errors.append("result_not_dict")
        return {"name": case.name, "errors": errors}

    movie_id = result.get("tmdb_id")
    description = result.get("description", "")
    if movie_id not in movie_lookup():
        errors.append("invalid_tmdb_id")
    if movie_id in watched_ids:
        errors.append("watched_leak")
    if not isinstance(description, str) or not description.strip():
        errors.append("empty_description")
    if isinstance(description, str) and len(description) > DESCRIPTION_LIMIT:
        errors.append("description_too_long")
    if elapsed > 20:
        errors.append("timeout")

    score = quality_heuristic(case, int(movie_id)) if not errors else 0.0
    return {
        "name": case.name,
        "tmdb_id": movie_id,
        "title": movie_lookup()[movie_id].title if movie_id in movie_lookup() else None,
        "runtime_seconds": round(elapsed, 3),
        "description_length": len(description) if isinstance(description, str) else None,
        "quality_score": round(score, 3),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local benchmark cases for the movie recommender.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text summary.")
    args = parser.parse_args()

    results = [run_case(case) for case in BENCHMARKS]
    failing = [result for result in results if result["errors"]]
    quality_scores = [result["quality_score"] for result in results if not result["errors"]]
    summary = {
        "benchmark_count": len(results),
        "pass_count": len(results) - len(failing),
        "fail_count": len(failing),
        "avg_runtime_seconds": round(statistics.mean(result["runtime_seconds"] for result in results), 3),
        "avg_quality_score": round(statistics.mean(quality_scores), 3) if quality_scores else 0.0,
    }

    if args.json:
        print(json.dumps({"summary": summary, "results": results}, indent=2))
        return

    print("Local benchmark summary")
    print(json.dumps(summary, indent=2))
    print("\nPer-case results")
    for result in results:
        marker = "PASS" if not result["errors"] else "FAIL"
        print(
            f"- {marker} {result['name']}: {result['title']} "
            f"(id={result['tmdb_id']}, runtime={result['runtime_seconds']}s, "
            f"desc_len={result['description_length']}, quality={result['quality_score']}, "
            f"errors={result['errors']})"
        )


if __name__ == "__main__":
    main()
