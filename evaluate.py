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
