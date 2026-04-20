# BAMS 521 Agentic Movie Recommender

This project implements a competition-oriented movie recommendation agent for the BAMS 521 Agentic AI movie recommender assignment. The system takes a user's free-text preferences plus watch history and returns exactly one valid recommendation from the provided TMDB candidate set, along with a short persuasive description.

## Goal

The assignment rewards more than basic correctness. A strong submission needs to:

- obey all schema and runtime constraints
- avoid recommending watched movies
- stay inside the candidate set
- write a description that sounds like a good recommendation, not a database summary
- remain fast and robust under grading

This solution is built with those grading incentives in mind.

## Approach

The recommender uses a practical hybrid pipeline instead of a fragile open-ended agent loop.

1. Load the local TMDB candidate workbook and cache the movie metadata.
2. Normalize watch history and filter watched movies using both `history_ids` and conservative title matching.
3. Extract signals from the user's free-text preferences:
   - genres
   - themes and tone
   - exclusions
   - useful keywords
4. Build a lightweight history profile from the watched movies to recover taste when the prompt is vague.
5. Score every candidate with a hybrid ranker that blends:
   - preference-token overlap
   - genre alignment
   - history similarity
   - exclusion penalties
   - movie quality and popularity as tie-breakers
6. Select the top unwatched movie.
7. Generate a short persuasive description with the required model (`gemma4:31b-cloud`) when `OLLAMA_API_KEY` is available.
8. Fall back to a deterministic description if the model call fails, the key is missing, or latency budget gets tight.
9. Validate the final output before returning it.

## Why This Design

This design aims to maximize both grade safety and competition strength.

- Deterministic retrieval/reranking is faster and easier to trust than letting the LLM choose from the full catalog.
- The LLM is only used for the part where it adds the most value: persuasive wording.
- The code has multiple safeguards against the two easiest ways to get disqualified:
  - recommending a watched movie
  - returning an invalid `tmdb_id`
- The scoring logic is transparent enough to explain clearly in class or in peer review.

## Architecture

Core file:

- `llm.py`: submission entry point with `get_recommendation(...)`

Support file:

- `evaluate.py`: local benchmark harness and rule-check script

Important constants in `llm.py`:

- `MODEL = "gemma4:31b-cloud"`
- `DESCRIPTION_LIMIT = 500`
- `REQUEST_TIMEOUT_SECONDS = 8`

## Evaluation Strategy

The assignment explicitly rewards thoughtful evaluation, so this repo includes a simple but useful local benchmark harness.

`evaluate.py` covers ten user archetypes:

- superhero fans
- horror fans
- rom-com viewers
- slow-burn drama viewers
- family movie night
- twist-ending fans
- narrow preferences with exclusions
- vague users
- users whose history should drive taste inference
- users who have already seen obvious recommendations

For each case, the evaluator checks:

- valid dict output
- candidate-set compliance
- no watched-movie leakage
- description non-empty and within 500 characters
- runtime under the assignment limit

It also reports a lightweight heuristic quality score based on genre and token match to the benchmark intent.

This is not a perfect proxy for human preference, but it is useful for regression testing and prompt/ranking iteration.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Ollama Cloud API key:

```bash
export OLLAMA_API_KEY=your_key_here
```

Run a manual recommendation:

```bash
python llm.py \
  --preferences "I want a funny, energetic superhero movie with banter." \
  --history "The Avengers|Iron Man 3" \
  --history-ids 24428 68721
```

Run local evaluation:

```bash
python evaluate.py
```

If `OLLAMA_API_KEY` is missing, the recommender still returns a valid result using the deterministic fallback description path. That is useful for local logic testing, but the intended final submission behavior is to use the injected key during grading.

## Deployment Note

The assignment materials include Leapcell deployment instructions with an API wrapper that accepts a payload containing `user_id`, `preferences`, and structured `history`. This repository focuses on the required graded deliverable, which is `llm.py` implementing `get_recommendation(...)`.

If you wrap this in an API, keep the wrapper thin:

- parse the incoming payload
- call `get_recommendation(preferences, history_titles, history_ids)`
- attach `user_id` in the HTTP response if needed

## Rule Protection Checklist

This implementation explicitly guards against:

- invalid output shape
- invalid `tmdb_id`
- recommending watched movies
- empty descriptions
- descriptions over 500 characters
- LLM failure causing invalid output
- excessive dependence on external calls

## Known Limitations

- Watch history is implicit, not rated, so the system treats it as a taste signal rather than a strict positive label.
- Title-based watch filtering is conservative but not perfect when multiple movies share near-identical normalized names.
- The quality evaluator is heuristic, not a substitute for real pairwise human judging.
- The recommendation quality depends on the metadata available in the provided TMDB sheet; no extra external movie knowledge is required for the core ranking path.

## Submission Hygiene

The submission is designed to stay clean:

- no hardcoded secrets
- minimal dependencies
- no `.env`, virtual environment, or cache files required
- uses the local provided dataset directly

Overall, the strategy is to be strong where the assignment is actually scored: valid outputs, good taste matching, convincing copy, fast runtime, and a clear evaluation story.
