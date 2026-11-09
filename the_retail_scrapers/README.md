# AI-Assisted Furniture Planner (CPSC 481)

This project is a command-line AI furniture assistant that converts a short natural-language home description into furniture bundles for each room. It compares two approaches:

- **Greedy baseline planner**: slot-by-slot selection using an effective-price ranking (+ simple tag bonuses).
- **A*-style planner**: best-first search over partial room bundles using a match-score heuristic and pruning.

---

## What’s Included in the Submission

- `planner_demo.py` — CLI driver that prompts for a home description, budget, and room counts; runs both planners.
- `planner_greedy.py` — greedy planner + shared data models (`BundleItem`, `BundleResult`) and candidate utilities.
- `planner_astar.py` — A* planner (priority queue search using match scores and an optimistic upper bound).
- `.env` — included in the ZIP submission (contains Supabase credentials needed to run).
- Report PDF + Slides (if included in your submission ZIP).

> ⚠️ The `.env` contains credentials. Keep the ZIP private and do not upload it publicly.

---

## Requirements

- Python **3.10+** recommended
- Internet access (for Supabase)
- A Supabase project already populated with the furniture dataset (this submission points to that Supabase instance via `.env`)

---

## Setup (Instructor / TA)

### 1) Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
WINDOWS: 
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Then,
pip install -r requirements.txt
The credentials for the supabase db are included in the .env file.