import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set. Add it to .env in the project root.")

CHROMA_DIR = ROOT_DIR / "chroma_db"
PLAYER_STATS_CSV = ROOT_DIR / "player-data" / "player_stats.csv"
PHOTO_CACHE_DIR = ROOT_DIR / "player-data" / "photo_cache"
COLLECTION_NAME = "player_stats"
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
