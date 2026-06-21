import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from artemis.api import app  # noqa: E402

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
