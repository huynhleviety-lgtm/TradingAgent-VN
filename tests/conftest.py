import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "ta_core"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

# config.py reads paths once at import time, so every database is pointed at a
# throwaway folder before any project module is imported. Tests never touch data/.
_TEST_DATA = Path(tempfile.mkdtemp(prefix="ta_vn_tests_"))
os.environ["VNSTOCK_DB_PATH"] = str(_TEST_DATA / "vnstock.db")
os.environ["VN_MARKET_DATA_DB"] = str(_TEST_DATA / "vn_market_data.db")
os.environ["NEWS_DB_PATH"] = str(_TEST_DATA / "news.db")
os.environ["COGNITIVE_DB_PATH"] = str(_TEST_DATA / "cognitive.db")
os.environ["DATA_DIR"] = str(_TEST_DATA)
