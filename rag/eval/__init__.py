from pathlib import Path

# 评测专用知识库 id，与生产 KB 隔离
EVAL_KB_ID = "00000000-0000-0000-0000-0000000000ee"

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"
