"""Matched R=W, K1=K2 forests. Sampling and target verification stay native."""
import os
import sys
from pathlib import Path
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
CHECKOUT = Path(os.environ.get('PAPER_CHECKOUT', str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(CHECKOUT / 'ssd'))
from mlsys_equal_budget_adapter import install
install()
from mlsys_tree_followup import main
if __name__ == '__main__':
    main()
