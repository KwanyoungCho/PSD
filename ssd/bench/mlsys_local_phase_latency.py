"""Local-only B1 tree phase measurements using the existing benchmark harness."""
import os
import socket
import sys
import faulthandler
import signal
from pathlib import Path

if socket.gethostname().split('.')[0] != 'orion':
    raise RuntimeError('User requested local orion server only; do not launch remotely')
os.environ.setdefault('SSD_CUDA_ARCH', '8.9')
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
faulthandler.register(signal.SIGUSR1, all_threads=True)

from mlsys_ps_preparation import main
from mlsys_phase_probe import install
from mlsys_phase_compilation import install as install_preparation_audit
install_preparation_audit()

if os.getenv('DUET_PHASE_PROBE', '0') == '1':
    install()

if __name__ == '__main__':
    main()
