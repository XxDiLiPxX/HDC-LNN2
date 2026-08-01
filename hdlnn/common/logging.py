import logging
import sys
from pathlib import Path

def setup_logging(log_file: Path = None, level: int = logging.INFO):
    """Sets up the global logging configuration to log to stdout and optionally a run file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True
    )
