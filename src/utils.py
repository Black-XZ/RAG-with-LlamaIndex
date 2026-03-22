"""
Utilities Module
================
Common utility functions for logging, timing, caching, and file operations.
"""

import os
import sys
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Any, Callable, List, Optional, Dict
from functools import wraps
from datetime import datetime
from contextlib import contextmanager

import yaml

logger = logging.getLogger(__name__)


# ============= Logging Utilities =============

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "results/logs"
) -> logging.Logger:
    """
    Set up logging with both file and console handlers.

    Args:
        level: Logging level.
        log_file: Optional specific log file name.
        log_dir: Directory for log files.

    Returns:
        Configured logger.
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"rag_{timestamp}.log"

    log_file_path = log_path / log_file

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger("RAG")
    logger.info(f"Logging initialized: {log_file_path}")

    return logger


# ============= Timing Utilities =============

@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing operations.

    Usage:
        with timer("Build index"):
            build_index()

    Args:
        name: Name of the operation being timed.
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} completed in {elapsed:.2f}s")


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Args:
        func: Function to time.

    Returns:
        Wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ============= File Utilities =============

def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path: Path to YAML file.

    Returns:
        Dictionary with configuration.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str):
    """
    Save data to a YAML file.

    Args:
        data: Dictionary to save.
        file_path: Output file path.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(file_path: str) -> Any:
    """
    Load a JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Loaded data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    Save data to a JSON file.

    Args:
        data: Data to save.
        file_path: Output file path.
        indent: JSON indentation.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ============= Caching Utilities =============

def get_cache_dir() -> Path:
    """Get or create cache directory."""
    cache_dir = Path.home() / ".cache" / "rag-llamaindex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        MD5 hash string.
    """
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


# ============= Path Utilities =============

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_results_dir() -> Path:
    """Get the results directory."""
    return get_project_root() / "results"


def get_configs_dir() -> Path:
    """Get the configs directory."""
    return get_project_root() / "configs"


# ============= String Utilities =============

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to add when truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing invalid characters.

    Args:
        filename: Original filename.

    Returns:
        Cleaned filename.
    """
    import re
    # Remove invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    cleaned = cleaned.strip('. ')
    return cleaned or "unnamed"


# ============= Date/Time Utilities =============

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_readable_timestamp() -> str:
    """Get human-readable timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============= System Utilities =============

def check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        'cuda_available': False,
        'cuda_device_count': 0,
        'device_name': 'cpu'
    }

    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()

        if info['cuda_available']:
            info['cuda_device_count'] = torch.cuda.device_count()
            info['device_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda

    except ImportError:
        pass

    return info


# ============= Random Utilities =============

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    random.seed(seed)
    np.random.seed(seed)

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"Random seed set to {seed}")


# ============= Progress Utilities =============

def create_progress_bar(iterable, desc: str = "Processing"):
    """
    Create a progress bar for an iterable.

    Args:
        iterable: Iterable to wrap.
        desc: Description for progress bar.

    Returns:
        tqdm progress bar.
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


# ============= Validation Utilities =============

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that a configuration has required keys.

    Args:
        config: Configuration dictionary.
        required_keys: List of required keys.

    Returns:
        True if valid, raises ValueError if not.
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return True


# ============= Export utilities =============

def export_to_markdown_table(
    headers: List[str],
    rows: List[List[Any]]
) -> str:
    """
    Create a markdown table string.

    Args:
        headers: Column headers.
        rows: Data rows.

    Returns:
        Markdown table string.
    """
    # Header row
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    # Data rows
    data_rows = []
    for row in rows:
        data_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join([header_row, separator] + data_rows)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Check device
    device_info = get_device_info()
    print(f"Device info: {device_info}")

    # Test timestamp
    print(f"Timestamp: {get_timestamp()}")

    # Test file paths
    print(f"Project root: {get_project_root()}")
    print(f"Data dir: {get_data_dir()}")

    # Test string utilities
    text = "This is a very long text that should be truncated"
    print(f"Truncated: {truncate_text(text, 30)}")

    # Test seed
    set_seed(42)
