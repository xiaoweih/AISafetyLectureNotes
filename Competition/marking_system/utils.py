"""
COMP219 Marking System - Utility Functions
Author: Lingfang Li
"""

import os
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from typing import Optional, Tuple, List, Dict


# ==================== Logging Setup ====================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("COMP219_Marking")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logging()


# ==================== Neural Network Definition ====================

class Net(nn.Module):
    """
    Fixed network architecture for COMP219 competition.
    4-layer fully connected network: 784 -> 128 -> 64 -> 32 -> 10
    DO NOT MODIFY THIS CLASS.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


# ==================== File Path Utilities ====================

def get_py_filename(student_id: str) -> str:
    """Get Python file name for a student ID."""
    return f"competition_{student_id}.py"


def get_pt_filename(student_id: str) -> str:
    """Get model file name for a student ID."""
    return f"{student_id}.pt"


def get_py_filepath(directory: str, student_id: str) -> str:
    """Get full path to a student's Python file."""
    return os.path.join(directory, get_py_filename(student_id))


def get_pt_filepath(directory: str, student_id: str) -> str:
    """Get full path to a student's model file."""
    return os.path.join(directory, get_pt_filename(student_id))


# ==================== Student ID Extraction ====================

def get_student_ids_from_directory(directory: str, extension: str) -> set:
    """
    Extract all student IDs from files in a directory.

    Args:
        directory: Directory path to scan
        extension: File extension to match (e.g., '.py', '.pt')

    Returns:
        Set of student ID strings
    """
    ids = set()
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.endswith(extension):
                match = re.search(r'(\d{9})', f)
                if match:
                    ids.add(match.group(1))
    return ids


def extract_student_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract 9-digit student ID from filename.

    Args:
        filename: File name or path

    Returns:
        Student ID string or None if not found
    """
    # Pattern: 9 consecutive digits, typically starting with 20
    patterns = [
        r'(\d{9})',  # Any 9 digits
        r'20\d{7}',  # Starting with 20
    ]

    basename = os.path.basename(filename)
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            return match.group(1) if '(' in pattern else match.group()

    return None


def extract_student_id_from_code(py_file: str) -> Optional[str]:
    """
    Extract student ID from the id_ variable in Python code.

    Args:
        py_file: Path to Python file

    Returns:
        Student ID string or None if not found
    """
    id_pattern = re.compile(r"id_\s*=\s*(\d{9})")

    try:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            match = id_pattern.search(content)
            if match:
                return match.group(1)
    except Exception as e:
        logger.warning(f"Could not read file {py_file}: {e}")

    return None


def extract_student_id(file_path: str) -> Optional[str]:
    """
    Extract student ID from file - tries filename first, then code content.

    Args:
        file_path: Path to file

    Returns:
        Student ID string or None if not found
    """
    # Try filename first
    student_id = extract_student_id_from_filename(file_path)

    # If .py file and no ID from filename, try code content
    if student_id is None and file_path.endswith('.py'):
        student_id = extract_student_id_from_code(file_path)

    return student_id


# ==================== Random Seed Management ====================

def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ==================== Model Loading ====================

def load_model(model_path: str, device: torch.device) -> Optional[Net]:
    """
    Load a trained model from .pt file.

    Handles different save formats:
    - Direct state_dict
    - Dictionary with 'state_dict' key
    - Dictionary with 'model_state_dict' key

    Args:
        model_path: Path to .pt file
        device: torch device

    Returns:
        Loaded model or None if failed
    """
    try:
        model = Net().to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Handle different save formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        logger.warning("=" * 60)
        logger.warning(f"[MANUAL CHECK] Model loading FAILED")
        logger.warning(f"File: {model_path}")
        logger.warning(f"Error: {e}")
        logger.warning(f"Possible reasons:")
        logger.warning(f"  1. Student modified network architecture (not using required Net class)")
        logger.warning(f"  2. Model saved with different format (e.g., full model vs state_dict)")
        logger.warning(f"  3. Corrupted or wrong .pt file submitted")
        logger.warning(f"Please manually check this student's submission.")
        logger.warning("=" * 60)
        return None


# ==================== Epsilon Verification ====================

def verify_epsilon_constraint(
        original: torch.Tensor,
        adversarial: torch.Tensor,
        epsilon: float = 0.11
) -> Tuple[bool, float]:
    """
    Verify that adversarial perturbation satisfies L-infinity constraint.

    Args:
        original: Original input tensor
        adversarial: Adversarial input tensor
        epsilon: Maximum allowed perturbation (strictly less than)

    Returns:
        Tuple of (is_valid, actual_perturbation)
    """
    perturbation = torch.norm(original - adversarial, float('inf')).item()
    is_valid = perturbation < epsilon
    return is_valid, perturbation


# ==================== Result File Management ====================

def get_timestamp() -> str:
    """Get current timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(results: Dict, output_path: str):
    """Save results dictionary to file"""
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


# ==================== Progress Display ====================

def print_progress_bar(
        iteration: int,
        total: int,
        prefix: str = '',
        suffix: str = '',
        length: int = 50
):
    """Print progress bar to console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


# ==================== Summary Statistics ====================

def print_summary_statistics(data: np.ndarray, name: str):
    """Print summary statistics for a numpy array"""
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Min:    {data.min():.4f}")
    logger.info(f"  Max:    {data.max():.4f}")
    logger.info(f"  Mean:   {data.mean():.4f}")
    logger.info(f"  Median: {np.median(data):.4f}")
    logger.info(f"  Std:    {data.std():.4f}")
