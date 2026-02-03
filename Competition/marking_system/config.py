"""
COMP219 Marking System - Configuration
Author: Lingfang Li
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class MarkingConfig:
    """Configuration for the marking system"""

    # ==================== Path Configuration ====================
    # Main submission zip file (relative to work_dir, or absolute path)
    submissions_zip: str = "submissions.zip"

    # Working directory for the marking system (auto-detected from this file's location)
    work_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    # Extracted files will be placed here
    extracted_dir: str = field(default="")

    # Separated py and pt files directories
    py_dir: str = field(default="")
    pt_dir: str = field(default="")

    # Fixed files directories (for incremental updates)
    fixed_py_dir: str = field(default="")
    fixed_pt_dir: str = field(default="")

    # Output directory for results
    output_dir: str = field(default="")

    # ==================== Evaluation Configuration ====================
    # Number of test samples (from FashionMNIST test set)
    num_test_samples: int = 500

    # Batch size for evaluation
    batch_size: int = 128

    # Random seed for reproducibility
    random_seed: int = 88

    # Number of evaluation rounds (to reduce randomness)
    num_rounds: int = 3

    # Epsilon constraint for L-infinity norm
    epsilon_constraint: float = 0.11

    # Timeout for each attack evaluation (seconds)
    attack_timeout: int = 60

    # ==================== Grading Configuration ====================
    # Base score (minimum score for valid submission) - used in min-max normalization
    base_score: float = 30.0

    # Maximum score for each assignment (Attack and Defence are separate assignments)
    max_score: float = 100.0

    # Legacy fields (kept for compatibility)
    max_attack_score: float = 35.0
    max_defence_score: float = 35.0

    # Z-Score normalization parameters
    use_zscore: bool = False          # Use Z-score instead of min-max
    zscore_target_mean: float = 60.0  # Target mean for both Attack and Defence
    zscore_target_std: float = 15.0   # Target standard deviation

    # ==================== System Configuration ====================
    # Device (cpu or cuda, auto-detected if empty)
    device: str = ""

    # Whether to auto-install dependencies
    auto_install_deps: bool = True

    # Log level
    log_level: str = "INFO"

    def __post_init__(self):
        """Initialize derived paths"""
        # Resolve submissions_zip relative to work_dir if not absolute
        if not os.path.isabs(self.submissions_zip):
            self.submissions_zip = os.path.join(self.work_dir, self.submissions_zip)
        if not self.extracted_dir:
            self.extracted_dir = os.path.join(self.work_dir, "extracted")
        if not self.py_dir:
            self.py_dir = os.path.join(self.work_dir, "py_files_v1")
        if not self.pt_dir:
            self.pt_dir = os.path.join(self.work_dir, "pt_files_v1")
        if not self.fixed_py_dir:
            self.fixed_py_dir = os.path.join(self.work_dir, "fixed_py_files")
        if not self.fixed_pt_dir:
            self.fixed_pt_dir = os.path.join(self.work_dir, "fixed_pt_files")
        if not self.output_dir:
            self.output_dir = os.path.join(self.work_dir, "results")
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_directories(self):
        """Create all necessary directories"""
        for dir_path in [self.extracted_dir, self.py_dir, self.pt_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def validate(self) -> list:
        """Validate configuration and return list of errors"""
        errors = []

        if not os.path.exists(self.submissions_zip):
            errors.append(f"Submissions zip not found: {self.submissions_zip}")

        if self.num_test_samples <= 0:
            errors.append("num_test_samples must be positive")

        if self.num_rounds <= 0:
            errors.append("num_rounds must be positive")

        return errors


# Scenario types for automatic detection
class ScenarioType:
    FULL_EVALUATION = "full_evaluation"           # First-time full evaluation
    INCREMENTAL_UPDATE = "incremental_update"     # Add new students to existing matrix
    RERUN_ATTACKS = "rerun_attacks"               # Re-run specific attack algorithms
    RERUN_MODELS = "rerun_models"                 # Re-run specific defence models
    EXTRACT_ONLY = "extract_only"                 # Only extract and organize files


# Default configuration instance
default_config = MarkingConfig()
