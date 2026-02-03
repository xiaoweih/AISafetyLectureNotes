"""
COMP219 Marking System
=====================

A unified marking system for the COMP219 Adversarial Attack Competition.

Modules:
    - config: Configuration management
    - utils: Utility functions and neural network definition
    - file_extractor: Student submission extraction
    - dependency_manager: Automatic dependency handling
    - evaluator: Mutual evaluation engine
    - grade_calculator: Final grade calculation
    - main: Main controller and CLI

Usage:
    from marking_system import MarkingSystem, MarkingConfig

    config = MarkingConfig()
    system = MarkingSystem(config)
    system.run_full_pipeline()

CLI Usage:
    python main.py                    # Full pipeline
    python main.py --extract-only     # Extract files only
    python main.py --evaluate-only    # Evaluate only
    python main.py --grade-only       # Calculate grades only
"""

from .config import MarkingConfig, ScenarioType
from .utils import Net, logger
from .file_extractor import FileExtractor, ExtractionResult, StudentSubmission
from .dependency_manager import DependencyManager, load_attack_function
from .evaluator import Evaluator, IncrementalEvaluator
from .grade_calculator import GradeCalculator, StudentGrade
from .main import MarkingSystem

__version__ = '2025.1.0'
__author__ = 'Lingfang Li'

__all__ = [
    'MarkingConfig',
    'ScenarioType',
    'Net',
    'logger',
    'FileExtractor',
    'ExtractionResult',
    'StudentSubmission',
    'DependencyManager',
    'load_attack_function',
    'Evaluator',
    'IncrementalEvaluator',
    'GradeCalculator',
    'StudentGrade',
    'MarkingSystem',
]
