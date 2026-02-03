"""
COMP219 Marking System - Dependency Manager Module
Author: Lingfang Li

Handles automatic detection and installation of dependencies in student code.
"""

import os
import re
import sys
import subprocess
import importlib
import importlib.util
import threading
import queue
from typing import Set, Optional, Callable, Tuple
from dataclasses import dataclass

from utils import logger

# Timeout for loading student code (seconds)
LOAD_TIMEOUT = 30


# Standard libraries that don't need installation
STANDARD_LIBS = {
    "os", "sys", "re", "math", "time", "copy", "random", "itertools",
    "functools", "collections", "typing", "dataclasses", "abc",
    "json", "csv", "pickle", "io", "pathlib", "datetime", "argparse",
    "warnings", "traceback", "inspect", "contextlib"
}

# PyTorch related packages (usually pre-installed)
PYTORCH_LIBS = {
    "torch", "torchvision", "torchaudio", "torch_optimizer"
}

# Scientific computing packages (usually pre-installed)
SCIENTIFIC_LIBS = {
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "PIL", "cv2"
}

# Known package mappings (import name -> pip package name)
PACKAGE_MAPPINGS = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "torch_optimizer": "torch-optimizer",
}

# Special packages requiring custom installation
SPECIAL_PACKAGES = {
    "autoattack": "git+https://github.com/fra31/auto-attack",
    "ranger": "torch-optimizer",  # ranger is part of torch-optimizer
}


@dataclass
class DependencyCheckResult:
    """Result of dependency checking for a file"""
    file_path: str
    dependencies: Set[str]
    missing_packages: Set[str]
    installation_errors: Set[str]
    requires_grad: bool = False
    has_autoattack: bool = False
    success: bool = True


class DependencyManager:
    """
    Manages dependencies for student submission files.

    Features:
    - Detects import statements
    - Identifies missing packages
    - Auto-installs missing packages
    - Handles special packages (AutoAttack, Ranger)
    - Detects gradient requirements
    """

    def __init__(self, auto_install: bool = True):
        """
        Initialize the dependency manager.

        Args:
            auto_install: Whether to automatically install missing packages
        """
        self.auto_install = auto_install
        self._installed_packages: Set[str] = set()

    def check_and_install(self, py_file: str) -> DependencyCheckResult:
        """
        Check dependencies in a Python file and optionally install missing ones.

        Args:
            py_file: Path to Python file

        Returns:
            DependencyCheckResult with dependency information
        """
        result = DependencyCheckResult(
            file_path=py_file,
            dependencies=set(),
            missing_packages=set(),
            installation_errors=set()
        )

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Could not read file {py_file}: {e}")
            result.success = False
            return result

        # Extract dependencies
        result.dependencies = self._extract_dependencies(code)

        # Check for gradient usage
        result.requires_grad = self._check_gradient_usage(code)

        # Check for AutoAttack
        result.has_autoattack = 'autoattack' in code.lower() or 'AutoAttack' in code

        # Find missing packages
        for dep in result.dependencies:
            if not self._is_package_available(dep):
                result.missing_packages.add(dep)

        # Install missing packages if enabled
        if self.auto_install and result.missing_packages:
            logger.info(f"Installing missing packages for {py_file}: {result.missing_packages}")
            for package in result.missing_packages.copy():
                success = self._install_package(package)
                if success:
                    result.missing_packages.discard(package)
                else:
                    result.installation_errors.add(package)

        result.success = len(result.installation_errors) == 0

        return result

    def _extract_dependencies(self, code: str) -> Set[str]:
        """
        Extract imported module names from Python code.

        Args:
            code: Python source code

        Returns:
            Set of module names
        """
        dependencies = set()

        # Match import statements
        import_pattern = re.compile(r'^\s*(import|from)\s+([a-zA-Z0-9_\.]+)', re.MULTILINE)

        for match in import_pattern.finditer(code):
            module_name = match.group(2).split('.')[0]  # Get top-level module

            # Skip standard libraries
            if module_name in STANDARD_LIBS:
                continue

            # Skip if it's a local import (starts with .)
            if module_name.startswith('.'):
                continue

            dependencies.add(module_name)

        return dependencies

    def _check_gradient_usage(self, code: str) -> bool:
        """
        Check if code uses gradient operations.

        Args:
            code: Python source code

        Returns:
            True if gradient operations are detected
        """
        gradient_patterns = [
            r'requires_grad',
            r'\.grad\b',
            r'backward\(\)',
            r'enable_grad',
            r'set_grad_enabled',
            r'no_grad',
            r'grad_fn'
        ]

        for pattern in gradient_patterns:
            if re.search(pattern, code):
                return True

        return False

    def _is_package_available(self, package_name: str) -> bool:
        """
        Check if a package is available for import.

        Args:
            package_name: Name of the package

        Returns:
            True if package can be imported
        """
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False

    def _install_package(self, package_name: str) -> bool:
        """
        Install a package using pip.

        Args:
            package_name: Name of the package

        Returns:
            True if installation succeeded
        """
        # Get pip package name (may differ from import name)
        pip_name = PACKAGE_MAPPINGS.get(package_name, package_name)

        # Check for special packages
        if package_name in SPECIAL_PACKAGES:
            pip_name = SPECIAL_PACKAGES[package_name]

        try:
            logger.info(f"Installing {pip_name}...")

            if pip_name.startswith('git+'):
                # Git-based installation
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Regular pip installation
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            self._installed_packages.add(package_name)
            logger.info(f"Successfully installed {pip_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {pip_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing {pip_name}: {e}")
            return False

    def fix_ranger_import(self, py_file: str) -> bool:
        """
        Fix Ranger optimizer import statement if needed.

        Some students may use 'from ranger import Ranger' which doesn't work.
        This fixes it to 'from torch_optimizer import Ranger'.

        Args:
            py_file: Path to Python file

        Returns:
            True if file was modified
        """
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()

            if 'from ranger import' in code:
                code = code.replace(
                    'from ranger import Ranger',
                    'from torch_optimizer import Ranger'
                )
                code = code.replace(
                    'from ranger import',
                    'from torch_optimizer import'
                )

                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(code)

                logger.info(f"Fixed Ranger import in {py_file}")
                return True

        except Exception as e:
            logger.error(f"Error fixing Ranger import in {py_file}: {e}")

        return False


def _exec_with_timeout(code: str, namespace: dict, timeout: int) -> Optional[Exception]:
    """
    Execute code with a timeout.

    Args:
        code: Python code to execute
        namespace: Namespace for execution
        timeout: Timeout in seconds

    Returns:
        None if successful, Exception if error, or None if timeout (check result_queue)
    """
    result_queue = queue.Queue()

    def exec_code():
        try:
            exec(code, namespace)
            result_queue.put(True)
        except Exception as e:
            result_queue.put(e)

    thread = threading.Thread(target=exec_code, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Timeout - thread is still running
        logger.warning(f"Code execution timeout after {timeout}s")
        return None  # Indicates timeout

    try:
        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            return result
        return True  # Success
    except queue.Empty:
        return None


def _preprocess_student_code(code: str) -> str:
    """
    Preprocess student code to remove problematic patterns.

    This removes or comments out code that might cause issues during loading:
    - Training loops
    - Dataset downloads
    - Main execution blocks that train models
    - DataLoader creation at module level

    Args:
        code: Original student code

    Returns:
        Preprocessed code
    """
    lines = code.split('\n')
    processed_lines = []

    # Patterns that indicate problematic code at module level (not inside functions)
    problematic_patterns = [
        r'^\s*for\s+.*\s+in\s+range\s*\(\s*\d{2,}',  # for i in range(100+)
        r'^\s*while\s+True',  # while True loops
        r'\.fit\s*\(',  # model.fit()
        r'download\s*=\s*True',  # download=True
        r'torchvision\.datasets\.',  # Dataset loading
        r'^\s*train_set\s*=',  # train_set = ...
        r'^\s*test_set\s*=',  # test_set = ...
        r'^\s*train_loader\s*=',  # train_loader = ...
        r'^\s*test_loader\s*=',  # test_loader = ...
        r'DataLoader\s*\(',  # DataLoader(...)
        r'^\s*model\s*=\s*train_model\s*\(',  # model = train_model()
        r'^\s*p_distance\s*\(',  # p_distance(...)
    ]

    # Patterns for code that references disabled variables (also problematic)
    # These catch print statements and other uses of train_set/test_set/loaders
    reference_patterns = [
        r'\btrain_set\b',  # Any reference to train_set
        r'\btest_set\b',   # Any reference to test_set
        r'\btrain_loader\b',  # Any reference to train_loader
        r'\btest_loader\b',   # Any reference to test_loader
    ]

    in_main_block = False
    main_indent = 0
    in_multiline_stmt = False
    paren_balance = 0  # Track parenthesis balance for multi-line statements

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped) if stripped else 0

        # Detect if __name__ == '__main__' block
        if stripped.startswith('if __name__') and '__main__' in stripped:
            in_main_block = True
            main_indent = current_indent
            processed_lines.append(f"# [DISABLED] {line}")
            continue

        # If in main block, comment out everything until dedent
        if in_main_block:
            if stripped and current_indent <= main_indent:
                in_main_block = False
            else:
                processed_lines.append(f"# [DISABLED] {line}" if stripped else line)
                continue

        # Continue disabling multi-line statements
        if in_multiline_stmt:
            processed_lines.append(f"# [DISABLED] {line}")
            # Update parenthesis balance
            paren_balance += line.count('(') - line.count(')')
            # Also count brackets and braces for completeness
            paren_balance += line.count('[') - line.count(']')
            paren_balance += line.count('{') - line.count('}')
            # Check if statement ends (balanced parentheses and no line continuation)
            if paren_balance <= 0 and not line.rstrip().endswith('\\'):
                in_multiline_stmt = False
                paren_balance = 0
            continue

        # Only check module-level code (indent = 0) for problematic patterns
        # Skip patterns inside function/class definitions
        if current_indent == 0:
            is_problematic = False

            # Check primary problematic patterns
            for pattern in problematic_patterns:
                if re.search(pattern, line):
                    is_problematic = True
                    break

            # Also check if line references disabled variables (like print(train_set))
            # But skip if it's a function/class definition or import
            if not is_problematic and not stripped.startswith(('def ', 'class ', 'import ', 'from ')):
                for pattern in reference_patterns:
                    if re.search(pattern, line):
                        is_problematic = True
                        break

            if is_problematic:
                processed_lines.append(f"# [DISABLED] {line}")
                # Check if this is start of multi-line statement
                paren_balance = line.count('(') - line.count(')')
                paren_balance += line.count('[') - line.count(']')
                paren_balance += line.count('{') - line.count('}')
                if line.rstrip().endswith('\\') or paren_balance > 0:
                    in_multiline_stmt = True
                else:
                    paren_balance = 0
                continue

        processed_lines.append(line)

    return '\n'.join(processed_lines)


def load_attack_function(
        py_file: str,
        dependency_manager: Optional[DependencyManager] = None
) -> Tuple[Optional[Callable], DependencyCheckResult]:
    """
    Load the adv_attack function from a student's Python file.

    Args:
        py_file: Path to Python file
        dependency_manager: Optional DependencyManager instance

    Returns:
        Tuple of (attack_function, dependency_result)
    """
    if dependency_manager is None:
        dependency_manager = DependencyManager(auto_install=True)

    # Check and install dependencies
    dep_result = dependency_manager.check_and_install(py_file)

    if not dep_result.success:
        logger.error(f"Dependency check failed for {py_file}")
        return None, dep_result

    # Fix Ranger import if needed
    dependency_manager.fix_ranger_import(py_file)

    try:
        # Read code
        with open(py_file, 'r', encoding='utf-8') as f:
            code = f.read()

        # Preprocess code to remove problematic patterns
        code = _preprocess_student_code(code)

        # Prepare namespace with common imports
        namespace = {
            '__builtins__': __builtins__,
            'torch': importlib.import_module('torch'),
            'np': importlib.import_module('numpy'),
            'numpy': importlib.import_module('numpy'),
        }

        # Add torch_optimizer if available
        try:
            namespace['torch_optimizer'] = importlib.import_module('torch_optimizer')
        except ImportError:
            pass

        # Add AutoAttack if available and required
        if dep_result.has_autoattack:
            try:
                from autoattack import AutoAttack
                namespace['AutoAttack'] = AutoAttack
                namespace['autoattack'] = importlib.import_module('autoattack')
            except ImportError:
                logger.warning(f"AutoAttack required but not available for {py_file}")

        # Execute code in namespace with timeout
        exec_result = _exec_with_timeout(code, namespace, LOAD_TIMEOUT)
        if exec_result is None:
            logger.error(f"Timeout loading {py_file} (>{LOAD_TIMEOUT}s)")
            logger.warning("=" * 60)
            logger.warning(f"[MANUAL CHECK] Student code loading TIMEOUT")
            logger.warning(f"File: {py_file}")
            logger.warning(f"Possible reasons:")
            logger.warning(f"  1. argparse.parse_args() at module level (blocking for input)")
            logger.warning(f"  2. Dataset download (download=True) at module level")
            logger.warning(f"  3. Training loop or heavy computation at module level")
            logger.warning(f"  4. Infinite loop in module-level code")
            logger.warning(f"Please manually check this student's submission.")
            logger.warning("=" * 60)
            return None, dep_result

        if isinstance(exec_result, Exception):
            raise exec_result

        # Check for adv_attack function
        if 'adv_attack' not in namespace:
            logger.error(f"No 'adv_attack' function found in {py_file}")
            return None, dep_result

        original_attack = namespace['adv_attack']

        # Wrap attack function to handle gradient context
        def wrapped_attack(model, X, y, device):
            """Wrapped attack function with proper gradient handling"""
            model.eval()

            import torch

            # Determine if we need gradient context
            if dep_result.requires_grad:
                # Attack uses gradients - ensure grad is enabled
                with torch.enable_grad():
                    X_input = X.clone().detach().requires_grad_(True)
                    result = original_attack(model, X_input, y, device)
            else:
                # Attack doesn't use gradients - try both ways
                try:
                    result = original_attack(model, X, y, device)
                except RuntimeError as e:
                    if 'requires_grad' in str(e) or 'grad' in str(e):
                        # Retry with gradient enabled
                        with torch.enable_grad():
                            X_input = X.clone().detach().requires_grad_(True)
                            result = original_attack(model, X_input, y, device)
                    else:
                        raise

            # Ensure result is detached
            if hasattr(result, 'detach'):
                return result.detach()
            return result

        return wrapped_attack, dep_result

    except SyntaxError as e:
        logger.error(f"Syntax error in {py_file}: {e}")
        return None, dep_result
    except Exception as e:
        logger.error(f"Error loading attack from {py_file}: {e}")
        return None, dep_result
