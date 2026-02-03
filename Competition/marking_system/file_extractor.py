"""
COMP219 Marking System - File Extractor Module
Author: Lingfang Li

Handles extraction and organization of student submission files.
"""

import os
import re
import json
import zipfile
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from utils import logger, extract_student_id, extract_student_id_from_code

# Try to import rarfile (optional dependency)
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False


def convert_ipynb_to_py(ipynb_path: str, py_path: Optional[str] = None) -> Optional[str]:
    """
    Convert a Jupyter notebook (.ipynb) to Python script (.py).

    Args:
        ipynb_path: Path to the .ipynb file
        py_path: Output path for .py file (default: same name with .py extension)

    Returns:
        Path to the created .py file, or None if conversion failed
    """
    if py_path is None:
        py_path = ipynb_path.replace('.ipynb', '.py')

    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        code_lines = []
        code_lines.append(f"# Converted from: {os.path.basename(ipynb_path)}")
        code_lines.append("")

        cells = notebook.get('cells', [])
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])

            if cell_type == 'code':
                # Add code cell
                code_lines.append(f"# --- Cell {i + 1} ---")
                if isinstance(source, list):
                    code_lines.extend([line.rstrip('\n') for line in source])
                else:
                    code_lines.append(source.rstrip('\n'))
                code_lines.append("")
            elif cell_type == 'markdown':
                # Add markdown as comments (first line only for context)
                if source:
                    first_line = source[0] if isinstance(source, list) else source
                    first_line = first_line.strip()[:80]
                    if first_line:
                        code_lines.append(f"# [Markdown]: {first_line}")

        # Write to .py file
        with open(py_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code_lines))

        logger.info(f"Converted {os.path.basename(ipynb_path)} -> {os.path.basename(py_path)}")
        return py_path

    except json.JSONDecodeError as e:
        logger.error(f"Invalid notebook JSON in {ipynb_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to convert {ipynb_path}: {e}")
        return None


@dataclass
class StudentSubmission:
    """Represents a student's submission"""
    student_id: str
    py_file: Optional[str] = None
    pt_file: Optional[str] = None
    has_both_files: bool = False
    error_message: Optional[str] = None

    def __post_init__(self):
        self.has_both_files = self.py_file is not None and self.pt_file is not None


@dataclass
class ExtractionResult:
    """Result of the extraction process"""
    students: Dict[str, StudentSubmission] = field(default_factory=dict)
    total_submissions: int = 0
    successful_extractions: int = 0
    failed_extractions: List[str] = field(default_factory=list)
    missing_py: List[str] = field(default_factory=list)
    missing_pt: List[str] = field(default_factory=list)
    id_mismatches: List[Tuple[str, str, str]] = field(default_factory=list)


class FileExtractor:
    """
    Extracts and organizes student submissions from zip files.

    Handles:
    - Nested zip files
    - Various naming conventions
    - __MACOSX folders
    - ID extraction from filename and code
    - File organization into py_dir and pt_dir
    """

    def __init__(self, config):
        """
        Initialize the file extractor.

        Args:
            config: MarkingConfig instance
        """
        self.config = config
        self.result = ExtractionResult()

    def extract_all(self) -> ExtractionResult:
        """
        Main extraction method - extracts all submissions.

        Returns:
            ExtractionResult with all extracted files and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting file extraction process")
        logger.info("=" * 60)

        # Ensure directories exist
        self.config.ensure_directories()

        # Check if main zip exists
        if not os.path.exists(self.config.submissions_zip):
            logger.error(f"Submissions zip not found: {self.config.submissions_zip}")
            return self.result

        # Extract main zip
        logger.info(f"Extracting main zip: {self.config.submissions_zip}")
        self._extract_main_zip()

        # Process extracted files
        logger.info("Processing extracted files...")
        self._process_extracted_directory()

        # Organize files into py_dir and pt_dir
        logger.info("Organizing files...")
        self._organize_files()

        # Generate summary
        self._generate_summary()

        return self.result

    def _extract_main_zip(self):
        """Extract the main submissions zip file"""
        try:
            with zipfile.ZipFile(self.config.submissions_zip, 'r') as zip_ref:
                zip_ref.extractall(self.config.extracted_dir)
            logger.info(f"Extracted to: {self.config.extracted_dir}")
        except Exception as e:
            logger.error(f"Failed to extract main zip: {e}")
            raise

        # Remove __MACOSX folders
        self._remove_macosx_folders(self.config.extracted_dir)

    def _remove_macosx_folders(self, directory: str):
        """Remove __MACOSX folders recursively"""
        for root, dirs, files in os.walk(directory, topdown=False):
            for dir_name in dirs:
                if dir_name == "__MACOSX":
                    macosx_path = os.path.join(root, dir_name)
                    logger.debug(f"Removing __MACOSX folder: {macosx_path}")
                    shutil.rmtree(macosx_path, ignore_errors=True)

    def _process_extracted_directory(self):
        """Process the extracted directory to find all student submissions"""
        self.result.total_submissions = 0

        for item in os.listdir(self.config.extracted_dir):
            item_path = os.path.join(self.config.extracted_dir, item)

            # Skip hidden files
            if item.startswith('.'):
                continue

            self.result.total_submissions += 1

            try:
                if zipfile.is_zipfile(item_path):
                    # It's a nested zip file (student submission)
                    self._process_student_zip(item_path)
                elif os.path.isdir(item_path):
                    # It's a folder (already extracted submission)
                    self._process_student_folder(item_path)
                elif item.endswith('.py') or item.endswith('.pt'):
                    # Individual file
                    self._process_individual_file(item_path)
            except Exception as e:
                logger.error(f"Error processing {item}: {e}")
                self.result.failed_extractions.append(item)

    # Directories to skip during extraction (common mistakes in student submissions)
    SKIP_DIRECTORIES = {
        '.venv', 'venv', '.env', 'env',           # Virtual environments
        '__pycache__', '.git', '.idea',            # Cache and IDE folders
        'site-packages', 'dist-packages',          # Package directories
        '__MACOSX', '.DS_Store',                   # MacOS artifacts
        'node_modules', '.npm',                    # Node.js (if any)
        'data', 'FashionMNIST',                    # Dataset folders
    }

    def _should_extract_file(self, filename: str) -> bool:
        """
        Check if a file should be extracted.
        Extract .py, .pt, .ipynb, and nested archive files (.rar, .zip, .7z).
        """
        # Skip files in problematic directories
        path_parts = filename.replace('\\', '/').split('/')
        for part in path_parts:
            if part in self.SKIP_DIRECTORIES:
                return False
            # Skip hidden directories/files
            if part.startswith('.') and part not in ['.py', '.pt', '.ipynb']:
                return False

        basename = os.path.basename(filename)
        if basename.startswith('._'):  # MacOS resource forks
            return False

        # Extract these file types
        allowed_extensions = ('.py', '.pt', '.ipynb', '.rar', '.zip', '.7z')
        return filename.endswith(allowed_extensions)

    def _extract_rar_file(self, rar_path: str, target_dir: str) -> bool:
        """
        Extract a RAR file to target directory.

        Args:
            rar_path: Path to .rar file
            target_dir: Directory to extract to

        Returns:
            True if extraction succeeded
        """
        if RARFILE_AVAILABLE:
            try:
                with rarfile.RarFile(rar_path, 'r') as rar_ref:
                    for rar_info in rar_ref.infolist():
                        if rar_info.is_dir():
                            continue
                        if self._should_extract_file(rar_info.filename):
                            basename = os.path.basename(rar_info.filename)
                            target_path = os.path.join(target_dir, basename)
                            # Handle duplicates
                            if os.path.exists(target_path):
                                name, ext = os.path.splitext(basename)
                                target_path = os.path.join(target_dir, f"{name}_dup{ext}")
                            try:
                                with rar_ref.open(rar_info) as source:
                                    with open(target_path, 'wb') as target:
                                        target.write(source.read())
                                logger.debug(f"Extracted from RAR: {rar_info.filename} -> {basename}")
                            except Exception as e:
                                logger.warning(f"Could not extract {rar_info.filename} from RAR: {e}")
                return True
            except Exception as e:
                logger.error(f"Failed to extract RAR {rar_path}: {e}")
                return False
        else:
            # Try using system unrar command
            try:
                subprocess.run(
                    ['unrar', 'x', '-y', rar_path, target_dir],
                    check=True, capture_output=True
                )
                logger.info(f"Extracted RAR using system unrar: {rar_path}")
                return True
            except FileNotFoundError:
                logger.error(f"Cannot extract RAR: rarfile not installed and unrar not found. File: {rar_path}")
                return False
            except subprocess.CalledProcessError as e:
                logger.error(f"unrar failed for {rar_path}: {e}")
                return False

    def _extract_nested_archive(self, archive_path: str, target_dir: str) -> bool:
        """
        Extract a nested archive file (zip, rar, 7z).

        Args:
            archive_path: Path to archive file
            target_dir: Directory to extract to

        Returns:
            True if extraction succeeded
        """
        ext = os.path.splitext(archive_path)[1].lower()

        if ext == '.zip':
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.is_dir():
                            continue
                        if self._should_extract_file(zip_info.filename):
                            basename = os.path.basename(zip_info.filename)
                            target_path = os.path.join(target_dir, basename)
                            if os.path.exists(target_path):
                                name, ext2 = os.path.splitext(basename)
                                target_path = os.path.join(target_dir, f"{name}_dup{ext2}")
                            try:
                                with zip_ref.open(zip_info) as source:
                                    with open(target_path, 'wb') as target:
                                        target.write(source.read())
                            except Exception as e:
                                logger.warning(f"Could not extract {zip_info.filename}: {e}")
                return True
            except Exception as e:
                logger.error(f"Failed to extract nested ZIP {archive_path}: {e}")
                return False

        elif ext == '.rar':
            return self._extract_rar_file(archive_path, target_dir)

        elif ext == '.7z':
            # Try using system 7z command
            try:
                subprocess.run(
                    ['7z', 'x', '-y', f'-o{target_dir}', archive_path],
                    check=True, capture_output=True
                )
                logger.info(f"Extracted 7z: {archive_path}")
                return True
            except FileNotFoundError:
                logger.error(f"Cannot extract 7z: 7z command not found. File: {archive_path}")
                return False
            except subprocess.CalledProcessError as e:
                logger.error(f"7z extraction failed for {archive_path}: {e}")
                return False

        return False

    def _process_student_zip(self, zip_path: str):
        """Process a single student's zip file with smart extraction"""
        student_id = extract_student_id(zip_path)

        # Create temporary extraction directory
        temp_dir = os.path.join(
            self.config.extracted_dir,
            f"temp_{os.path.splitext(os.path.basename(zip_path))[0]}"
        )

        try:
            os.makedirs(temp_dir, exist_ok=True)

            # Smart extraction: only extract .py and .pt files
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for zip_info in zip_ref.infolist():
                    # Skip directories
                    if zip_info.is_dir():
                        continue

                    # Check if file should be extracted
                    if self._should_extract_file(zip_info.filename):
                        # Extract to flat structure (no nested folders)
                        basename = os.path.basename(zip_info.filename)
                        target_path = os.path.join(temp_dir, basename)

                        # Handle duplicate filenames
                        if os.path.exists(target_path):
                            name, ext = os.path.splitext(basename)
                            target_path = os.path.join(temp_dir, f"{name}_dup{ext}")

                        # Extract file
                        try:
                            with zip_ref.open(zip_info) as source:
                                with open(target_path, 'wb') as target:
                                    target.write(source.read())
                            logger.debug(f"Extracted: {zip_info.filename} -> {basename}")
                        except Exception as e:
                            logger.warning(f"Could not extract {zip_info.filename}: {e}")

            # Process the extracted folder
            self._process_student_folder(temp_dir, student_id)

        except zipfile.BadZipFile as e:
            logger.error(f"Bad zip file {zip_path}: {e}")
            if student_id:
                self.result.failed_extractions.append(student_id)
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            if student_id:
                self.result.failed_extractions.append(student_id)

    def _get_file_mtime(self, file_path: str) -> float:
        """Get file modification time, return 0 if error"""
        try:
            return os.path.getmtime(file_path)
        except:
            return 0

    def _select_best_py_file(self, py_files: List[str], student_id: str) -> Tuple[str, bool]:
        """
        Select the best .py file from multiple candidates.

        Priority:
        1. Most recently modified file (PRIMARY)
        2. Among files with same mtime: prefer competition_{student_id}.py
        3. Among files with same mtime: prefer file containing 'competition'

        Returns:
            Tuple of (selected_file, is_certain)
        """
        if len(py_files) == 1:
            return py_files[0], True

        # Sort by modification time (newest first)
        sorted_files = sorted(py_files, key=self._get_file_mtime, reverse=True)
        newest_file = sorted_files[0]
        newest_mtime = self._get_file_mtime(newest_file)

        # Get all files with the same newest mtime (within 1 second tolerance)
        newest_group = [f for f in sorted_files if abs(self._get_file_mtime(f) - newest_mtime) < 1]

        if len(newest_group) == 1:
            # Only one newest file
            basename = os.path.basename(newest_file).lower()
            is_certain = 'competition' in basename or student_id in basename
            return newest_file, is_certain

        # Multiple files with same mtime - use name matching as tiebreaker
        # Tiebreaker 1: Exact match competition_{student_id}.py
        for f in newest_group:
            if os.path.basename(f) == f"competition_{student_id}.py":
                return f, True

        # Tiebreaker 2: Contains 'competition' and student_id
        for f in newest_group:
            basename = os.path.basename(f).lower()
            if 'competition' in basename and student_id in basename:
                return f, True

        # Tiebreaker 3: Contains 'competition'
        for f in newest_group:
            if 'competition' in os.path.basename(f).lower():
                return f, True

        # Return the newest file
        return newest_file, False

    def _select_best_pt_file(self, pt_files: List[str], student_id: str) -> Tuple[str, bool]:
        """
        Select the best .pt file from multiple candidates.

        Priority:
        1. Most recently modified file (PRIMARY)
        2. Among files with same mtime: prefer {student_id}.pt
        3. Among files with same mtime: prefer file containing student_id

        Returns:
            Tuple of (selected_file, is_certain)
        """
        if len(pt_files) == 1:
            return pt_files[0], True

        # Sort by modification time (newest first)
        sorted_files = sorted(pt_files, key=self._get_file_mtime, reverse=True)
        newest_file = sorted_files[0]
        newest_mtime = self._get_file_mtime(newest_file)

        # Get all files with the same newest mtime (within 1 second tolerance)
        newest_group = [f for f in sorted_files if abs(self._get_file_mtime(f) - newest_mtime) < 1]

        if len(newest_group) == 1:
            # Only one newest file
            is_certain = student_id in os.path.basename(newest_file)
            return newest_file, is_certain

        # Multiple files with same mtime - use name matching as tiebreaker
        # Tiebreaker 1: Exact match {student_id}.pt
        for f in newest_group:
            if os.path.basename(f) == f"{student_id}.pt":
                return f, True

        # Tiebreaker 2: Contains student_id
        for f in newest_group:
            if student_id in os.path.basename(f):
                return f, True

        # Return the newest file (uncertain selection)
        logger.warning("=" * 60)
        logger.warning(f"[MANUAL CHECK] Multiple .pt files, uncertain selection")
        logger.warning(f"Student: {student_id}")
        logger.warning(f"Files: {[os.path.basename(f) for f in pt_files]}")
        logger.warning(f"Selected: {os.path.basename(newest_file)}")
        logger.warning(f"Please verify this is the correct model file.")
        logger.warning("=" * 60)
        return newest_file, False

    def _process_student_folder(self, folder_path: str, fallback_id: Optional[str] = None):
        """Process a student's extracted folder"""
        py_files = []
        pt_files = []
        ipynb_files = []
        nested_archives = []

        # Find all .py, .pt, .ipynb files and nested archives recursively
        for root, dirs, files in os.walk(folder_path):
            # Skip problematic directories
            dirs[:] = [d for d in dirs
                       if d not in self.SKIP_DIRECTORIES
                       and not d.startswith('.')]

            for file in files:
                if file.startswith('.') or file.startswith('._'):
                    continue

                file_path = os.path.join(root, file)
                file_lower = file.lower()

                if file.endswith('.py'):
                    py_files.append(file_path)
                elif file.endswith('.pt'):
                    pt_files.append(file_path)
                elif file.endswith('.ipynb'):
                    ipynb_files.append(file_path)
                elif file_lower.endswith(('.rar', '.zip', '.7z')):
                    nested_archives.append(file_path)

        # Extract nested archives if we don't have both .py and .pt files yet
        if nested_archives and (not py_files or not pt_files):
            logger.info(f"Found {len(nested_archives)} nested archive(s), extracting...")
            for archive_path in nested_archives:
                logger.debug(f"Extracting nested archive: {os.path.basename(archive_path)}")
                if self._extract_nested_archive(archive_path, folder_path):
                    # Re-scan for new files after extraction
                    for root, dirs, files in os.walk(folder_path):
                        dirs[:] = [d for d in dirs
                                   if d not in self.SKIP_DIRECTORIES
                                   and not d.startswith('.')]
                        for file in files:
                            if file.startswith('.') or file.startswith('._'):
                                continue
                            file_path = os.path.join(root, file)
                            if file.endswith('.py') and file_path not in py_files:
                                py_files.append(file_path)
                            elif file.endswith('.pt') and file_path not in pt_files:
                                pt_files.append(file_path)
                            elif file.endswith('.ipynb') and file_path not in ipynb_files:
                                ipynb_files.append(file_path)

        # Convert .ipynb files to .py if no .py files found (or if ipynb contains 'competition')
        if ipynb_files:
            # Check if any ipynb has 'competition' in name - prioritize that
            competition_ipynb = [f for f in ipynb_files if 'competition' in os.path.basename(f).lower()]

            if not py_files or competition_ipynb:
                # Convert notebooks
                notebooks_to_convert = competition_ipynb if competition_ipynb else ipynb_files
                for ipynb_file in notebooks_to_convert:
                    converted_py = convert_ipynb_to_py(ipynb_file)
                    if converted_py:
                        py_files.append(converted_py)

        # Extract student ID
        student_id = fallback_id or extract_student_id(folder_path)

        # Try to get ID from py file content
        if py_files and not student_id:
            student_id = extract_student_id_from_code(py_files[0])

        # Try to get ID from pt file name
        if pt_files and not student_id:
            student_id = extract_student_id(pt_files[0])

        if not student_id:
            logger.warning(f"Could not extract student ID from {folder_path}")
            self.result.failed_extractions.append(folder_path)
            return

        # Create or update student submission
        if student_id not in self.result.students:
            self.result.students[student_id] = StudentSubmission(student_id=student_id)

        submission = self.result.students[student_id]

        # Smart file selection for .py files
        if py_files:
            selected_py, py_certain = self._select_best_py_file(py_files, student_id)
            submission.py_file = selected_py
            if len(py_files) > 1:
                if py_certain:
                    # Found the right file, just log at debug level
                    logger.debug(f"Student {student_id}: selected {os.path.basename(selected_py)} from {len(py_files)} .py files")
                else:
                    # Uncertain selection, warn
                    logger.warning(f"Student {student_id}: multiple .py files, uncertain selection: {os.path.basename(selected_py)}")

        # Smart file selection for .pt files
        if pt_files:
            selected_pt, pt_certain = self._select_best_pt_file(pt_files, student_id)
            submission.pt_file = selected_pt
            if len(pt_files) > 1:
                if pt_certain:
                    logger.debug(f"Student {student_id}: selected {os.path.basename(selected_pt)} from {len(pt_files)} .pt files")
                else:
                    logger.warning(f"Student {student_id}: multiple .pt files, uncertain selection: {os.path.basename(selected_pt)}")

        submission.has_both_files = submission.py_file is not None and submission.pt_file is not None

        # Check for ID mismatch between py and pt
        if submission.py_file and submission.pt_file:
            py_id = extract_student_id(submission.py_file)
            pt_id = extract_student_id(submission.pt_file)
            code_id = extract_student_id_from_code(submission.py_file)

            ids = {student_id, py_id, pt_id, code_id} - {None}
            if len(ids) > 1:
                self.result.id_mismatches.append((student_id, str(ids), folder_path))
                logger.warning(f"ID mismatch for {student_id}: found IDs {ids}")

        self.result.successful_extractions += 1

    def _process_individual_file(self, file_path: str):
        """Process an individual .py or .pt file"""
        student_id = extract_student_id(file_path)

        if not student_id and file_path.endswith('.py'):
            student_id = extract_student_id_from_code(file_path)

        if not student_id:
            logger.warning(f"Could not extract student ID from {file_path}")
            return

        if student_id not in self.result.students:
            self.result.students[student_id] = StudentSubmission(student_id=student_id)

        submission = self.result.students[student_id]

        if file_path.endswith('.py'):
            submission.py_file = file_path
        elif file_path.endswith('.pt'):
            submission.pt_file = file_path

        submission.has_both_files = submission.py_file is not None and submission.pt_file is not None

    def _organize_files(self):
        """Copy and rename files to organized directories"""
        for student_id, submission in self.result.students.items():
            # Copy and rename .py file
            if submission.py_file:
                new_py_name = f"competition_{student_id}.py"
                new_py_path = os.path.join(self.config.py_dir, new_py_name)
                try:
                    shutil.copy2(submission.py_file, new_py_path)
                    submission.py_file = new_py_path
                except Exception as e:
                    logger.error(f"Failed to copy py file for {student_id}: {e}")
            else:
                self.result.missing_py.append(student_id)

            # Copy and rename .pt file
            if submission.pt_file:
                new_pt_name = f"{student_id}.pt"
                new_pt_path = os.path.join(self.config.pt_dir, new_pt_name)
                try:
                    shutil.copy2(submission.pt_file, new_pt_path)
                    submission.pt_file = new_pt_path
                except Exception as e:
                    logger.error(f"Failed to copy pt file for {student_id}: {e}")
            else:
                self.result.missing_pt.append(student_id)

    def _generate_summary(self):
        """Generate and log extraction summary"""
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total submissions processed:    {self.result.total_submissions}")
        logger.info(f"Successful extractions:         {self.result.successful_extractions}")
        logger.info(f"Failed extractions:             {len(self.result.failed_extractions)}")
        if self.result.failed_extractions:
            logger.warning(f"[MANUAL CHECK] Failed to extract these submissions:")
            for failed in self.result.failed_extractions:
                logger.warning(f"  - {failed}")
        logger.info(f"Students with both files:       {sum(1 for s in self.result.students.values() if s.has_both_files)}")
        logger.info(f"Students missing .py file:      {len(self.result.missing_py)}")
        logger.info(f"Students missing .pt file:      {len(self.result.missing_pt)}")
        logger.info(f"ID mismatches detected:         {len(self.result.id_mismatches)}")

        if self.result.missing_py:
            logger.warning(f"\nStudents missing .py: {self.result.missing_py}")
        if self.result.missing_pt:
            logger.warning(f"\nStudents missing .pt: {self.result.missing_pt}")
        if self.result.id_mismatches:
            logger.warning(f"\nID mismatches: {self.result.id_mismatches}")

        logger.info("=" * 60)

    def get_valid_students(self) -> List[str]:
        """
        Get list of student IDs that have both .py and .pt files.

        Returns:
            List of valid student IDs
        """
        return [
            student_id for student_id, submission in self.result.students.items()
            if submission.has_both_files
        ]

    def get_student_files(self, student_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get file paths for a specific student.

        Args:
            student_id: Student ID

        Returns:
            Tuple of (py_file_path, pt_file_path)
        """
        if student_id in self.result.students:
            submission = self.result.students[student_id]
            return submission.py_file, submission.pt_file
        return None, None


def find_new_students(
        existing_ids: Set[str],
        extracted_result: ExtractionResult
) -> Set[str]:
    """
    Find students that are in the new extraction but not in existing results.

    Args:
        existing_ids: Set of existing student IDs
        extracted_result: New extraction result

    Returns:
        Set of new student IDs
    """
    new_ids = set(extracted_result.students.keys()) - existing_ids
    return new_ids


def find_missing_students(
        expected_ids: Set[str],
        extracted_result: ExtractionResult
) -> Set[str]:
    """
    Find students that were expected but not found in extraction.

    Args:
        expected_ids: Set of expected student IDs
        extracted_result: Extraction result

    Returns:
        Set of missing student IDs
    """
    found_ids = set(extracted_result.students.keys())
    return expected_ids - found_ids
