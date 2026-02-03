"""
COMP219 Marking System - Main Controller
Author: Lingfang Li

Unified marking system with automatic scenario detection.

Usage:
    python main.py                    # Full pipeline with auto-detection
    python main.py --extract-only     # Only extract files
    python main.py --evaluate-only    # Only run evaluation (files already extracted)
    python main.py --grade-only       # Only calculate grades from existing results
    python main.py --rerun-attacks ID1 ID2  # Re-run specific attacks
    python main.py --rerun-models ID1 ID2   # Re-run specific models
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Optional, Set

from config import MarkingConfig, ScenarioType
from utils import logger, setup_logging, set_random_seed, get_student_ids_from_directory
from file_extractor import FileExtractor, ExtractionResult
from evaluator import Evaluator, IncrementalEvaluator
from grade_calculator import GradeCalculator, load_and_average_results


class MarkingSystem:
    """
    Unified marking system for COMP219 competition.

    Features:
    - Automatic scenario detection
    - Full pipeline: Extract -> Evaluate -> Grade
    - Incremental updates
    - Re-run specific attacks/models
    """

    def __init__(self, config: Optional[MarkingConfig] = None):
        """
        Initialize the marking system.

        Args:
            config: MarkingConfig instance (uses default if None)
        """
        self.config = config or MarkingConfig()
        self.extraction_result: Optional[ExtractionResult] = None
        self.evaluator: Optional[Evaluator] = None
        self.grade_calculator: Optional[GradeCalculator] = None

    def detect_scenario(self) -> str:
        """
        Automatically detect the current scenario.

        Returns:
            ScenarioType constant
        """
        has_extracted = os.path.exists(self.config.py_dir) and os.path.exists(self.config.pt_dir)
        has_results = any(
            f.endswith('.csv') for f in os.listdir(self.config.output_dir)
        ) if os.path.exists(self.config.output_dir) else False

        if not has_extracted:
            logger.info("Scenario: FULL_EVALUATION (no extracted files found)")
            return ScenarioType.FULL_EVALUATION
        elif has_extracted and not has_results:
            logger.info("Scenario: Files extracted, need evaluation")
            return ScenarioType.FULL_EVALUATION
        else:
            logger.info("Scenario: Results exist, checking for updates...")
            # Could detect new students here
            return ScenarioType.FULL_EVALUATION

    def run_full_pipeline(self):
        """Run the complete marking pipeline: Extract -> Evaluate -> Grade"""
        logger.info("\n" + "=" * 70)
        logger.info("COMP219 MARKING SYSTEM - FULL PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Submissions: {self.config.submissions_zip}")
        logger.info(f"Work directory: {self.config.work_dir}")
        logger.info("=" * 70)

        # Step 1: Extract files
        self.run_extraction()

        # Step 2: Run evaluation
        self.run_evaluation()

        # Step 3: Calculate grades
        self.run_grading()

        logger.info("\n" + "=" * 70)
        logger.info("MARKING COMPLETE")
        logger.info(f"Results saved to: {self.config.output_dir}")
        logger.info("=" * 70)

    def run_extraction(self) -> ExtractionResult:
        """
        Run file extraction only.

        Returns:
            ExtractionResult with extracted files
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: FILE EXTRACTION")
        logger.info("=" * 60)

        extractor = FileExtractor(self.config)
        self.extraction_result = extractor.extract_all()

        return self.extraction_result

    def run_evaluation(self, student_ids: Optional[List[str]] = None,
                       partition: Optional[tuple] = None,
                       start_round: int = 1) -> List:
        """
        Run evaluation only (assumes files already extracted).

        Args:
            student_ids: Optional list of student IDs to evaluate
            partition: Optional tuple (part_num, total_parts) for distributed evaluation
            start_round: Starting round number (default: 1)

        Returns:
            List of evaluation DataFrames
        """
        logger.info("\n" + "=" * 60)
        if partition:
            logger.info(f"STEP 2: MUTUAL EVALUATION - PARTITION {partition[0]}/{partition[1]}")
        else:
            logger.info("STEP 2: MUTUAL EVALUATION")
        logger.info("=" * 60)

        # Get student IDs
        if student_ids is None:
            if self.extraction_result:
                # Include students with at least one file (.py or .pt)
                student_ids = [
                    sid for sid, sub in self.extraction_result.students.items()
                    if sub.py_file is not None or sub.pt_file is not None
                ]
            else:
                # Load from directories
                student_ids = self._get_student_ids_from_directories()

        if not student_ids:
            logger.error("No valid students found for evaluation")
            return []

        logger.info(f"Total students: {len(student_ids)}")

        # Create evaluator with partition support
        self.evaluator = Evaluator(self.config, partition=partition)

        # Load components
        self.evaluator.load_student_components(
            student_ids,
            self.config.py_dir,
            self.config.pt_dir
        )

        # Run multiple rounds
        results = self.evaluator.run_multiple_rounds(start_round=start_round)

        # Save results
        self.evaluator.save_results()

        return results

    def run_grading(self, result_files: Optional[List[str]] = None):
        """
        Calculate grades from evaluation results.

        Args:
            result_files: Optional list of result CSV files
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: GRADE CALCULATION")
        logger.info("=" * 60)

        # Find result files
        if result_files is None:
            result_files = self._find_result_files()

        if not result_files:
            logger.error("No evaluation results found for grading")
            return

        logger.info(f"Using {len(result_files)} result files")

        # Calculate grades
        grades, averaged_matrix = load_and_average_results(result_files, self.config)

        # Save results
        self.grade_calculator = GradeCalculator(self.config)
        self.grade_calculator.student_grades = grades
        self.grade_calculator.save_results(self.config.output_dir)

    def run_rerun_attacks(self, attack_ids: List[str], existing_matrix: str = None,
                          round_num: int = None, rounds: List[int] = None,
                          matrix_pattern: str = None):
        """
        Re-run specific attack evaluations.

        Args:
            attack_ids: List of student IDs whose attacks to re-run
            existing_matrix: Path to existing evaluation matrix (for single round)
            round_num: Single round number (1, 2, 3...)
            rounds: List of round numbers to process
            matrix_pattern: Pattern for matrix files (e.g., "evaluation_round_{}.csv")
        """
        # Determine which rounds to process
        if rounds:
            rounds_to_process = rounds
        elif round_num is not None:
            rounds_to_process = [round_num]
        else:
            rounds_to_process = [1]  # Default to round 1

        # Determine matrix pattern
        if matrix_pattern is None:
            matrix_pattern = os.path.join(self.config.output_dir, "evaluation_round_{}.csv")

        for rnd in rounds_to_process:
            # Get matrix file for this round
            if existing_matrix and len(rounds_to_process) == 1:
                matrix_file = existing_matrix
            else:
                matrix_file = matrix_pattern.format(rnd)

            if not os.path.exists(matrix_file):
                logger.warning(f"Skipping round {rnd}: {matrix_file} not found")
                continue

            logger.info("\n" + "=" * 60)
            logger.info(f"RE-RUNNING ATTACKS: {attack_ids} (Round {rnd})")
            logger.info(f"Matrix: {matrix_file}")
            logger.info("=" * 60)

            evaluator = IncrementalEvaluator(self.config, matrix_file)
            test_loader = evaluator.prepare_test_data(round_num=rnd)

            evaluator.rerun_specific_attacks(
                attack_ids,
                self.config.py_dir,
                self.config.pt_dir,
                test_loader,
                output_path=matrix_file
            )

    def run_rerun_models(self, model_ids: List[str], existing_matrix: str = None,
                         round_num: int = None, rounds: List[int] = None,
                         matrix_pattern: str = None):
        """
        Re-run specific model evaluations.

        Args:
            model_ids: List of student IDs whose models to re-run
            existing_matrix: Path to existing evaluation matrix (for single round)
            round_num: Single round number (1, 2, 3...)
            rounds: List of round numbers to process
            matrix_pattern: Pattern for matrix files (e.g., "evaluation_round_{}.csv")
        """
        # Determine which rounds to process
        if rounds:
            rounds_to_process = rounds
        elif round_num is not None:
            rounds_to_process = [round_num]
        else:
            rounds_to_process = [1]  # Default to round 1

        # Determine matrix pattern
        if matrix_pattern is None:
            matrix_pattern = os.path.join(self.config.output_dir, "evaluation_round_{}.csv")

        for rnd in rounds_to_process:
            # Get matrix file for this round
            if existing_matrix and len(rounds_to_process) == 1:
                matrix_file = existing_matrix
            else:
                matrix_file = matrix_pattern.format(rnd)

            if not os.path.exists(matrix_file):
                logger.warning(f"Skipping round {rnd}: {matrix_file} not found")
                continue

            logger.info("\n" + "=" * 60)
            logger.info(f"RE-RUNNING MODELS: {model_ids} (Round {rnd})")
            logger.info(f"Matrix: {matrix_file}")
            logger.info("=" * 60)

            evaluator = IncrementalEvaluator(self.config, matrix_file)
            test_loader = evaluator.prepare_test_data(round_num=rnd)

            evaluator.rerun_specific_models(
                model_ids,
                self.config.py_dir,
                self.config.pt_dir,
                test_loader,
                output_path=matrix_file
            )

    def run_update_from_fixed(self, existing_matrix: str) -> List:
        """
        Update evaluation matrix with fixed files.

        This method:
        1. Loads fixed attacks from fixed_py_dir
        2. Loads fixed models from fixed_pt_dir
        3. Evaluates fixed attacks against ALL models (original + fixed)
        4. Evaluates ALL attacks against fixed models (avoiding duplicates)
        5. Updates the evaluation matrix

        Args:
            existing_matrix: Path to existing evaluation matrix CSV

        Returns:
            List of updated evaluation DataFrames
        """
        from evaluator import FixedFilesEvaluator

        logger.info("\n" + "=" * 60)
        logger.info("UPDATE FROM FIXED FILES")
        logger.info("=" * 60)
        logger.info(f"Existing matrix: {existing_matrix}")
        logger.info(f"Fixed py dir: {self.config.fixed_py_dir}")
        logger.info(f"Fixed pt dir: {self.config.fixed_pt_dir}")

        # Create evaluator
        evaluator = FixedFilesEvaluator(self.config, existing_matrix)

        # Run update
        results = evaluator.run_update(
            self.config.py_dir,
            self.config.pt_dir,
            self.config.fixed_py_dir,
            self.config.fixed_pt_dir
        )

        logger.info("\n" + "=" * 60)
        logger.info("UPDATE FROM FIXED FILES COMPLETE")
        logger.info("=" * 60)

        return results

    def _get_student_ids_from_directories(self) -> List[str]:
        """Get student IDs from py and pt directories"""
        py_ids = get_student_ids_from_directory(self.config.py_dir, '.py')
        pt_ids = get_student_ids_from_directory(self.config.pt_dir, '.pt')

        # Return students with at least one file (.py or .pt)
        valid_ids = py_ids | pt_ids
        return sorted(list(valid_ids))

    def _find_result_files(self) -> List[str]:
        """Find evaluation result CSV files"""
        if not os.path.exists(self.config.output_dir):
            return []

        files = []
        for f in os.listdir(self.config.output_dir):
            if f.startswith('evaluation_round_') and f.endswith('.csv'):
                files.append(os.path.join(self.config.output_dir, f))

        return sorted(files)


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='COMP219 Marking System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run full pipeline
  python main.py --extract-only           # Only extract files
  python main.py --evaluate-only          # Only run evaluation
  python main.py --grade-only             # Only calculate grades
  python main.py --submissions path/to/submissions.zip

Re-run attacks/models (single round):
  python main.py --rerun-attacks 201234567 201234568 --round 1
  python main.py --rerun-models 201234567 --existing-matrix path/to/matrix.csv --round 2

Re-run attacks/models (multiple rounds):
  python main.py --rerun-attacks 201234567 --rounds 1,2,3
  python main.py --rerun-models 201234567 201234568 --rounds 1,2,3
  python main.py --rerun-attacks 201234567 --rounds 1,2,3 --matrix-pattern custom_round_{}.csv

Distributed Evaluation:
  python main.py --evaluate-only --partition 1/3   # Run partition 1 of 3
  python main.py --evaluate-only --partition 2/3   # Run partition 2 of 3
  python main.py --evaluate-only --partition 3/3   # Run partition 3 of 3
  python main.py --merge-partitions 3              # Merge 3 partitions
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--extract-only', action='store_true',
                            help='Only extract and organize files')
    mode_group.add_argument('--evaluate-only', action='store_true',
                            help='Only run evaluation (files must be extracted)')
    mode_group.add_argument('--grade-only', action='store_true',
                            help='Only calculate grades from existing results')
    mode_group.add_argument('--rerun-attacks', nargs='+', metavar='ID',
                            help='Re-run specific attack evaluations')
    mode_group.add_argument('--rerun-models', nargs='+', metavar='ID',
                            help='Re-run specific model evaluations')
    mode_group.add_argument('--merge-partitions', type=int, metavar='N',
                            help='Merge N partition results into complete matrices')
    mode_group.add_argument('--update-from-fixed', action='store_true',
                            help='Update evaluation matrix with fixed files from fixed_py_files and fixed_pt_files')

    # Partition support for distributed evaluation
    parser.add_argument('--partition', type=str, metavar='N/M',
                        help='Run partition N of M (e.g., 1/3 for first of 3 partitions)')

    # Configuration overrides
    parser.add_argument('--submissions', type=str,
                        help='Path to submissions zip file')
    parser.add_argument('--work-dir', type=str,
                        help='Working directory for the marking system')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for results')
    parser.add_argument('--num-rounds', type=int, default=None,
                        help='Number of evaluation rounds (default: from config, 3)')
    parser.add_argument('--start-round', type=int, default=1,
                        help='Starting round number (default: 1)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of test samples (default: from config, 500)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: from config, 88)')
    parser.add_argument('--existing-matrix', type=str,
                        help='Path to existing evaluation matrix (for single round rerun)')
    parser.add_argument('--round', type=int, default=None,
                        help='Single round number for rerun (default: None, use --rounds instead)')
    parser.add_argument('--rounds', type=str, default=None,
                        help='Comma-separated round numbers for rerun (e.g., "1,2,3")')
    parser.add_argument('--matrix-pattern', type=str, default='evaluation_round_{}.csv',
                        help='Matrix file pattern with {} for round number (default: evaluation_round_{}.csv)')

    args = parser.parse_args()

    # Create configuration - collect overrides first, then construct once
    # so that __post_init__ derives all paths from the correct work_dir
    config_kwargs = {}
    if args.submissions:
        config_kwargs['submissions_zip'] = args.submissions
    if args.work_dir:
        config_kwargs['work_dir'] = args.work_dir
    if args.output_dir:
        config_kwargs['output_dir'] = args.output_dir
    if args.num_rounds is not None:
        config_kwargs['num_rounds'] = args.num_rounds
    if args.num_samples is not None:
        config_kwargs['num_test_samples'] = args.num_samples
    if args.seed is not None:
        config_kwargs['random_seed'] = args.seed

    config = MarkingConfig(**config_kwargs)

    # Parse partition argument
    partition = None
    if args.partition:
        try:
            part_num, total_parts = args.partition.split('/')
            partition = (int(part_num), int(total_parts))
            if partition[0] < 1 or partition[0] > partition[1]:
                raise ValueError("Invalid partition range")
            logger.info(f"Running partition {partition[0]} of {partition[1]}")
        except Exception as e:
            logger.error(f"Invalid partition format '{args.partition}'. Use N/M format (e.g., 1/3)")
            sys.exit(1)

    # Handle merge-partitions mode early (doesn't need full validation)
    if args.merge_partitions:
        from evaluator import merge_partition_results
        config.ensure_directories()
        setup_logging(config.log_level, None)
        merge_partition_results(config.output_dir, args.merge_partitions, config.num_rounds)
        logger.info("Partition merge complete. You can now run --grade-only to calculate grades.")
        sys.exit(0)

    # Validate configuration
    errors = config.validate()
    if errors and not (args.evaluate_only or args.grade_only or args.update_from_fixed or args.rerun_attacks or args.rerun_models):
        for error in errors:
            logger.error(error)
        sys.exit(1)

    # Ensure directories
    config.ensure_directories()

    # Setup logging to file
    partition_suffix = f"_part{partition[0]}of{partition[1]}" if partition else ""
    log_file = os.path.join(config.output_dir, f"marking_{datetime.now().strftime('%Y%m%d_%H%M%S')}{partition_suffix}.log")
    setup_logging(config.log_level, log_file)

    # Create marking system
    system = MarkingSystem(config)

    # Parse rounds argument
    rounds_list = None
    if args.rounds:
        rounds_list = [int(r.strip()) for r in args.rounds.split(',')]

    # Resolve matrix pattern to absolute path
    matrix_pattern = args.matrix_pattern
    if matrix_pattern and not os.path.isabs(matrix_pattern):
        matrix_pattern = os.path.join(config.output_dir, matrix_pattern)

    # Run appropriate mode
    if args.extract_only:
        system.run_extraction()
    elif args.evaluate_only:
        system.run_evaluation(partition=partition, start_round=args.start_round)
    elif args.grade_only:
        system.run_grading()
    elif args.rerun_attacks:
        # Either --existing-matrix (single file) or --rounds (multiple files via pattern)
        if not args.existing_matrix and not args.rounds and args.round is None:
            logger.error("--existing-matrix, --round, or --rounds required for rerun mode")
            sys.exit(1)
        system.run_rerun_attacks(
            args.rerun_attacks,
            existing_matrix=args.existing_matrix,
            round_num=args.round,
            rounds=rounds_list,
            matrix_pattern=matrix_pattern
        )
    elif args.rerun_models:
        # Either --existing-matrix (single file) or --rounds (multiple files via pattern)
        if not args.existing_matrix and not args.rounds and args.round is None:
            logger.error("--existing-matrix, --round, or --rounds required for rerun mode")
            sys.exit(1)
        system.run_rerun_models(
            args.rerun_models,
            existing_matrix=args.existing_matrix,
            round_num=args.round,
            rounds=rounds_list,
            matrix_pattern=matrix_pattern
        )
    elif args.update_from_fixed:
        if not args.existing_matrix:
            logger.error("--existing-matrix required for --update-from-fixed mode")
            sys.exit(1)
        system.run_update_from_fixed(args.existing_matrix)
    else:
        # Default: full pipeline
        system.run_full_pipeline()


if __name__ == '__main__':
    main()
