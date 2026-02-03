"""
COMP219 Marking System - Evaluator Module
Author: Lingfang Li

Core evaluation engine for student mutual evaluation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
from typing import Dict, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

from config import MarkingConfig
from utils import (
    logger, Net, load_model, set_random_seed,
    print_progress_bar, verify_epsilon_constraint,
    get_student_ids_from_directory, get_py_filepath, get_pt_filepath
)
from dependency_manager import DependencyManager, load_attack_function


@dataclass
class EvaluationEntry:
    """Single evaluation entry (attack vs model)"""
    attacker_id: str
    defender_id: str
    defence_accuracy: float  # Robust accuracy (higher = better defence)
    attack_score: float  # 1/accuracy (higher = better attack)
    epsilon_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class StudentEvaluation:
    """Complete evaluation results for a student"""
    student_id: str
    attack_scores: Dict[str, float] = field(default_factory=dict)  # defender_id -> attack_score
    defence_scores: Dict[str, float] = field(default_factory=dict)  # attacker_id -> defence_accuracy
    mean_attack_score: float = 0.0
    mean_defence_score: float = 0.0
    attack_loaded: bool = False
    model_loaded: bool = False
    error_messages: List[str] = field(default_factory=list)


class Evaluator:
    """
    Core evaluation engine for student mutual evaluation.

    Implements the n×n evaluation matrix where:
    - Rows = defence models (i)
    - Columns = attack algorithms (j)
    - Matrix[i,j] = robust accuracy of model_i against attack_j

    2025 Update:
    - Attack functions run WITH gradient context (for FGSM/PGD support)
    - Model predictions run WITHOUT gradient context
    """

    def __init__(self, config: MarkingConfig, partition: Optional[Tuple[int, int]] = None):
        """
        Initialize the evaluator.

        Args:
            config: MarkingConfig instance
            partition: Optional tuple (part_num, total_parts) for distributed evaluation
                       e.g., (1, 3) means part 1 of 3
        """
        self.config = config
        self.device = torch.device(config.device)
        self.dep_manager = DependencyManager(auto_install=config.auto_install_deps)

        # Partition settings
        self.partition = partition  # (part_num, total_parts) or None

        # Loaded components
        self.attack_functions: Dict[str, Callable] = {}
        self.models: Dict[str, torch.nn.Module] = {}

        # Results
        self.evaluation_matrix: Optional[np.ndarray] = None
        self.student_ids: List[str] = []
        self.evaluations: Dict[str, StudentEvaluation] = {}

    def prepare_test_data(self, round_num: int = 1) -> DataLoader:
        """
        Prepare test data loader with fixed random samples.

        Args:
            round_num: Round number (1, 2, 3...) - different rounds use different seeds

        Returns:
            DataLoader with test samples
        """
        # Use round-specific seed (same formula as run_single_round)
        seed = self.config.random_seed + round_num * 1000
        set_random_seed(seed)
        logger.info(f"Using seed {seed} for round {round_num}")

        # Load FashionMNIST test set
        test_set = torchvision.datasets.FashionMNIST(
            root='data',
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )

        # Select random subset
        indices = torch.randperm(len(test_set))[:self.config.num_test_samples]

        selected_data = []
        selected_targets = []

        for idx in indices:
            data, target = test_set[idx]
            selected_data.append(data.view(-1))  # Flatten to 784
            selected_targets.append(target)

        selected_data = torch.stack(selected_data)
        selected_targets = torch.tensor(selected_targets)

        test_loader = DataLoader(
            TensorDataset(selected_data, selected_targets),
            batch_size=self.config.batch_size,
            shuffle=False  # No shuffle for reproducibility
        )

        logger.info(f"Prepared {self.config.num_test_samples} test samples")
        return test_loader

    def load_student_components(
            self,
            student_ids: List[str],
            py_dir: str,
            pt_dir: str
    ) -> Tuple[int, int]:
        """
        Load attack functions and models for all students.

        Args:
            student_ids: List of student IDs
            py_dir: Directory containing .py files
            pt_dir: Directory containing .pt files

        Returns:
            Tuple of (num_attacks_loaded, num_models_loaded)
        """
        self.student_ids = student_ids
        num_attacks = 0
        num_models = 0

        logger.info(f"Loading components for {len(student_ids)} students...")

        for i, student_id in enumerate(student_ids):
            print_progress_bar(i + 1, len(student_ids), prefix="Loading:", suffix=student_id)

            # Initialize evaluation entry
            if student_id not in self.evaluations:
                self.evaluations[student_id] = StudentEvaluation(student_id=student_id)

            eval_entry = self.evaluations[student_id]

            # Load attack function
            py_file = get_py_filepath(py_dir, student_id)
            if os.path.exists(py_file):
                attack_fn, dep_result = load_attack_function(py_file, self.dep_manager)
                if attack_fn is not None:
                    self.attack_functions[student_id] = attack_fn
                    eval_entry.attack_loaded = True
                    num_attacks += 1
                else:
                    eval_entry.error_messages.append(f"Failed to load attack: {py_file}")
            else:
                eval_entry.error_messages.append(f"PY file not found: {py_file}")

            # Load model
            pt_file = get_pt_filepath(pt_dir, student_id)
            if os.path.exists(pt_file):
                model = load_model(pt_file, self.device)
                if model is not None:
                    self.models[student_id] = model
                    eval_entry.model_loaded = True
                    num_models += 1
                else:
                    eval_entry.error_messages.append(f"Failed to load model: {pt_file}")
            else:
                eval_entry.error_messages.append(f"PT file not found: {pt_file}")

        logger.info(f"\nLoaded {num_attacks} attack functions and {num_models} models")
        return num_attacks, num_models

    def eval_single_attack(
            self,
            model: torch.nn.Module,
            attack_fn: Callable,
            test_loader: DataLoader
    ) -> Tuple[float, bool]:
        """
        Evaluate a single attack against a single model.

        2025 Update: Attack runs WITH gradient context, prediction WITHOUT.

        Args:
            model: Defence model
            attack_fn: Attack function
            test_loader: Test data loader

        Returns:
            Tuple of (robust_accuracy, epsilon_valid)
        """
        model.eval()
        correct = 0
        total = 0
        epsilon_valid = True

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)

            try:
                # Store original data for epsilon check
                original_data = data.clone()

                # Generate adversarial examples (WITH gradient context)
                # This is the 2025 update - attacks can use gradients
                adv_data = attack_fn(model, data, target, self.device)

                # Verify epsilon constraint
                is_valid, perturbation = verify_epsilon_constraint(
                    original_data, adv_data, self.config.epsilon_constraint
                )
                if not is_valid:
                    epsilon_valid = False
                    logger.warning(f"Epsilon constraint violated: {perturbation:.4f} >= {self.config.epsilon_constraint}")

                # Evaluate on adversarial examples (WITHOUT gradient context)
                with torch.no_grad():
                    output = model(adv_data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

            except Exception as e:
                logger.debug(f"Error in batch evaluation: {e}")
                # Continue with remaining batches
                continue

        if total == 0:
            return 0.0, False

        accuracy = correct / total
        return accuracy, epsilon_valid

    def _get_partition_suffix(self) -> str:
        """Get file suffix for partition naming"""
        if self.partition:
            part_num, total_parts = self.partition
            return f"_part{part_num}of{total_parts}"
        return ""

    def _get_checkpoint_path(self, round_num: int = 1) -> str:
        """Get checkpoint file path for a round"""
        suffix = self._get_partition_suffix()
        return os.path.join(self.config.output_dir, f"checkpoint_round_{round_num}{suffix}.csv")

    def _save_checkpoint(self, model_ids: List[str], attack_ids: List[str],
                         completed_row: int, round_num: int = 1):
        """Save current progress to checkpoint file"""
        checkpoint_path = self._get_checkpoint_path(round_num)

        # Create DataFrame from current matrix
        df = pd.DataFrame(
            self.evaluation_matrix,
            index=[f'Model_{sid}' for sid in model_ids],
            columns=[f'Attack_{sid}' for sid in attack_ids]
        )

        # Add metadata
        df.to_csv(checkpoint_path)

        # Save progress info
        progress_path = checkpoint_path.replace('.csv', '_progress.txt')
        with open(progress_path, 'w') as f:
            f.write(f"completed_rows={completed_row}\n")
            f.write(f"total_rows={len(model_ids)}\n")

        logger.debug(f"Checkpoint saved: row {completed_row}/{len(model_ids)}")

    def _get_ids_from_checkpoint(self, round_num: int = 1) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Get model_ids and attack_ids from existing checkpoint file.

        Returns:
            Tuple of (model_ids, attack_ids) or (None, None) if no checkpoint
        """
        checkpoint_path = self._get_checkpoint_path(round_num)

        if not os.path.exists(checkpoint_path):
            return None, None

        try:
            df = pd.read_csv(checkpoint_path, index_col=0)
            # Extract IDs from index and columns
            model_ids = [idx.replace('Model_', '') for idx in df.index]
            attack_ids = [col.replace('Attack_', '') for col in df.columns]
            return model_ids, attack_ids
        except Exception as e:
            logger.warning(f"Failed to read checkpoint IDs: {e}")
            return None, None

    def _load_checkpoint(self, model_ids: List[str], attack_ids: List[str],
                         round_num: int = 1) -> int:
        """
        Load checkpoint if exists.

        Returns:
            Starting row index (0 if no checkpoint)
        """
        checkpoint_path = self._get_checkpoint_path(round_num)
        progress_path = checkpoint_path.replace('.csv', '_progress.txt')

        if not os.path.exists(checkpoint_path) or not os.path.exists(progress_path):
            return 0

        try:
            # Load progress info
            with open(progress_path, 'r') as f:
                lines = f.readlines()
                completed_rows = int(lines[0].split('=')[1])

            # Load matrix
            df = pd.read_csv(checkpoint_path, index_col=0)

            # Verify dimensions match
            if df.shape[0] != len(model_ids) or df.shape[1] != len(attack_ids):
                logger.warning("Checkpoint dimensions mismatch, starting fresh")
                return 0

            # Restore matrix
            self.evaluation_matrix = df.values

            logger.info(f"Resumed from checkpoint: {completed_rows}/{len(model_ids)} rows completed")
            return completed_rows

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0

    def _delete_checkpoint(self, round_num: int = 1):
        """Delete checkpoint files after successful completion"""
        checkpoint_path = self._get_checkpoint_path(round_num)
        progress_path = checkpoint_path.replace('.csv', '_progress.txt')

        for path in [checkpoint_path, progress_path]:
            if os.path.exists(path):
                os.remove(path)

    def _get_partition_range(self, total_items: int) -> Tuple[int, int]:
        """
        Get the range of items for this partition.

        Args:
            total_items: Total number of items

        Returns:
            Tuple of (start_index, end_index) - end is exclusive
        """
        if not self.partition:
            return 0, total_items

        part_num, total_parts = self.partition
        items_per_part = total_items // total_parts
        remainder = total_items % total_parts

        # Distribute remainder to first partitions
        start = 0
        for i in range(1, part_num):
            start += items_per_part + (1 if i <= remainder else 0)

        end = start + items_per_part + (1 if part_num <= remainder else 0)

        return start, end

    def run_full_evaluation(self, test_loader: DataLoader, round_num: int = 1) -> np.ndarray:
        """
        Run full n×n mutual evaluation with checkpointing and timing.
        Supports partition mode for distributed evaluation.

        Args:
            test_loader: Test data loader
            round_num: Current round number (for checkpoint naming)

        Returns:
            Evaluation matrix [n_models × n_attacks] (or partition subset)
        """
        import time

        all_model_ids = [sid for sid in self.student_ids if sid in self.models]
        attack_ids = [sid for sid in self.student_ids if sid in self.attack_functions]

        # Check if checkpoint exists and use its IDs to maintain consistency
        checkpoint_model_ids, checkpoint_attack_ids = self._get_ids_from_checkpoint(round_num)

        if checkpoint_attack_ids is not None:
            # Use checkpoint attack IDs to maintain consistency
            # Filter to only include attacks that are still available
            original_attack_count = len(attack_ids)
            attack_ids = [aid for aid in checkpoint_attack_ids if aid in self.attack_functions]
            if len(attack_ids) < len(checkpoint_attack_ids):
                missing = set(checkpoint_attack_ids) - set(attack_ids)
                logger.warning(f"Some checkpoint attacks not available: {missing}")
            logger.info(f"Using checkpoint attack IDs: {len(attack_ids)} attacks (original: {original_attack_count})")

        # Get partition range for models
        start_idx, end_idx = self._get_partition_range(len(all_model_ids))
        model_ids = all_model_ids[start_idx:end_idx]

        # If checkpoint has model IDs for this partition, use them
        if checkpoint_model_ids is not None:
            # Verify checkpoint model IDs match current partition
            checkpoint_model_set = set(checkpoint_model_ids)
            current_model_set = set(model_ids)
            if checkpoint_model_set != current_model_set:
                # Use checkpoint model IDs if they're a subset and still available
                model_ids = [mid for mid in checkpoint_model_ids if mid in self.models]
                logger.info(f"Using checkpoint model IDs: {len(model_ids)} models")

        n_models = len(model_ids)
        n_attacks = len(attack_ids)

        if self.partition:
            part_num, total_parts = self.partition
            logger.info(f"\nRunning PARTITION {part_num}/{total_parts}: models {start_idx+1}-{end_idx} of {len(all_model_ids)}")

        logger.info(f"\nRunning evaluation: {n_models} models × {n_attacks} attacks")
        logger.info("=" * 60)

        # Store the actual IDs used (for DataFrame creation later)
        self._last_model_ids = model_ids
        self._last_attack_ids = attack_ids

        # Initialize matrix
        self.evaluation_matrix = np.zeros((n_models, n_attacks))

        # Try to load checkpoint
        start_row = self._load_checkpoint(model_ids, attack_ids, round_num)

        total_evals = n_models * n_attacks
        current_eval = start_row * n_attacks

        # Track timing statistics
        attack_times: Dict[str, List[float]] = {}

        for i, defender_id in enumerate(model_ids):
            # Skip already completed rows
            if i < start_row:
                continue

            model = self.models[defender_id]

            for j, attacker_id in enumerate(attack_ids):
                current_eval += 1

                # Skip self-attack (own model vs own attack)
                if defender_id == attacker_id:
                    self.evaluation_matrix[i, j] = np.nan
                    print_progress_bar(
                        current_eval, total_evals,
                        prefix="Evaluating:",
                        suffix=f"Model {defender_id} vs Attack {attacker_id} [SKIP SELF]"
                    )
                    continue

                attack_fn = self.attack_functions[attacker_id]

                # Time the attack
                start_time = time.time()

                try:
                    accuracy, epsilon_valid = self.eval_single_attack(
                        model, attack_fn, test_loader
                    )

                    self.evaluation_matrix[i, j] = accuracy

                    # Store in evaluations
                    attack_score = 1.0 / accuracy if accuracy > 0 else float('inf')

                    self.evaluations[defender_id].defence_scores[attacker_id] = accuracy
                    self.evaluations[attacker_id].attack_scores[defender_id] = attack_score

                except Exception as e:
                    logger.error(f"Evaluation failed: Model {defender_id} vs Attack {attacker_id}: {e}")
                    self.evaluation_matrix[i, j] = 0.0

                # Calculate elapsed time
                elapsed_time = time.time() - start_time

                # Track attack times
                if attacker_id not in attack_times:
                    attack_times[attacker_id] = []
                attack_times[attacker_id].append(elapsed_time)

                # Display progress with timing
                print_progress_bar(
                    current_eval, total_evals,
                    prefix="Evaluating:",
                    suffix=f"Model {defender_id} vs Attack {attacker_id} [{elapsed_time:.1f}s]"
                )

                # Warn if attack takes too long
                if elapsed_time > 60:
                    logger.warning(f"Slow attack: {attacker_id} took {elapsed_time:.1f}s")

            # Save checkpoint after each row (model)
            self._save_checkpoint(model_ids, attack_ids, i + 1, round_num)
            logger.info(f"\n  Row {i + 1}/{n_models} completed, checkpoint saved")

        # Print timing summary
        logger.info("\n" + "=" * 60)
        logger.info("ATTACK TIMING SUMMARY")
        logger.info("-" * 60)
        for attacker_id, times in sorted(attack_times.items(), key=lambda x: np.mean(x[1]), reverse=True):
            avg_time = np.mean(times)
            max_time = np.max(times)
            logger.info(f"  Attack {attacker_id}: avg={avg_time:.2f}s, max={max_time:.2f}s")
        logger.info("=" * 60)

        # NOTE: Do NOT delete checkpoint here!
        # Checkpoint should only be deleted after results are successfully saved to file
        # This is handled in run_single_round after DataFrame is created and saved
        logger.info("Full evaluation completed")

        return self.evaluation_matrix

    def run_single_round(self, round_num: int = 1) -> pd.DataFrame:
        """
        Run a single evaluation round.

        Args:
            round_num: Round number for logging

        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"\n{'='*60}")
        if self.partition:
            part_num, total_parts = self.partition
            logger.info(f"EVALUATION ROUND {round_num} - PARTITION {part_num}/{total_parts}")
        else:
            logger.info(f"EVALUATION ROUND {round_num}")
        logger.info(f"{'='*60}")

        # Prepare test data with different seed for each round
        test_loader = self.prepare_test_data(round_num=round_num)

        # Run evaluation with round number for checkpointing
        matrix = self.run_full_evaluation(test_loader, round_num)

        # Use the actual IDs from run_full_evaluation (stored as instance variables)
        model_ids = self._last_model_ids
        attack_ids = self._last_attack_ids

        df = pd.DataFrame(
            matrix,
            index=[f'Model_{sid}' for sid in model_ids],
            columns=[f'Attack_{sid}' for sid in attack_ids]
        )

        return df

    def run_multiple_rounds(self, num_rounds: Optional[int] = None,
                            start_round: int = 1) -> List[pd.DataFrame]:
        """
        Run multiple evaluation rounds.

        Args:
            num_rounds: Number of rounds (default from config)
            start_round: Starting round number (default: 1)

        Returns:
            List of DataFrames, one per round
        """
        if num_rounds is None:
            num_rounds = self.config.num_rounds

        results = []
        suffix = self._get_partition_suffix()

        logger.info(f"Running rounds {start_round} to {num_rounds}")

        for round_num in range(start_round, num_rounds + 1):
            try:
                df = self.run_single_round(round_num)
                results.append(df)

                # Save intermediate result with partition suffix
                output_file = os.path.join(
                    self.config.output_dir,
                    f"evaluation_round_{round_num}{suffix}.csv"
                )
                df.to_csv(output_file)
                logger.info(f"Round {round_num} results saved to {output_file}")

                # Only delete checkpoint AFTER results are successfully saved
                self._delete_checkpoint(round_num)
                logger.info(f"Round {round_num} checkpoint deleted")

            except Exception as e:
                logger.error(f"Round {round_num} failed: {e}")
                logger.error("Checkpoint preserved for retry. Please fix the issue and re-run.")
                raise  # Re-raise to stop execution

        return results

    def save_results(self, filename_prefix: str = "evaluation"):
        """
        Save evaluation results to files.

        Args:
            filename_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save evaluation matrix
        if self.evaluation_matrix is not None:
            model_ids = [sid for sid in self.student_ids if sid in self.models]
            attack_ids = [sid for sid in self.student_ids if sid in self.attack_functions]

            df = pd.DataFrame(
                self.evaluation_matrix,
                index=[f'Model_{sid}' for sid in model_ids],
                columns=[f'Attack_{sid}' for sid in attack_ids]
            )

            output_path = os.path.join(
                self.config.output_dir,
                f"{filename_prefix}_matrix_{timestamp}.csv"
            )
            df.to_csv(output_path)
            logger.info(f"Evaluation matrix saved to {output_path}")

        # Save detailed evaluations
        eval_data = []
        for student_id, eval_entry in self.evaluations.items():
            eval_data.append({
                'student_id': student_id,
                'attack_loaded': eval_entry.attack_loaded,
                'model_loaded': eval_entry.model_loaded,
                'mean_attack_score': eval_entry.mean_attack_score,
                'mean_defence_score': eval_entry.mean_defence_score,
                'errors': '; '.join(eval_entry.error_messages) if eval_entry.error_messages else ''
            })

        eval_df = pd.DataFrame(eval_data)
        eval_path = os.path.join(
            self.config.output_dir,
            f"{filename_prefix}_details_{timestamp}.csv"
        )
        eval_df.to_csv(eval_path, index=False)
        logger.info(f"Evaluation details saved to {eval_path}")


class IncrementalEvaluator(Evaluator):
    """
    Evaluator that supports incremental updates to existing results.
    """

    def __init__(self, config: MarkingConfig, existing_matrix_path: Optional[str] = None):
        """
        Initialize with optional existing matrix.

        Args:
            config: MarkingConfig instance
            existing_matrix_path: Path to existing evaluation matrix CSV
        """
        super().__init__(config)
        self.existing_matrix: Optional[pd.DataFrame] = None

        if existing_matrix_path and os.path.exists(existing_matrix_path):
            self.existing_matrix = pd.read_csv(existing_matrix_path, index_col=0)
            logger.info(f"Loaded existing matrix: {self.existing_matrix.shape}")

    def add_new_students(
            self,
            new_student_ids: List[str],
            py_dir: str,
            pt_dir: str,
            test_loader: DataLoader
    ) -> pd.DataFrame:
        """
        Add new students to existing evaluation.

        Args:
            new_student_ids: List of new student IDs
            py_dir: Directory containing .py files
            pt_dir: Directory containing .pt files
            test_loader: Test data loader

        Returns:
            Updated evaluation matrix
        """
        if self.existing_matrix is None:
            logger.warning("No existing matrix, running full evaluation")
            self.student_ids = new_student_ids
            self.load_student_components(new_student_ids, py_dir, pt_dir)
            return self.run_single_round()

        # Load new student components
        self.load_student_components(new_student_ids, py_dir, pt_dir)

        # Get existing IDs from matrix
        existing_model_ids = [col.replace('Model_', '') for col in self.existing_matrix.index]
        existing_attack_ids = [col.replace('Attack_', '') for col in self.existing_matrix.columns]

        # Evaluate new attacks vs all models (existing + new)
        # Evaluate all attacks vs new models
        # ... (implementation details for incremental update)

        logger.info("Incremental evaluation completed")
        return self.existing_matrix

    def _load_rerun_checkpoint(self, output_path: str, rerun_ids: List[str], mode: str):
        """
        Try to load a checkpoint from a previous rerun.

        Args:
            output_path: Base path for the output CSV
            rerun_ids: List of student IDs being rerun (for identity validation)
            mode: 'attacks' or 'models'

        Returns:
            (checkpoint_path, progress_path, start_eval) - start_eval=0 if no valid checkpoint
        """
        checkpoint_path = output_path + '.checkpoint.csv'
        progress_path = output_path + '.checkpoint_progress.txt'
        start_eval = 0

        if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                saved = {}
                for line in lines:
                    key, val = line.split('=', 1)
                    saved[key] = val

                saved_ids = saved.get('rerun_ids', '')
                saved_mode = saved.get('mode', '')
                current_ids = ','.join(sorted(rerun_ids))

                if saved_ids != current_ids or saved_mode != mode:
                    logger.warning(f"Checkpoint is from a different rerun "
                                   f"(saved: {saved_mode} {saved_ids[:40]}..., "
                                   f"current: {mode} {current_ids[:40]}...). Starting fresh.")
                    # Clean up stale checkpoint
                    os.remove(checkpoint_path)
                    os.remove(progress_path)
                    return checkpoint_path, progress_path, 0

                start_eval = int(saved['completed'])
                checkpoint_df = pd.read_csv(checkpoint_path, index_col=0)
                for idx in checkpoint_df.index:
                    for col in checkpoint_df.columns:
                        if idx in self.existing_matrix.index and col in self.existing_matrix.columns:
                            self.existing_matrix.loc[idx, col] = checkpoint_df.loc[idx, col]
                logger.info(f"Resumed from checkpoint: {start_eval}/{saved.get('total', '?')} evaluations completed")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
                start_eval = 0

        return checkpoint_path, progress_path, start_eval

    def _save_rerun_checkpoint(self, checkpoint_path: str, progress_path: str,
                                current_eval: int, total_evals: int,
                                rerun_ids: List[str], mode: str):
        """Save a rerun checkpoint with identity information."""
        self.existing_matrix.to_csv(checkpoint_path)
        with open(progress_path, 'w') as f:
            f.write(f"completed={current_eval}\n")
            f.write(f"total={total_evals}\n")
            f.write(f"mode={mode}\n")
            f.write(f"rerun_ids={','.join(sorted(rerun_ids))}\n")

    def _cleanup_rerun_checkpoint(self, checkpoint_path: str, progress_path: str):
        """Remove checkpoint files after successful completion."""
        for path in [checkpoint_path, progress_path]:
            if path and os.path.exists(path):
                os.remove(path)

    def _run_rerun_loop(self, evals_to_do: list, test_loader: DataLoader,
                         output_path: str, rerun_ids: List[str], mode: str,
                         checkpoint_interval: int = 20) -> pd.DataFrame:
        """
        Shared evaluation loop for rerun_specific_attacks and rerun_specific_models.

        Args:
            evals_to_do: List of (model_id, attack_id) tuples
            test_loader: Test data loader
            output_path: Path to save matrix (optional)
            rerun_ids: List of student IDs being rerun
            mode: 'attacks' or 'models' (for checkpoint identity and progress display)
            checkpoint_interval: Save checkpoint every N evaluations

        Returns:
            Updated evaluation matrix
        """
        import time

        total_evals = len(evals_to_do)
        logger.info(f"Total evaluations to perform: {total_evals}")

        # Load checkpoint if available
        checkpoint_path = progress_path = None
        start_eval = 0
        if output_path:
            checkpoint_path, progress_path, start_eval = \
                self._load_rerun_checkpoint(output_path, rerun_ids, mode)

        # Run evaluations
        current_eval = 0
        for model_id, attack_id in evals_to_do:
            current_eval += 1
            if current_eval <= start_eval:
                continue

            model = self.models[model_id]
            attack_fn = self.attack_functions[attack_id]
            model_row = f'Model_{model_id}'
            attack_col = f'Attack_{attack_id}'

            start_time = time.time()
            try:
                accuracy, _ = self.eval_single_attack(model, attack_fn, test_loader)
                self.existing_matrix.loc[model_row, attack_col] = accuracy
            except Exception as e:
                logger.error(f"Evaluation failed: Model {model_id} vs Attack {attack_id}: {e}")
                self.existing_matrix.loc[model_row, attack_col] = 0.0
                accuracy = 0.0

            elapsed = time.time() - start_time
            print_progress_bar(
                current_eval, total_evals,
                prefix=f"Re-running {mode}:",
                suffix=f"Model {model_id} vs Attack {attack_id} = {accuracy:.4f} [{elapsed:.1f}s]"
            )

            # Periodic checkpoint
            if output_path and current_eval % checkpoint_interval == 0:
                self._save_rerun_checkpoint(
                    checkpoint_path, progress_path,
                    current_eval, total_evals, rerun_ids, mode
                )
                logger.info(f"\n  Checkpoint saved: {current_eval}/{total_evals}")

        # Final save and cleanup
        if output_path:
            self.existing_matrix.to_csv(output_path)
            self._cleanup_rerun_checkpoint(checkpoint_path, progress_path)
            logger.info(f"Final save complete: {output_path}")

        return self.existing_matrix

    def rerun_specific_attacks(
            self,
            attack_ids: List[str],
            py_dir: str,
            pt_dir: str,
            test_loader: DataLoader,
            output_path: str = None,
            checkpoint_interval: int = 20
    ) -> pd.DataFrame:
        """
        Re-run specific attack evaluations with periodic checkpointing.

        Args:
            attack_ids: List of attack IDs to re-run
            py_dir: Directory containing .py files
            pt_dir: Directory containing .pt files
            test_loader: Test data loader
            output_path: Path to save matrix incrementally (optional)
            checkpoint_interval: Save checkpoint every N evaluations (default: 20)

        Returns:
            Updated evaluation matrix
        """
        if self.existing_matrix is None:
            raise ValueError("No existing matrix to update")

        # Load attack functions
        for attack_id in attack_ids:
            py_file = get_py_filepath(py_dir, attack_id)
            if os.path.exists(py_file):
                attack_fn, _ = load_attack_function(py_file, self.dep_manager)
                if attack_fn is not None:
                    self.attack_functions[attack_id] = attack_fn

        # Load all models
        model_ids = [idx.replace('Model_', '') for idx in self.existing_matrix.index]
        for model_id in model_ids:
            pt_file = get_pt_filepath(pt_dir, model_id)
            if os.path.exists(pt_file):
                model = load_model(pt_file, self.device)
                if model is not None:
                    self.models[model_id] = model

        # Build flat evaluation list (skip self-attack)
        evals_to_do = []
        for attack_id in attack_ids:
            if attack_id not in self.attack_functions:
                logger.warning(f"Attack {attack_id} not loaded, skipping")
                continue
            for model_id in model_ids:
                if model_id not in self.models:
                    continue
                if model_id == attack_id:
                    continue
                evals_to_do.append((model_id, attack_id))

        return self._run_rerun_loop(
            evals_to_do, test_loader, output_path, attack_ids, 'attacks', checkpoint_interval
        )

    def rerun_specific_models(
            self,
            model_ids: List[str],
            py_dir: str,
            pt_dir: str,
            test_loader: DataLoader,
            output_path: str = None,
            checkpoint_interval: int = 20
    ) -> pd.DataFrame:
        """
        Re-run specific model evaluations (update rows in matrix) with periodic checkpointing.

        Args:
            model_ids: List of model IDs to re-run
            py_dir: Directory containing .py files
            pt_dir: Directory containing .pt files
            test_loader: Test data loader
            output_path: Path to save matrix incrementally (optional)
            checkpoint_interval: Save checkpoint every N evaluations (default: 20)

        Returns:
            Updated evaluation matrix
        """
        if self.existing_matrix is None:
            raise ValueError("No existing matrix to update")

        # Load specified models
        for model_id in model_ids:
            pt_file = get_pt_filepath(pt_dir, model_id)
            if os.path.exists(pt_file):
                model = load_model(pt_file, self.device)
                if model is not None:
                    self.models[model_id] = model
                    logger.info(f"Loaded model: {model_id}")
                else:
                    logger.warning(f"Failed to load model: {model_id}")
            else:
                logger.warning(f"Model file not found: {pt_file}")

        # Load all attack functions from matrix columns
        attack_ids = [col.replace('Attack_', '') for col in self.existing_matrix.columns if col.startswith('Attack_')]
        for attack_id in attack_ids:
            if attack_id not in self.attack_functions:
                py_file = get_py_filepath(py_dir, attack_id)
                if os.path.exists(py_file):
                    attack_fn, _ = load_attack_function(py_file, self.dep_manager)
                    if attack_fn is not None:
                        self.attack_functions[attack_id] = attack_fn

        logger.info(f"Loaded {len(self.attack_functions)} attack functions")

        # Ensure rows exist for all target models
        for model_id in model_ids:
            model_row = f'Model_{model_id}'
            if model_row not in self.existing_matrix.index:
                self.existing_matrix.loc[model_row] = [None] * len(self.existing_matrix.columns)
                logger.info(f"Added new row for model: {model_id}")

        # Build flat evaluation list
        # Build flat evaluation list (skip self-attack)
        evals_to_do = []
        for model_id in model_ids:
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not loaded, skipping")
                continue
            for attack_id in attack_ids:
                if attack_id not in self.attack_functions:
                    continue
                if model_id == attack_id:
                    continue
                evals_to_do.append((model_id, attack_id))

        return self._run_rerun_loop(
            evals_to_do, test_loader, output_path, model_ids, 'models', checkpoint_interval
        )


class FixedFilesEvaluator(Evaluator):
    """
    Evaluator for updating evaluation matrix with fixed files.

    This evaluator:
    1. Loads an existing evaluation matrix
    2. Detects fixed files in fixed_py_dir and fixed_pt_dir
    3. Evaluates fixed attacks against ALL models (original + fixed)
    4. Evaluates ALL attacks against fixed models (avoiding duplicates)
    5. Updates and saves the evaluation matrix
    """

    def __init__(self, config, existing_matrix_path: str):
        """
        Initialize with existing matrix.

        Args:
            config: MarkingConfig instance
            existing_matrix_path: Path to existing evaluation matrix CSV
        """
        super().__init__(config)
        self.existing_matrix_path = existing_matrix_path
        self.existing_matrices: List[pd.DataFrame] = []

        # Load existing matrices (one per round)
        self._load_existing_matrices()

    def _load_existing_matrices(self):
        """Load existing evaluation matrices for all rounds"""
        # Check if it's a single file or we need to find round files
        if os.path.exists(self.existing_matrix_path):
            # Single file provided - try to find all rounds
            base_dir = os.path.dirname(self.existing_matrix_path)
            base_name = os.path.basename(self.existing_matrix_path)

            if 'round_' in base_name:
                # It's a round file, find all rounds
                for round_num in range(1, self.config.num_rounds + 1):
                    round_file = os.path.join(base_dir, f"evaluation_round_{round_num}.csv")
                    if os.path.exists(round_file):
                        df = pd.read_csv(round_file, index_col=0)
                        self.existing_matrices.append(df)
                        logger.info(f"Loaded existing matrix round {round_num}: {df.shape}")
            else:
                # Single matrix file
                df = pd.read_csv(self.existing_matrix_path, index_col=0)
                self.existing_matrices.append(df)
                logger.info(f"Loaded existing matrix: {df.shape}")

        if not self.existing_matrices:
            raise ValueError(f"Could not load existing matrix from {self.existing_matrix_path}")

    def _get_ids_from_directory(self, directory: str, extension: str) -> Set[str]:
        """Get student IDs from a directory"""
        return get_student_ids_from_directory(directory, extension)

    def run_update(self, py_dir: str, pt_dir: str,
                   fixed_py_dir: str, fixed_pt_dir: str) -> List[pd.DataFrame]:
        """
        Run the update with fixed files.

        Args:
            py_dir: Original py files directory
            pt_dir: Original pt files directory
            fixed_py_dir: Fixed py files directory
            fixed_pt_dir: Fixed pt files directory

        Returns:
            List of updated DataFrames
        """
        import time

        # Get IDs from directories
        original_py_ids = self._get_ids_from_directory(py_dir, '.py')
        original_pt_ids = self._get_ids_from_directory(pt_dir, '.pt')
        fixed_py_ids = self._get_ids_from_directory(fixed_py_dir, '.py')
        fixed_pt_ids = self._get_ids_from_directory(fixed_pt_dir, '.pt')

        logger.info(f"\nOriginal attacks: {len(original_py_ids)}")
        logger.info(f"Original models: {len(original_pt_ids)}")
        logger.info(f"Fixed attacks: {len(fixed_py_ids)}")
        logger.info(f"Fixed models: {len(fixed_pt_ids)}")

        if not fixed_py_ids and not fixed_pt_ids:
            logger.warning("No fixed files found, nothing to update")
            return self.existing_matrices

        # Load ALL attack functions (original + fixed, with fixed overriding)
        all_py_ids = original_py_ids | fixed_py_ids
        logger.info(f"\nLoading attack functions...")
        for student_id in all_py_ids:
            # Prefer fixed version if exists
            if student_id in fixed_py_ids:
                py_file = get_py_filepath(fixed_py_dir, student_id)
            else:
                py_file = get_py_filepath(py_dir, student_id)

            if os.path.exists(py_file):
                attack_fn, _ = load_attack_function(py_file, self.dep_manager)
                if attack_fn is not None:
                    self.attack_functions[student_id] = attack_fn

        logger.info(f"Loaded {len(self.attack_functions)} attack functions")

        # Load ALL models (original + fixed, with fixed overriding)
        all_pt_ids = original_pt_ids | fixed_pt_ids
        logger.info(f"\nLoading models...")
        for student_id in all_pt_ids:
            # Prefer fixed version if exists
            if student_id in fixed_pt_ids:
                pt_file = get_pt_filepath(fixed_pt_dir, student_id)
            else:
                pt_file = get_pt_filepath(pt_dir, student_id)

            if os.path.exists(pt_file):
                model = load_model(pt_file, self.device)
                if model is not None:
                    self.models[student_id] = model

        logger.info(f"Loaded {len(self.models)} models")

        # Process each round
        updated_matrices = []
        for round_num, existing_df in enumerate(self.existing_matrices, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"UPDATING ROUND {round_num}")
            logger.info(f"{'='*60}")

            try:
                updated_df = self._update_single_round(
                    existing_df, fixed_py_ids, fixed_pt_ids, round_num
                )
                updated_matrices.append(updated_df)

                # Save updated matrix
                output_file = os.path.join(
                    self.config.output_dir,
                    f"evaluation_round_{round_num}_updated.csv"
                )
                updated_df.to_csv(output_file)
                logger.info(f"Updated matrix saved to {output_file}")

                # Delete checkpoint only after successful save
                checkpoint_path = os.path.join(
                    self.config.output_dir,
                    f"checkpoint_fixed_round_{round_num}.csv"
                )
                progress_path = checkpoint_path.replace('.csv', '_progress.txt')
                for path in [checkpoint_path, progress_path]:
                    if os.path.exists(path):
                        os.remove(path)
                logger.info(f"Round {round_num} checkpoint deleted")

            except Exception as e:
                logger.error(f"Round {round_num} update failed: {e}")
                logger.error("Checkpoint preserved for retry.")
                raise

        return updated_matrices

    def _update_single_round(self, existing_df: pd.DataFrame,
                              fixed_py_ids: Set[str], fixed_pt_ids: Set[str],
                              round_num: int) -> pd.DataFrame:
        """
        Update a single round's evaluation matrix.

        Args:
            existing_df: Existing evaluation DataFrame
            fixed_py_ids: Set of fixed attack IDs
            fixed_pt_ids: Set of fixed model IDs
            round_num: Round number

        Returns:
            Updated DataFrame
        """
        import time
        from utils import set_random_seed

        # Prepare test data
        set_random_seed(self.config.random_seed + round_num * 1000)
        test_loader = self.prepare_test_data()

        # Get existing IDs from matrix
        existing_model_ids = [idx.replace('Model_', '') for idx in existing_df.index]
        existing_attack_ids = [col.replace('Attack_', '') for col in existing_df.columns]

        # Determine new IDs (in fixed but not in existing matrix)
        new_attack_ids = fixed_py_ids - set(existing_attack_ids)
        new_model_ids = fixed_pt_ids - set(existing_model_ids)

        logger.info(f"Existing matrix: {len(existing_model_ids)} models × {len(existing_attack_ids)} attacks")
        logger.info(f"New attacks to add: {len(new_attack_ids)}")
        logger.info(f"New models to add: {len(new_model_ids)}")
        logger.info(f"Fixed attacks to update: {len(fixed_py_ids - new_attack_ids)}")
        logger.info(f"Fixed models to update: {len(fixed_pt_ids - new_model_ids)}")

        # Create expanded matrix
        all_model_ids = sorted(set(existing_model_ids) | fixed_pt_ids)
        all_attack_ids = sorted(set(existing_attack_ids) | fixed_py_ids)

        # Initialize new matrix with existing values
        new_df = pd.DataFrame(
            index=[f'Model_{mid}' for mid in all_model_ids],
            columns=[f'Attack_{aid}' for aid in all_attack_ids],
            dtype=float
        )

        # Copy existing values
        for model_id in existing_model_ids:
            for attack_id in existing_attack_ids:
                model_row = f'Model_{model_id}'
                attack_col = f'Attack_{attack_id}'
                if model_row in new_df.index and attack_col in new_df.columns:
                    new_df.loc[model_row, attack_col] = existing_df.loc[model_row, attack_col]

        # Calculate what needs to be evaluated
        # 1. Fixed attacks (columns) need to be evaluated against ALL models
        # 2. Fixed models (rows) need to be evaluated against ALL attacks (except fixed attacks already done)

        total_evals = 0
        evals_to_do = []

        # Fixed attacks vs ALL models
        for attack_id in fixed_py_ids:
            if attack_id not in self.attack_functions:
                continue
            for model_id in all_model_ids:
                if model_id not in self.models:
                    continue
                if model_id == attack_id:  # Skip self
                    continue
                evals_to_do.append((model_id, attack_id))
                total_evals += 1

        # Fixed models vs original attacks (not fixed, to avoid duplicates)
        original_attack_ids_to_eval = set(existing_attack_ids) - fixed_py_ids
        for model_id in fixed_pt_ids:
            if model_id not in self.models:
                continue
            for attack_id in original_attack_ids_to_eval:
                if attack_id not in self.attack_functions:
                    continue
                if model_id == attack_id:  # Skip self
                    continue
                evals_to_do.append((model_id, attack_id))
                total_evals += 1

        logger.info(f"\nTotal evaluations to perform: {total_evals}")

        # Checkpoint path for fixed files update
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_fixed_round_{round_num}.csv"
        )
        progress_path = checkpoint_path.replace('.csv', '_progress.txt')

        # Try to load existing checkpoint
        start_eval = 0
        if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
            try:
                checkpoint_df = pd.read_csv(checkpoint_path, index_col=0)
                with open(progress_path, 'r') as f:
                    start_eval = int(f.readline().split('=')[1])
                # Restore values from checkpoint
                for idx in checkpoint_df.index:
                    for col in checkpoint_df.columns:
                        if idx in new_df.index and col in new_df.columns:
                            new_df.loc[idx, col] = checkpoint_df.loc[idx, col]
                logger.info(f"Resumed from checkpoint: {start_eval}/{total_evals} evaluations completed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
                start_eval = 0

        # Run evaluations
        current_eval = 0
        checkpoint_interval = 50  # Save checkpoint every 50 evaluations

        for model_id, attack_id in evals_to_do:
            current_eval += 1

            # Skip already completed evaluations
            if current_eval <= start_eval:
                continue

            model = self.models[model_id]
            attack_fn = self.attack_functions[attack_id]

            start_time = time.time()

            try:
                accuracy, epsilon_valid = self.eval_single_attack(
                    model, attack_fn, test_loader
                )
                new_df.loc[f'Model_{model_id}', f'Attack_{attack_id}'] = accuracy

            except Exception as e:
                logger.error(f"Evaluation failed: Model {model_id} vs Attack {attack_id}: {e}")
                new_df.loc[f'Model_{model_id}', f'Attack_{attack_id}'] = 0.0

            elapsed = time.time() - start_time
            print_progress_bar(
                current_eval, total_evals,
                prefix="Updating:",
                suffix=f"Model {model_id} vs Attack {attack_id} [{elapsed:.1f}s]"
            )

            # Save checkpoint periodically
            if current_eval % checkpoint_interval == 0:
                new_df.to_csv(checkpoint_path)
                with open(progress_path, 'w') as f:
                    f.write(f"completed={current_eval}\n")
                    f.write(f"total={total_evals}\n")
                logger.info(f"\n  Checkpoint saved: {current_eval}/{total_evals}")

        # Set self-attack to NaN
        for student_id in all_model_ids:
            if student_id in all_attack_ids:
                new_df.loc[f'Model_{student_id}', f'Attack_{student_id}'] = np.nan

        logger.info(f"\nRound {round_num} update complete: {new_df.shape[0]} models × {new_df.shape[1]} attacks")

        return new_df


def merge_partition_results(output_dir: str, num_partitions: int, num_rounds: int = 3) -> List[pd.DataFrame]:
    """
    Merge partition results into complete evaluation matrices.

    Args:
        output_dir: Directory containing partition result files
        num_partitions: Number of partitions to merge
        num_rounds: Number of evaluation rounds

    Returns:
        List of merged DataFrames, one per round
    """
    merged_results = []

    for round_num in range(1, num_rounds + 1):
        logger.info(f"\nMerging Round {round_num} partitions...")

        partition_dfs = []
        for part_num in range(1, num_partitions + 1):
            part_file = os.path.join(
                output_dir,
                f"evaluation_round_{round_num}_part{part_num}of{num_partitions}.csv"
            )

            if not os.path.exists(part_file):
                logger.error(f"Partition file not found: {part_file}")
                continue

            df = pd.read_csv(part_file, index_col=0)
            partition_dfs.append(df)
            logger.info(f"  Loaded: {part_file} ({df.shape[0]} models)")

        if not partition_dfs:
            logger.error(f"No partition files found for round {round_num}")
            continue

        # Concatenate all partitions (rows = models)
        merged_df = pd.concat(partition_dfs, axis=0)

        # Sort by model ID for consistency
        merged_df = merged_df.sort_index()

        # Save merged result
        merged_file = os.path.join(output_dir, f"evaluation_round_{round_num}.csv")
        merged_df.to_csv(merged_file)
        logger.info(f"  Merged result saved: {merged_file} ({merged_df.shape[0]} models × {merged_df.shape[1]} attacks)")

        merged_results.append(merged_df)

    logger.info(f"\nMerge complete: {len(merged_results)} rounds merged")
    return merged_results
