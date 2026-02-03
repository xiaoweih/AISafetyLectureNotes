"""
COMP219 Marking System - Grade Calculator Module
Author: Lingfang Li

Calculates final grades from evaluation matrices.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from config import MarkingConfig
from utils import logger, print_summary_statistics


@dataclass
class StudentGrade:
    """Final grade for a student"""
    student_id: str

    # Raw scores
    raw_attack_score: float = 0.0  # Mean of 1/accuracy for attacks
    raw_defence_score: float = 0.0  # Mean accuracy for defence

    # Normalized scores (0-1)
    normalized_attack: float = 0.0
    normalized_defence: float = 0.0

    # Separate assignment scores (each out of 100)
    attack_assignment_score: float = 0.0  # Attack ability score (0-100)
    defence_assignment_score: float = 0.0  # Defence ability score (0-100)

    # Combined final score (legacy, for compatibility)
    final_score: float = 0.0
    rank: int = 0
    attack_rank: int = 0
    defence_rank: int = 0

    # Feedback text
    feedback: str = ""


class GradeCalculator:
    """
    Calculates final grades from evaluation results.

    Formula (from Assignment PDF):
    - Attack Ability_j = sum(Score_i,j) for all i (column sum)
    - Defence Ability_i = sum(Score_i,j) for all j (row sum)
    - Normalize both using softmax
    - Final Score = base_score + attack_weight * normalized_attack + defence_weight * normalized_defence

    Note: In this implementation:
    - Attack score = mean(1/accuracy) excluding self - higher is better
    - Defence score = mean(accuracy) excluding self - higher is better
    """

    def __init__(self, config: MarkingConfig):
        """
        Initialize the grade calculator.

        Args:
            config: MarkingConfig instance
        """
        self.config = config
        self.student_grades: Dict[str, StudentGrade] = {}

    def calculate_from_single_matrix(
            self,
            matrix_df: pd.DataFrame,
            exclude_self: bool = True
    ) -> Dict[str, StudentGrade]:
        """
        Calculate grades from a single evaluation matrix.

        Args:
            matrix_df: DataFrame with Model_{id} as index, Attack_{id} as columns
            exclude_self: Whether to exclude self-attack scores

        Returns:
            Dictionary mapping student_id to StudentGrade
        """
        # Extract student IDs
        model_ids = [idx.replace('Model_', '') for idx in matrix_df.index]
        attack_ids = [col.replace('Attack_', '') for col in matrix_df.columns]

        # All unique student IDs
        all_ids = list(set(model_ids) | set(attack_ids))

        # Calculate raw scores
        attack_scores = {}
        defence_scores = {}

        for student_id in all_ids:
            # Calculate attack score (mean of 1/accuracy for this attack against all models)
            attack_col = f'Attack_{student_id}'
            if attack_col in matrix_df.columns:
                col_values = matrix_df[attack_col].values

                if exclude_self:
                    # Exclude self-attack (same model)
                    model_row = f'Model_{student_id}'
                    if model_row in matrix_df.index:
                        mask = matrix_df.index != model_row
                        col_values = matrix_df.loc[mask, attack_col].values

                # Attack score = mean(1/accuracy)
                # Higher attack score = better attack (lower accuracy on victims)
                # Filter out NaN and zero values
                col_values = col_values.astype(float)
                valid_values = col_values[(~np.isnan(col_values)) & (col_values > 0)]
                if len(valid_values) > 0:
                    attack_scores[student_id] = np.mean(1.0 / valid_values)
                else:
                    attack_scores[student_id] = 0.0
            else:
                # Student has no attack - mark as None to distinguish from low score
                attack_scores[student_id] = None

            # Calculate defence score (mean accuracy against all attacks)
            model_row = f'Model_{student_id}'
            if model_row in matrix_df.index:
                row_values = matrix_df.loc[model_row].values

                if exclude_self:
                    # Exclude self-attack
                    attack_col = f'Attack_{student_id}'
                    if attack_col in matrix_df.columns:
                        col_idx = list(matrix_df.columns).index(attack_col)
                        row_values = np.delete(row_values, col_idx)

                # Defence score = mean accuracy (use nanmean to handle NaN from averaging)
                # Higher defence score = better defence (higher accuracy under attack)
                defence_scores[student_id] = np.nanmean(row_values) if len(row_values) > 0 else 0.0
            else:
                # Student has no model - mark as None to distinguish from low score
                defence_scores[student_id] = None

        # Filter out None values for normalization
        valid_attack_scores = {k: v for k, v in attack_scores.items() if v is not None}
        valid_defence_scores = {k: v for k, v in defence_scores.items() if v is not None}

        # Calculate final scores based on normalization method
        attack_final_dict = {}
        defence_final_dict = {}
        normalized_attack_dict = {}
        normalized_defence_dict = {}

        if self.config.use_zscore:
            # Z-Score normalization: directly calculate final scores
            if valid_attack_scores:
                attack_ids = list(valid_attack_scores.keys())
                attack_values = list(valid_attack_scores.values())
                attack_final_values = self._zscore_to_final_scores(attack_values)
                for idx, student_id in enumerate(attack_ids):
                    attack_final_dict[student_id] = attack_final_values[idx]

            if valid_defence_scores:
                defence_ids = list(valid_defence_scores.keys())
                defence_values = list(valid_defence_scores.values())
                defence_final_values = self._zscore_to_final_scores(defence_values)
                for idx, student_id in enumerate(defence_ids):
                    defence_final_dict[student_id] = defence_final_values[idx]
        else:
            # Min-max normalization
            if valid_attack_scores:
                attack_values = list(valid_attack_scores.values())
                normalized_attack_values = self._normalize_scores(attack_values)
                for idx, student_id in enumerate(valid_attack_scores.keys()):
                    normalized_attack_dict[student_id] = normalized_attack_values[idx]

            if valid_defence_scores:
                defence_values = list(valid_defence_scores.values())
                normalized_defence_values = self._normalize_scores(defence_values)
                for idx, student_id in enumerate(valid_defence_scores.keys()):
                    normalized_defence_dict[student_id] = normalized_defence_values[idx]

        # Create StudentGrade objects
        for i, student_id in enumerate(all_ids):
            grade = StudentGrade(student_id=student_id)

            # Raw scores (None means no submission)
            raw_attack = attack_scores.get(student_id)
            raw_defence = defence_scores.get(student_id)

            grade.raw_attack_score = raw_attack if raw_attack is not None else 0.0
            grade.raw_defence_score = raw_defence if raw_defence is not None else 0.0

            # Calculate assignment scores
            if self.config.use_zscore:
                # Z-Score: use pre-calculated final scores
                if attack_scores.get(student_id) is None:
                    grade.attack_assignment_score = 0.0
                else:
                    grade.attack_assignment_score = attack_final_dict.get(student_id, 0.0)

                if defence_scores.get(student_id) is None:
                    grade.defence_assignment_score = 0.0
                else:
                    grade.defence_assignment_score = defence_final_dict.get(student_id, 0.0)

                # Normalized scores not used in Z-score mode, set to 0
                grade.normalized_attack = 0.0
                grade.normalized_defence = 0.0
            else:
                # Min-max: calculate from normalized scores
                grade.normalized_attack = normalized_attack_dict.get(student_id, 0.0)
                grade.normalized_defence = normalized_defence_dict.get(student_id, 0.0)

                base = self.config.base_score
                max_score = self.config.max_score

                if attack_scores.get(student_id) is None:
                    grade.attack_assignment_score = 0.0
                else:
                    grade.attack_assignment_score = base + (max_score - base) * grade.normalized_attack

                if defence_scores.get(student_id) is None:
                    grade.defence_assignment_score = 0.0
                else:
                    grade.defence_assignment_score = base + (max_score - base) * grade.normalized_defence

            # Calculate combined final score (legacy, average of both assignments)
            grade.final_score = (grade.attack_assignment_score + grade.defence_assignment_score) / 2

            # Generate feedback
            grade.feedback = self._generate_feedback(grade)

            self.student_grades[student_id] = grade

        # Calculate ranks
        self._calculate_ranks()

        return self.student_grades

    def calculate_from_multiple_matrices(
            self,
            matrices: List[pd.DataFrame],
            exclude_self: bool = True
    ) -> Dict[str, StudentGrade]:
        """
        Calculate grades from multiple evaluation rounds (averaged).

        Args:
            matrices: List of DataFrames from multiple rounds
            exclude_self: Whether to exclude self-attack scores

        Returns:
            Dictionary mapping student_id to StudentGrade
        """
        if len(matrices) == 0:
            raise ValueError("No matrices provided")

        if len(matrices) == 1:
            return self.calculate_from_single_matrix(matrices[0], exclude_self)

        # Average the matrices
        logger.info(f"Averaging {len(matrices)} evaluation rounds...")

        # Ensure all matrices have the same shape
        reference_shape = matrices[0].shape
        for i, m in enumerate(matrices):
            if m.shape != reference_shape:
                logger.warning(f"Matrix {i} has different shape: {m.shape} vs {reference_shape}")

        # Average
        averaged_matrix = sum(matrices) / len(matrices)

        return self.calculate_from_single_matrix(averaged_matrix, exclude_self)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of raw scores

        Returns:
            List of normalized scores
        """
        scores_array = np.array(scores)

        min_val = scores_array.min()
        max_val = scores_array.max()

        if max_val == min_val:
            # All scores are the same
            return [0.5] * len(scores)

        normalized = (scores_array - min_val) / (max_val - min_val)

        # Clip to ensure [0.01, 1.0] range (avoid zero scores)
        normalized = np.clip(normalized, 0.01, 1.0)

        return normalized.tolist()

    def _zscore_to_final_scores(self, raw_scores: List[float]) -> List[float]:
        """
        Convert raw scores to final scores using Z-score normalization.

        Formula: final_score = target_mean + (raw - mean) / std * target_std
        Then clip to [0, 100].

        Args:
            raw_scores: List of raw scores

        Returns:
            List of final scores (0-100 scale)
        """
        scores_array = np.array(raw_scores)

        mean_val = scores_array.mean()
        std_val = scores_array.std()

        target_mean = self.config.zscore_target_mean
        target_std = self.config.zscore_target_std

        if std_val == 0:
            # All scores are the same
            return [target_mean] * len(raw_scores)

        # Z-score transformation
        final_scores = target_mean + (scores_array - mean_val) / std_val * target_std

        # Clip to [0, 100]
        final_scores = np.clip(final_scores, 0, 100)

        return final_scores.tolist()

    def _calculate_ranks(self):
        """Calculate ranks based on scores"""
        # Overall rank (by combined final score)
        sorted_grades = sorted(
            self.student_grades.values(),
            key=lambda g: g.final_score,
            reverse=True
        )
        for rank, grade in enumerate(sorted_grades, 1):
            grade.rank = rank
            self.student_grades[grade.student_id].rank = rank

        # Attack rank (by attack assignment score)
        sorted_by_attack = sorted(
            self.student_grades.values(),
            key=lambda g: g.attack_assignment_score,
            reverse=True
        )
        for rank, grade in enumerate(sorted_by_attack, 1):
            self.student_grades[grade.student_id].attack_rank = rank

        # Defence rank (by defence assignment score)
        sorted_by_defence = sorted(
            self.student_grades.values(),
            key=lambda g: g.defence_assignment_score,
            reverse=True
        )
        for rank, grade in enumerate(sorted_by_defence, 1):
            self.student_grades[grade.student_id].defence_rank = rank

    def _generate_feedback(self, grade: StudentGrade) -> str:
        """
        Generate feedback text for a student.

        Args:
            grade: StudentGrade object

        Returns:
            Feedback string
        """
        if self.config.use_zscore:
            # Z-Score feedback
            target_mean = self.config.zscore_target_mean
            feedback = f"""COMP219 Competition Results for Student {grade.student_id}
{'='*70}

NOTE: Scores have been scaled using Z-score normalization to achieve
a class mean of {target_mean:.0f} for both attack and defence components.

{'='*70}

ASSIGNMENT 1: ATTACK ABILITY (Score: {grade.attack_assignment_score:.2f}/100)
{'─'*70}
  Raw Attack Score: {grade.raw_attack_score:.4f} (mean of 1/accuracy)
  Final Score (after Z-score scaling): {grade.attack_assignment_score:.2f}/100

{'='*70}

ASSIGNMENT 2: DEFENCE ABILITY (Score: {grade.defence_assignment_score:.2f}/100)
{'─'*70}
  Raw Defence Score: {grade.raw_defence_score:.4f} (mean accuracy under attack)
  Final Score (after Z-score scaling): {grade.defence_assignment_score:.2f}/100

{'='*70}

SUMMARY:
  Attack Assignment:  {grade.attack_assignment_score:.2f}/100
  Defence Assignment: {grade.defence_assignment_score:.2f}/100

{'='*70}
"""
        else:
            # Min-max feedback (original)
            base = self.config.base_score
            max_score = self.config.max_score

            feedback = f"""COMP219 Competition Results for Student {grade.student_id}
{'='*70}

ASSIGNMENT 1: ATTACK ABILITY (Score: {grade.attack_assignment_score:.2f}/100)
{'─'*70}
  Raw Attack Score: {grade.raw_attack_score:.4f} (mean of 1/accuracy)
  Normalized Score (0-1): {grade.normalized_attack:.4f}

  Score Calculation:
    Base Score: {base:.0f}
    + Performance Bonus: ({max_score:.0f} - {base:.0f}) × {grade.normalized_attack:.4f} = {(max_score - base) * grade.normalized_attack:.2f}
    ────────────────────────────────────────────────────────────
    ATTACK ASSIGNMENT SCORE: {grade.attack_assignment_score:.2f}/100

{'='*70}

ASSIGNMENT 2: DEFENCE ABILITY (Score: {grade.defence_assignment_score:.2f}/100)
{'─'*70}
  Raw Defence Score: {grade.raw_defence_score:.4f} (mean accuracy under attack)
  Normalized Score (0-1): {grade.normalized_defence:.4f}

  Score Calculation:
    Base Score: {base:.0f}
    + Performance Bonus: ({max_score:.0f} - {base:.0f}) × {grade.normalized_defence:.4f} = {(max_score - base) * grade.normalized_defence:.2f}
    ────────────────────────────────────────────────────────────
    DEFENCE ASSIGNMENT SCORE: {grade.defence_assignment_score:.2f}/100

{'='*70}

SUMMARY:
  Attack Assignment:  {grade.attack_assignment_score:.2f}/100
  Defence Assignment: {grade.defence_assignment_score:.2f}/100

{'='*70}
"""
        return feedback

    def save_results(self, output_dir: str, prefix: str = "grades"):
        """
        Save grade results to files.

        Args:
            output_dir: Output directory
            prefix: File name prefix
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save grades CSV with separate assignment scores
        grades_data = []
        for student_id, grade in self.student_grades.items():
            grades_data.append({
                'student_id': student_id,
                'attack_score': round(grade.attack_assignment_score, 2),
                'attack_rank': grade.attack_rank,
                'defence_score': round(grade.defence_assignment_score, 2),
                'defence_rank': grade.defence_rank,
                'raw_attack_score': grade.raw_attack_score,
                'raw_defence_score': grade.raw_defence_score,
                'normalized_attack': grade.normalized_attack,
                'normalized_defence': grade.normalized_defence,
            })

        grades_df = pd.DataFrame(grades_data)
        grades_df = grades_df.sort_values('attack_rank')

        csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
        grades_df.to_csv(csv_path, index=False)
        logger.info(f"Grades saved to {csv_path}")

        # Save individual feedback files
        feedback_dir = os.path.join(output_dir, "feedback")
        os.makedirs(feedback_dir, exist_ok=True)

        for student_id, grade in self.student_grades.items():
            feedback_path = os.path.join(feedback_dir, f"{student_id}_feedback.txt")
            with open(feedback_path, 'w', encoding='utf-8') as f:
                f.write(grade.feedback)

        logger.info(f"Individual feedback saved to {feedback_dir}")

        # Print summary statistics
        self._print_summary()

    def _print_summary(self):
        """Print grade distribution summary"""
        attack_scores = [g.attack_assignment_score for g in self.student_grades.values()]
        defence_scores = [g.defence_assignment_score for g in self.student_grades.values()]

        attack_array = np.array(attack_scores)
        defence_array = np.array(defence_scores)

        logger.info("\n" + "=" * 70)
        logger.info("GRADE DISTRIBUTION SUMMARY")
        logger.info("=" * 70)

        # Grade bands
        bands = [
            (0, 40, '<40'),
            (40, 50, '40-49'),
            (50, 60, '50-59'),
            (60, 70, '60-69'),
            (70, 80, '70-79'),
            (80, 90, '80-89'),
            (90, 101, '90-100')
        ]

        logger.info("\nATTACK ASSIGNMENT Distribution:")
        for low, high, label in bands:
            count = np.sum((attack_array >= low) & (attack_array < high))
            logger.info(f"  {label}: {count} students")
        logger.info(f"  Mean: {np.mean(attack_array):.2f}, Min: {np.min(attack_array):.2f}, Max: {np.max(attack_array):.2f}")

        logger.info("\nDEFENCE ASSIGNMENT Distribution:")
        for low, high, label in bands:
            count = np.sum((defence_array >= low) & (defence_array < high))
            logger.info(f"  {label}: {count} students")
        logger.info(f"  Mean: {np.mean(defence_array):.2f}, Min: {np.min(defence_array):.2f}, Max: {np.max(defence_array):.2f}")

        logger.info(f"\nTotal students: {len(attack_scores)}")
        logger.info("=" * 70)


def load_and_average_results(
        result_files: List[str],
        config: MarkingConfig
) -> Tuple[Dict[str, StudentGrade], pd.DataFrame]:
    """
    Load multiple result files and calculate averaged grades.

    Args:
        result_files: List of paths to evaluation CSV files
        config: MarkingConfig instance

    Returns:
        Tuple of (grades_dict, averaged_matrix)
    """
    matrices = []
    for file_path in result_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            matrices.append(df)
            logger.info(f"Loaded: {file_path}")
        else:
            logger.warning(f"File not found: {file_path}")

    if not matrices:
        raise ValueError("No valid result files found")

    calculator = GradeCalculator(config)
    grades = calculator.calculate_from_multiple_matrices(matrices)

    averaged_matrix = sum(matrices) / len(matrices)

    return grades, averaged_matrix
