# COMP219 Marking System

Automated marking system for the COMP219 adversarial attack/defence competition.

## Directory Structure

```
marking_system/
├── config.py             # Configuration (paths, evaluation params, grading)
├── main.py               # Main controller and CLI entry point
├── utils.py              # Utility functions and neural network definitions
├── file_extractor.py     # Extracts and organises student submissions
├── dependency_manager.py # Auto-installs dependencies from student code
├── evaluator.py          # Mutual evaluation engine
├── grade_calculator.py   # Score calculation and normalisation
├── __init__.py           # Package init
└── README.md             # This file
```

## Quick Start

### 1. Setup

Place your `submissions.zip` file in this directory (or pass a custom path with `--submissions`).

### 2. Run the Full Pipeline

```bash
python main.py
```

This executes the complete pipeline:
1. **Extract** student `.py` (attack) and `.pt` (model) files from the zip
2. **Evaluate** all attacks against all models over multiple rounds
3. **Grade** using normalised scoring

### 3. Step-by-Step Execution

```bash
# Extract files only
python main.py --extract-only

# Evaluate only (files must already be extracted)
python main.py --evaluate-only

# Calculate grades from existing evaluation results
python main.py --grade-only
```

## CLI Reference

### Mode Selection (mutually exclusive)

| Flag | Description |
|------|-------------|
| *(none)* | Run full pipeline (extract + evaluate + grade) |
| `--extract-only` | Only extract and organise submission files |
| `--evaluate-only` | Only run evaluation (requires extracted files) |
| `--grade-only` | Only calculate grades from existing results |
| `--rerun-attacks ID [ID ...]` | Re-run specific attack evaluations |
| `--rerun-models ID [ID ...]` | Re-run specific model evaluations |
| `--merge-partitions N` | Merge N partition results into complete matrices |
| `--update-from-fixed` | Update matrix with corrected files from `fixed_*_files/` |

### Configuration Overrides

| Flag | Default | Description |
|------|---------|-------------|
| `--submissions PATH` | `submissions.zip` | Path to submissions zip file |
| `--work-dir PATH` | *(auto-detected)* | Working directory |
| `--output-dir PATH` | `results/` | Output directory for results |
| `--num-rounds N` | 3 | Number of evaluation rounds |
| `--start-round N` | 1 | Starting round number |
| `--num-samples N` | 500 | Number of FashionMNIST test samples |
| `--seed N` | 88 | Random seed for reproducibility |

### Re-run Options

| Flag | Description |
|------|-------------|
| `--existing-matrix PATH` | Path to existing evaluation CSV (single round) |
| `--round N` | Single round number for re-run |
| `--rounds 1,2,3` | Comma-separated round numbers |
| `--matrix-pattern PATTERN` | Matrix filename pattern with `{}` for round number |

### Distributed Evaluation

| Flag | Description |
|------|-------------|
| `--partition N/M` | Run partition N of M (e.g., `1/3`) |

## Usage Examples

```bash
# Full pipeline with custom submissions file
python main.py --submissions /path/to/submissions.zip

# Re-run specific attacks for round 1
python main.py --rerun-attacks 201234567 201234568 --round 1

# Re-run specific models across multiple rounds
python main.py --rerun-models 201234567 --rounds 1,2,3

# Distributed evaluation (run on separate machines, then merge)
python main.py --evaluate-only --partition 1/3
python main.py --evaluate-only --partition 2/3
python main.py --evaluate-only --partition 3/3
python main.py --merge-partitions 3

# Update matrix with corrected student files
python main.py --update-from-fixed --existing-matrix results/evaluation_round_1.csv
```

## How It Works

### Evaluation Matrix

The system generates an *n x n* matrix where:
- **Rows** = defence models (Model_i)
- **Columns** = attack algorithms (Attack_j)
- **Cell [i, j]** = accuracy of Model_i under Attack_j

### Scoring

- **Attack score** = mean of (1 / accuracy) when attacking all other models (self excluded)
- **Defence score** = mean accuracy when attacked by all others (self excluded)

Scores are normalised (min-max by default, Z-score optional via `use_zscore=True`) and mapped to the configured grade range.

### Validity of Attack Outputs

Attack outputs are checked **image by image, on the raw output before clamping**:

| Case | Treatment |
|---|---|
| Image contains NaN or Inf | Replaced by the original image (counted as **not attacked**) |
| Image has L∞ perturbation **≥ 0.11** (the limit is strict) | Replaced by the original image (counted as **not attacked**) |
| Attack raises an error, or returns a non-tensor / wrong shape | Whole batch counted as **not attacked** |
| Valid image | Clamped to **[0, 1]** (`clamp_to_valid_range`), then evaluated |

So an attack only gets credit for the images it attacks validly, and a defender is never rewarded or penalised for another student's invalid output: on an invalid image it is simply evaluated on the clean image. Clamping can only reduce the perturbation, so it never causes an L∞ violation. Every test image is always counted.

The number of invalid images per (model, attack) is saved to `results/invalid_images_round_N.csv` (informational; it is already reflected in the accuracies).

### 0% Accuracy

The attack score of one result is `1 / max(accuracy, 1 / (num_test_samples × rounds))`. Every non-zero result is unchanged, and a 0% result counts as the strongest result instead of being dropped.

## Output Files

After a full run, the `results/` directory contains:

```
results/
├── evaluation_round_1.csv       # Round 1 evaluation matrix
├── evaluation_round_2.csv       # Round 2 evaluation matrix
├── evaluation_round_3.csv       # Round 3 evaluation matrix
├── invalid_images_round_*.csv   # Invalid adversarial images per (model, attack)
├── grades_summary.csv           # Final grades
├── feedback/                    # Per-student feedback files
│   ├── 201234567_feedback.txt
│   └── ...
└── marking_*.log                # Run logs
```

## Author

COMP219 Teaching Team, University of Liverpool
