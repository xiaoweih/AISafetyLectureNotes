
# COMP219 Assignment - Testing Guide

## Setup

### 1. Environment Requirements

**Required: Python 3.13**

This assignment requires Python 3.13 environment. For installation instructions, please refer to the GitHub repository main page.

### 2. Download Files from GitHub

Download the following files from the GitHub repository:
- `Competition.py` - Your submission template
- `Reference_attacks.py` - Reference attack methods (for testing only)
- `Evaluate_attack.py` - Attack score evaluation script
- `Evaluate_defence.py` - Defence score evaluation script
- `README_baseline.md` - This guide

**Optional (for Extended mode testing):**
- `Extended_attacks.py` - Extended Attack loader
- `_attacks_internal/` folder - 15 compiled attack files (.pyc)

### 3. Download Reference Models from Canvas

1. Go to Canvas → COMP219 Module → Lecture 14
2. Download `Defenders.zip`
3. Extract to `./Defenders` folder in Competition directory

---

## File Structure

```
Competition/
├── Competition.py              # Your submission file
├── Reference_attacks.py        # Baseline attacks (for testing only)
├── Evaluate_defence.py         # Test defence score
├── Evaluate_attack.py          # Test attack score
├── Extended_attacks.py         # Extended Attack loader (optional)
├── _attacks_internal/          # Compiled attack files (optional)
│   ├── attack_001.pyc
│   ├── attack_002.pyc
│   └── ... (15 attacks total)
└── Defenders/                  # Reference models (download from Canvas - see Setup)
    ├── 0.pt
    ├── 1.pt
    └── ... (10 models total)
```

**Note**:
- The `Defenders/` folder is **not included in GitHub**. You must download `Defenders.zip` from Canvas → COMP219 Module → Lecture 14, then extract it to the Competition directory.
- The `Extended_attacks.py` and `_attacks_internal/` are **optional** and only needed if you want to use extended mode testing.

---

## Usage

### 1. Test Your Attack Score

```bash
python Evaluate_attack.py
```

**Configure** in `Evaluate_attack.py`:

```python
# Step 1: Set Defenders path
defenders_path = "./Defenders"

# Step 2: Define your attack (copy from Competition.py)
def your_attack(model, X, y, device):
    # Your attack implementation here
    return X_adv
```

**Output**:
```
ATTACK MATRIX
Model          Robust Accuracy      Attack Score (1/acc)
0.pt                 52.30%              1.91
1.pt                 48.15%              2.08
...
Mean                 49.82%              2.15

ATTACK SCORE
Your Attack Score: 2.1523

Note:
  - Higher score is better (stronger attack)
  - Final grading will normalize scores across ALL students
```

---

### 2. Test Your Defence Score

```bash
python Evaluate_defence.py
```

**Configure** in `Evaluate_defence.py`:

```python
# Step 1: Specify your model file
your_model_file = "1000.pt"  # <-- CHANGE THIS to your model path

# Step 2: Select attack mode
attack_mode = 'baseline'  # <-- CHANGE THIS to 'baseline' or 'extended'
```

#### Attack Modes

**Baseline Mode** - Fast testing (recommended for development)
- Tests against 3 reference attacks: FGSM, PGD-5, PGD-20
- Quick execution
- Good for rapid iteration during development

**Extended Mode** - Comprehensive testing (recommended before submission)
- Tests against 15 Advanced Attacks
- More diverse attack strategies
- Better assessment of defence robustness

**Which mode to use?**
- Use `'baseline'` for quick testing during development
- Use `'extended'` for thorough testing before final submission
- Your defence should perform well on both modes

**Output Example**:
```
DEFENCE EVALUATION RESULTS

Attack               Robust Accuracy
--------------------------------------------------------------------------------
FGSM                  45.23%
PGD-5                 38.67%
PGD-20                32.15%
--------------------------------------------------------------------------------
Mean (Defence Score)  38.68%

Your Defence Score: 0.3868

Note:
  - Higher score is better (more robust model)
  - Final grading will normalize scores across ALL students
  - Current mode: baseline
  - Tip: Try 'extended' mode for more comprehensive testing
```

#### Optional: Add Custom Attacks

You can add your own attacks for testing (Step 3 in `Evaluate_defence.py`):

```python
def my_custom_attack(model, X, y, device):
    """
    Your custom attack implementation

    Args:
        model: the neural network to attack
        X: input data [batch_size, 784]
        y: true labels [batch_size]
        device: torch.device

    Returns:
        X_adv: adversarial examples [batch_size, 784]

    Note: Perturbation L∞ norm must be STRICTLY < 0.11 (use ≤ 0.109 to be safe)
    """
    epsilon = 0.1
    # Your attack logic here
    X_adv = X + epsilon * torch.randn_like(X)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)
    return X_adv

# Add to test suite
test_attacks.append(('MyAttack', my_custom_attack))
```

---

## Understanding Scores

### Attack Score
- Calculated as: mean of `1/robust_accuracy` of your attack against reference models
- **Higher is better** (stronger attack)
- Lower robust accuracy → stronger attack → higher score

### Defence Score
- Calculated as: mean robust accuracy of your model against reference attacks
- **Higher is better** (more robust model)

### Final Grading
- Your self-test score is for reference only
- Final grading will normalize scores using **all students' scores**
- This ensures fair comparison across the class

---

## Reference Attacks

Three baseline attacks are provided **for testing purposes only**:

| Attack  | Description |
|---------|-------------|
| FGSM    | Fast one-step attack |
| PGD-5   | 5-step iterative attack |
| PGD-20  | 20-step iterative attack |

### ⚠️ IMPORTANT - Epsilon Constraint

- All attacks must satisfy: **L∞ norm STRICTLY < 0.11** (NOT equal to 0.11)
- Safe value: use epsilon ≤ 0.109
- Verify your attack: check `p_distance` output in `Competition.py`
- The limit applies to **every image**. Any adversarial image that contains **NaN/Inf** or has L∞ ≥ 0.11 is replaced by the original image, i.e. counted as **not attacked** (no credit for your attack). If your attack raises an error or returns the wrong shape, that whole batch counts as not attacked.
- Valid images are **clamped to [0, 1]** before evaluation, so leaving the image range gives no advantage. We recommend clamping in your own attack.
- Both evaluation scripts apply exactly the same rules as the final marking and report how many of your images were invalid.

### ⚠️ IMPORTANT - Academic Integrity

- These baselines are **for evaluating your defence only**
- **DO NOT directly copy these methods for your submission**
- Simply using these attacks **will NOT give you good marks**
- You must **develop your own attack methods** based on:
  - Course materials (Lab 7.1 and 7.2)
  - Lecture content
  - Your own research and improvements

**Develop your own methods!** These baselines help you test, not solve the assignment.

---

## Recommended Workflow

1. **Develop** your attack/defence in `Competition.py`
2. **Train** your model and generate `.pt` file using `Competition.py`
3. **Test** attack: `python Evaluate_attack.py`
4. **Test** defence (quick): Set `attack_mode = 'baseline'` in `Evaluate_defence.py`, then run `python Evaluate_defence.py`
5. **Test** defence (thorough): Set `attack_mode = 'extended'` in `Evaluate_defence.py`, then run `python Evaluate_defence.py`
6. **Iterate** based on scores
7. **Verify** epsilon constraint: check `p_distance < 0.11` in `Competition.py` output
8. **Submit** your final `Competition.py` + model `.pt` file

---

## Troubleshooting

**"Defenders folder not found"**
- Download and extract `Defenders.zip` from Canvas Lecture 14

**"Model file not found"**
- Check path in `Evaluate_defence.py` Step 1
- Train model first with `Competition.py`

**"Epsilon constraint violated"**
- Check `p_distance` output in `Competition.py`
- Must be STRICTLY < 0.11 (use ≤ 0.109 to be safe)

**"RuntimeError: element 0 of tensors does not require grad"**
- This has been fixed in the latest version
- Make sure you're using the updated evaluation scripts

**"Could not load Extended Attacks" when using extended mode**
- Download `Extended_attacks.py` and `_attacks_internal/` folder from GitHub
- Or use `attack_mode = 'baseline'` instead (baseline mode is sufficient for testing)

**"Bad magic number" or "RuntimeError" when using extended mode**
- Ensure you are using Python 3.13 as required
- The .pyc files are compiled with Python 3.13 and won't work on other versions
- See GitHub main page for Python 3.13 installation instructions

---

## Summary

**Quick Start:**
1. ✅ Download files from GitHub
2. ✅ Download `Defenders.zip` from Canvas
3. ✅ Set model file in `Evaluate_defence.py` (Step 1)
4. ✅ Set `attack_mode = 'baseline'` for quick testing (Step 2)
5. ✅ Run: `python Evaluate_defence.py`
6. ✅ Before submission: try `attack_mode = 'extended'`
7. ✅ Submit: `yourID.zip` (containing `yourID.pt` and `competition_yourID.py`)

**Good luck! 🎯**
