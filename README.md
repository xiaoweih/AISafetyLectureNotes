# Machine Learning Safety

This repository includes the code (in Chapter-X/) and the draft textbook (file "Machine_Learning_Safety.pdf"). 
It is used in "COMP219: Advanced Artificial Intelligence" at the Univesity of Liverpool for second year undergraduate students. 

### Table of Contents:  

#### Part 1: Safety Properties
#### Part 2: Safety Threats
#### Part 3: Safety Solutions
#### Part 4: Extended Safety Solutions
#### Part 5: Appendix: Mathematical Foundations and Competition


# Installation

## conda installation
``windows: https://conda.io/projects/conda/en/latest/user-guide/install/windows.html``

First of all, you can set up a conda environment (Note that you do not need to set up conda in the lab)

```sh
conda create --name aisafety python==3.7
conda activate aisafety
```
This should be followed by installing software dependencies:
```sh
conda install -c pandas numpy matplotlib tensorflow scikit-learn pandas pytorch torchvision
```
