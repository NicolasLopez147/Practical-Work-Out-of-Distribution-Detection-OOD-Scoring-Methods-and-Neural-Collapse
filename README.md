# TP_OOD – OOD Detection and Neural Collapse (ResNet-18)

## Group members
- Nicolas LOPEZ NIETO
- Nicolas RINCON VIJA

## Overview
This repository contains our work for the **TP_OOD** practical session on **Out-of-Distribution (OOD) detection** and **Neural Collapse**. We train a **ResNet-18** classifier on **CIFAR-100** (in-distribution) and evaluate post-hoc OOD detection using **SVHN** (out-of-distribution). We also analyze Neural Collapse using penultimate-layer features and additional layer-wise analysis.

The main experiments were executed on a server using SLURM scripts, and the results (figures + report) are stored in this repository.

## Repository structure
- `model.py`  
  Training script for ResNet-18 on CIFAR-100. It creates the train/val/test split, trains the model, saves the best checkpoint, and exports useful artifacts (training curves, penultimate features, NECO parameters).

- `calculate.py`  
  Post-processing script. It loads the best checkpoint and the saved artifacts, computes OOD scores (MSP, MaxLogit, Energy, Mahalanobis, ViM, NECO), generates plots, and runs Neural Collapse analysis (NC1–NC5 + bonus across layers).

- `run.sh`  
  SLURM script used to run the training on the server. 

- `run_calculate.sh`  
  SLURM script used to run the post-processing (`calculate.py`) on the server. 

- `figures_part2/`  
  Output folder containing the generated plots (OOD histograms/curves and Neural Collapse figures).

- `Final_Report_ODD.pdf`  
  Final report with the full methodology, experiments, results, and discussion.

- `slurm-12881.out`, `slurm-12914.out`  
  Example SLURM logs from server runs.  
