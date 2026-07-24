### Target Search ###

Target Search is a machine-learning workflow for accelerating geometry optimization of molecular structures.

Instead of performing geometry optimization on every generated initial structure, the model learns the relationship between **initial structural descriptors** and the **optimized energy** from a training dataset. It then predicts the optimized energies of unseen initial structures, allowing likely low-energy candidates to be identified before expensive quantum chemical calculations.

By ranking structures according to their predicted energies (or classifying them as low-energy candidates), only approximately **10–30%** of all generated structures need to be optimized while still recovering nearly all structures within the target energy window (typically the global minimum + 20 kJ/mol).


## Overview

Quantum chemical geometry optimization is often the computational bottleneck during conformational searches because every generated initial structure must be optimized before its energy is known.

Target Search addresses this problem by training a Random Forest model using:

- **Input:** structural descriptors extracted from initial (unoptimized) structures
- **Target:** energies after geometry optimization

The trained model predicts the optimized energies of new initial structures, allowing users to:

- Rank structures by predicted energy
- Select only promising low-energy candidates
- Greatly reduce the number of expensive geometry optimizations

In typical applications, only **10–30%** of all generated structures need to be optimized while maintaining excellent coverage of conformers within **20 kJ/mol above the global minimum**.

# Ｗorkflow
Generated initial structures
            │
            ▼
Extract structural descriptors
(dihedrals, puckering, ion coordination, ...)
            │
            ▼
Training dataset
Initial descriptors + Optimized energies
            │
            ▼
Random Forest
            │
            ▼
Predict optimized energies
for unseen structures
            │
            ▼
Rank candidates
or classify low-energy structures
            │
            ▼
Geometry optimization
only for the top 10–30%
            │
            ▼
Recover structures within
GM + 20 kJ/mol


Reference:

P. K. Tsou, H. T. Phan, and J. L. Kuo, Phys. Chem. Chem. Phys., 2025, 27, 4355. Using building block structures
and a cooperative approach with neural networks and random forest to identify reactions: a case study on the
dissociation of sodiated disaccharides.
