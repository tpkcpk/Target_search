## Example Dataset

This directory contains a small example dataset demonstrating the workflow of **Target Search**.

The example is based on the dehydration CID channel dataset used in our published study.

## Reference

P. K. Tsou, H. T. Phan, and J. L. Kuo, Phys. Chem. Chem. Phys., 2025, 27, 4355. Using building block structures and a cooperative approach with neural networks and random forest to identify reactions: a case study on the dissociation of sodiated disaccharides.


## Contents

```
example/
├── train.csv
├── test.csv
└── README.md
```

- `train.csv` contains the training dataset used to build the Random Forest model.
- `test.csv` contains unseen initial structures for prediction.

Each sample contains:

- Structural descriptors extracted from the **initial (unoptimized) structures**
- The corresponding optimized energies (training set only)

## Running the Example

Train a model

```bash
python target_search.py \
run=train \
train=example/train.csv
```

Predict the test set

```bash
python target_search.py \
run=predict \
model=rf_example.joblib \
test=example/test.csv
```

Expected outputs

```
pred_example.csv
plot_rf_hist*.png
```

## Notes

This dataset is provided solely as a demonstration of the workflow. Users can replace the example dataset with their own structural descriptors and optimized energies without modifying the source code.
