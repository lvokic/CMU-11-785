# HW1P2 starter

This directory is a clean starter for the frame-level speech classification
task. The data is not included. Download the five files from the course/Kaggle
data page:

```text
train.npy
train_labels.npy
dev.npy
dev_labels.npy
test.npy
```

Implement the TODOs in this order:

1. `learninghw1.py`: pad utterances and return flattened context windows.
2. `models.py`: create a small MLP that outputs 71 logits.
3. `learninghw1.py`: create train/validation/test data loaders.
4. `learninghw1.py`: generate `id,label` predictions in `test()`.
5. `main.py`: train for a short run, then tune context, batch size, learning
   rate, width, depth, batch normalization, and dropout.

Start with `K=5`, batch size `1024`, learning rate `1e-3`, and no dropout.
