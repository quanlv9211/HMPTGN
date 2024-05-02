## 1. Overview

This is the Pytorch implementation for "Toward a Manifold-Preserving Temporal Graph Network in Hyperbolic Space (IJCAI24)"

Authors: Quan Le, Cuong Ta

Paper:

![alt text](https://github.com/quanlv9211/HMPTGN/blob/main/figures/HMPTGN_framework.pdf)

## 2. Setup

### 2.1 Environment
`pip install -r requirements.txt`

### 2.2 Datasets
The data is cached in `./data/input/cached`.

## 3. Experiments
3.0 Go to the script at first

```cd ./script```

3.1 To run HMPTGN:

```python main.py --model=HMPTGN --dataset=enron10 --lr=0.002 --seed=998877```

3.2 Seed: 998877, 23456, 900.

3.3 Dataset choices: disease, enron10, dblp, uci, mathoverflow, fbw.

## 4. Baselines
For the baselines, please follow these repos and papers:
- [HGWaveNet](https://github.com/TaiLvYuanLiang/HGWaveNet)
- [HTGN](https://github.com/marlin-codes/HTGN)
- [VGRNN](https://github.com/VGraphRNN/VGRNN)
- [EvolveGCN](https://github.com/IBM/EvolveGCN)
- [DySAT](https://github.com/FeiGSSS/DySAT_pytorch)
- [DHGAT](https://doi.org/10.1016/j.neucom.2023.127038)
