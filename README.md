# Code for work A Robust Training Method for Federated Learning with Partial Participation
**Experimental Setup for the Paper**

This repository includes all experiments and implementations required for the paper.

---

## ✅ Architectures and Datasets

- **Model**: ResNet-18  
- **Dataset**: CIFAR-10  
- **Client Configurations**: Experiments conducted with 10  
- **Additional Model Support**: FasterVIT with the Food101 dataset  

---

## ✅ Implemented Client Selection Strategies

| Method | Command | Description |
|--------|---------|-------------|
| Top-k via loss | `method: 'loss'; k: k` | Selects top-k clients with the highest loss values |
| Top-k via gradient norm | `method: 'gradient_norm'; k: k` | Selects top-k clients with the largest gradient norms |
| Top-k via BANT method | `method: 'bant'; k: k` | Uses BANT trust scores to select k clients with the highest trust |
| Top-k via random sampling | `method: 'random'; k: k` | Randomly selects k clients on each training round |

---

## ✅ Additional Strategies

| Method | Description |
|--------|-------------|
| Hybrid top-k via method1 + top-t via method2 | Selects k clients using method1 and t additional clients using method2 |
| Top-k via gradient angle similarity | Selects k clients whose gradients have the smallest deviation from the mean gradient direction |

---

## ✅ Command to Launch Experiments

```bash
    python utils/cifar_download.py
    python train.py
```
