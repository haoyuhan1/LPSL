# Code for "Towards Label Position Bias in Graph Neural Networks"

Here we provide one sample script for the Cora dataset, due to the paper is still under review.

```python
python fair_train.py  --lr 0.01 --dropout 0.8  --weight_decay 5e-4 --runs 3  --model Fairness --device 0 --random_split 10 --fix_num 20 --Rho 0.01 --alr 0.01 --epsilon 0.01 --dataset Cora --lambda1 10  --alpha 0.1 --K 10  --C 1
```

