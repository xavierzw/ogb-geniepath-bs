# GeniePath-BS: GeniePath with Bandit Sampler

The paper will be published on Arxiv soon.

## Datasets
The ogbn-proteins dataset will be downloaded in directory ./dataset automatically.

## Dependencies
- [tensorflow = 1.13.1](https://github.com/tensorflow/tensorflow)
- [ogb 1.1.1](https://github.com/snap-stanford/ogb)

## Prepare
```bash
python cython_sampler/setup.py build_ext -i
```

## How to run
```bash
python train.py --dataset ogbn-proteins --learning_rate 1e-3 --epochs 300 --hidden1 64 --neighbor_limit 10 --batchsize 256
```

or
```bash
source main.sh
```

## Performance
We train our models for 200 epochs and report the **rocauc** on the test dataset.
|dataset|mean|std|#experiments|
|-|-|-|-|
|ogbn-proteins|0.78253|0.00352|10|
