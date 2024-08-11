# PGN SAM Pytorch

A PyTorch implementation of *Penalizing Gradient Norm for Efficiently Improving
Generalization in Deep Learning* (
Zhao+2022) [Paper](https://arxiv.org/abs/2202.03599), [Official implementation](https://github.com/zhaoyang-0204/gnp)
.

# Disclaimer

This is a fork of the original (unofficial) sam.pytorch [repo](https://github.com/moskomule/sam.pytorch). I have extended it to include the following:
- Implementation of the Penalizing gradient norm (PGN) method.
- Adjusted the code to work with a subset of the training data. This allows for parameter finetuning/

## Requirements

* Python>=3.8
* PyTorch>=1.7.1

To run the example, you further need

* `homura` by `pip install -U homura-core==2020.12.0`
* `chika` by `pip install -U chika`

## Example

```commandline
python cifar10.py [--optim.name {pgn_sam, sam, sgd}] [--model {renst20, wrn28_2}] [--optim.rho 0.05] [--optim.alpha 0.1]
```

## Citation

```bibtex
@ARTICLE{2020arXiv201001412F,
    author = {{Foret}, Pierre and {Kleiner}, Ariel and {Mobahi}, Hossein and {Neyshabur}, Behnam},
    title = "{Sharpness-Aware Minimization for Efficiently Improving Generalization}",
    year = 2020,
    eid = {arXiv:2010.01412},
    eprint = {2010.01412},
}

@software{sampytorch
    author = {Ryuichiro Hataya},
    titile = {sam.pytorch},
    url    = {https://github.com/moskomule/sam.pytorch},
    year   = {2020}
}

@software{pgnsampytorch
    author = {Syed Safwan Khalid},
    titile = {pgn_sam.pytorch},
    url    = {},
    year   = {2024}
}
```
