# A Solver-Free Framework for Scalable Learning in Neural ILP Architectures

This repository contains the code to reproduce the results reported in the paper [A Solver-Free Framework for Scalable Learning in Neural ILP Architectures][paper], which has been accepted at NeurIPS 2022. We also provide the core components of our technique as a light-weight python [library][library].

# Install

```
git clone https://github.com/dair-iitd/ilploss
cd ilploss
conda env create -f env_export.yaml
conda activate ilploss
```

Download and unzip the data from [here](https://drive.google.com/file/d/1jP80OhGPCbkYudhvC1EjOZXtipgNgFYL/view?usp=sharing) into a directory named `data/`.

# Run

```
./trainer.py --config <path-to-config>
```

All our experiments are available as config files in the `conf/` directory. For example to train and test ILP-Loss on random constraints for the binary domain with 8 ground truth constraints and dataset seed 0, run:

```
./trainer.py --config conf/neurips/binary_random/ilploss/8x16/0.yaml
./trainer.py --config conf/neurips/binary_random/ilploss/test/8x16/0.yaml
```

Similarly, for CombOptNet, run:

```
./trainer.py --config conf/neurips/binary_random/comboptnet/8x16/0.yaml
./trainer.py --config conf/neurips/binary_random/comboptnet/test/8x16/0.yaml
```

__[TODO: add instructions for neural baselines]__

# Citation

```
@inproceedings{ilploss,
  author = {Nandwani, Yatin and Ranjan, Rishabh and Mausam and Singla, Parag},
  title = {A Solver-Free Framework for Scalable Learning in Neural ILP Architectures},
  booktitle = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, November 29-Decemer 1, 2022},
  year = {2022},
}
```


[paper]: https://arxiv.org/abs/2210.09082
[library]: https://github.com/rishabh-ranjan/ilploss
