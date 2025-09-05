# mnist

A minimal, self-contained PyTorch 2.7 repository that reproduces two foundational convolutional networks for handwritten-digit recognition on MNIST:

1. **MLP** – a small fully-connected baseline (784 -> 128 -> 10)
2. **LeNet-5** – faithful re-implementation of the 1998 LeCun et al. architecture ([paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf))

## Quick start
```bash
# 1. install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh       # macOS / Linux
# or on Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. clone & enter repo
git clone https://github.com/asatpathy314/mnist.git
cd mnist

# 3. create venv + install deps
uv sync
```

## Structure
```
mnist/
├── src/
│   ├── mlp.ipynb
│   └── lenet5.ipynb
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

## Citation
If you use this code, please cite the original LeNet-5 paper:

## Results
Currently our network returns

## Future Work
In the original paper, they state:
> "The parameter vectors of these units were chosen by hand and kept fixed (at least initially). [...] they were instead designed to represent a stylized image of the corresponding character classdrawn on a 7x12 bitmap."

Currently the RBF unit initializes randomly with 0s and 1s. In future extensions of this work I may implement this.

```
@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  year={1998}
}
```