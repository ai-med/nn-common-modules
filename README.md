# NN Common Modules

Common modules, blocks and losses which can be reused in a deep neural netwok specifically for segmentation Please use [technical documentation](https://shayansiddiqui.github.io/nn-common-modules/build/html/) for a reference to API manual

This project has 3 modules 
* Losses (losses.py) -> It has all the loss functions defined as python classes
    1. DiceLoss
    2. IoULoss
    3. CrossEntropyLoss2d
    4. CombinedLoss
    5. FocalLoss
* Modules (modules.py) -> It has all the commonly used building blocks of an FCN
    1. DenseBlock
    2. EncoderBlock
    3. DecoderBlock
    4. ClassifierBlock
    5. GenericBlock
    6. SDNetEncoderBlock
    7. SDNetDecoderBlock    
* Bayesian Modules (bayesian_modules.py) -> It has all the commonly used building blocks of a Bayesian FCN
    1. BayesianConvolutionBlock
    2. BayesianEncoderBlock
    3. BayesianDecoderBlock
    4. BayesianClassifierBlock

## Getting Started

### Pre-requisites

You need to have following in order for this library to work as expected
1. Python >= 3.5
2. Pytorch >= 1.0.0
3. Numpy >= 1.14.0

### Installing

Always use the latest release. Use following command with appropriate version no(v1.0) in this particular case to install. You can find the link for the latest release in the release section of this github repo

```
pip install https://github.com/shayansiddiqui/nn-common-modules/releases/download/v1.0/nn_common_modules-1.0-py2.py3-none-any.whl
```

## Authors

* **Shayan Ahmad Siddiqui**  - [shayansiddiqui](https://github.com/shayansiddiqui)
* **Abhijit Guha Roy**  - [abhi4ssj](https://github.com/abhi4ssj)


## Help us improve
Let us know if you face any issues. You are always welcome to report new issues and bugs and also suggest further improvements. And if you like our work hit that start button on top. Enjoy :)
