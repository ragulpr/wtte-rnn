# WTTE-RNN (Python Implementation / API)

[![Build Status](https://travis-ci.org/ragulpr/wtte-rnn.svg?branch=master)](https://travis-ci.org/ragulpr/wtte-rnn)

Weibull Time To Event Recurrent Neural Network

A less hacky machine-learning framework for churn- and time to event prediction.
Forecasting problems as diverse as server monitoring to earthquake- and
churn-prediction can be posed as the problem of predicting the time to an event.
WTTE-RNN is an algorithm and a philosophy about how this should be done.

* [blog post](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/)
* [master thesis](https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf)
* Quick visual intro to the [model](https://imgur.com/a/HX4KQ)


# Installation

Install via PyPI.
We recommend updating pip/setuptools first.

`wtte` provides multiple extra dependency sets to install additional
dependencies for your environment and purposes.

```console
$ pip install -U pip setuptools
$ pip install wtte[extras]
```

In `extras` you may specify the followings:

 * `tf`: install with TensorFlow (CPU-version)
 * `tf_gpu`: install with TensorFlow (GPU-version)
 * `plot`: install matplotlib for additional plotting support
 * `build`: install additional packages to build your own distribution package
 * `test`: install additional packages to run test suite
 * `dev`: install additional packages used for local development (including
   documentation tools)
 * `docs`: install additional packages used for documentation builds in
   readthedocs build-farm servers

## Development

Follow the below instructions.
Change `tf` to `tf_gpu` if your machine has CUDA GPUs.

```console
$ git clone $thisrepo
$ cd ./wtte-rnn/python/
$ pip install -U pip setuptools
$ pip install -r requirements-dev.txt
$ pip install -e .[build,test,dev,tf]
```


# Licensing

* MIT License

## Citation

```
@MastersThesis{martinsson:Thesis:2016,
    author = {Egil Martinsson},
    title  = {{WTTE-RNN : Weibull Time To Event Recurrent Neural Network}},
    school = {Chalmers University Of Technology},
    year   = {2016},
}
```

## Contributing
Contributions/PR/Comments etc are very welcome! Post an issue if you have any questions and feel free to reach out to egil.martinsson[at]gmail.com.

### Contributors (by order of commit)

* Egil Martinsson
* Clay Kim
* Jannik Hoffjann
* Daniel Klevebring
* Jeongkyu Shin 
* Joongi Kim 
* Jonghyun Park
