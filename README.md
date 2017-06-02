# WTTE-RNN
Weibull Time To Event Reccurent Neural Network

A less hacky machine-learning framework for churn- and time to event prediction. Forecasting problems as diverse as server monitoring to earthquake- and churn-prediction can be posed as the problem of predicting the time to an event. WTTE-RNN is an algorithm and a philosophy about how this should be done. 

* [blog post](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/)
* [master thesis](https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf)
* Quick visual intro to the [model](https://imgur.com/a/HX4KQ) 
* Jupyter notebook: [basics](examples/keras/simple_example.ipynb), [end to end](examples/data_pipeline/data_pipeline.ipynb )

# Installation
For the python implementation:

    $ git clone https://github.com/ragulpr/wtte-rnn  
    $ cd python
    $ python setup.py install

If this seems like overkill, the basic implementation can be found inlined as a [jupyter notebook](examples/keras/standalone_simple_example.ipynb) 

## Basics
You have data consisting of many time-series of events and want to use historic data to predict the time to the next event (TTE). If you haven't observed the last event yet we've only observed a minimum bound of the TTE to train on. This results in what's called *censored data* (in red):

![Censored data](./readme_figs/data.gif)

Instead of predicting the TTE itself the trick is to let your machine learning model output the *parameters of a distribution*. This could be anything but we like the *Weibull distribution* because it's [awesome](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/#embrace-the-weibull-euphoria). The machine learning algorithm could be anything gradient-based but we like RNNs because they are [awesome](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) too.

![example WTTE-RNN architecture](./readme_figs/fig_rnn_weibull.png)

The next step is to train the algo of choice with a special log-loss that can work with censored data. The intuition behind it is that we want to assign high probability at the *next* event or low probability where there *wasn't* any events (censored data): 

![WTTE-RNN prediction over a timeline](./readme_figs/solution_beta_2.gif)

What we get is a pretty neat prediction about the *distribution of the TTE* in each step (here for a single event):

![WTTE-RNN prediction](./readme_figs/it_61786_pmf_151.png)

A neat sideresult is that the predicted params is a 2-d embedding that can be used to visualize and group predictions about *how soon* (alpha) and *how sure* (beta). Here by stacking timelines of predicted alpha (left) and beta (right):

![WTTE-RNN alphabeta.png](./readme_figs/alphabeta.png)

## Warnings
There's alot of mathematical theory basically justifying us to use this nice loss function in certain situations:

![loss-equation](./readme_figs/equation.png)


So for censored data it only rewards *pushing the distribution up*, beyond the point of censoring. To get this to work you need the censoring mechanism to be independent from your feature data. If your features contains information about the point of censoring your algorithm will learn to cheat by predicting far away based on probability of censoring instead of tte. A type of overfitting/artifact learning. Global features can have this effect if not properly treated.

# Repo
Currently implemented in python/numpy:

* transformations
    * data pipeline transformations (pandas dataframes of expected format to numpy)
    * time to event and censoring indicator calculations
* weibull functions (cdf,pdf,quantile,mean etc)

* Objective functions
    * tensorflow
    * keras
* layers
    * weibull output layer in keras

# ROADMAP/TODO
The project is on the TODO-state. The goal is to create a forkable and easily deployable model framework. WTTE-RNN is the algorithm, churn_watch is the deployment - an opinionated idea about how churn-monitoring and reporting can be made beautiful and easy. Pull-requests, recommendations, comments and contributions very welcome.

## Implementations of the objective functions
The core technology is the objective functions. These can be used with any machine-learning algorithm. To spread the word we should implement and commit them to various ML-projects. 

* Tensorflow (DONE)
* Keras (tf wrapper DONE)
* MXnet
* Theano
* TORCH
* H2o
* scikitFlow
* MLlib

## Auxiliary

To use the model one needs basic tte-transforms of raw data. To consume the models we need weibull related functions for the final output.
* Ready to run helper functions implemented in SQL, R, Python.

## Monitoring 
The WTTE-RNN is as much an ML-algorithm as a visual language to talk about this shape of data and our predictions.
* Plots (partly done)
* Shiny webapp or/and similar (partly done elsewhere)
* Integration. Slack/E-mail bots & summaries
* API 

# Licensing
* MIT-license. 

# Citation

	@MastersThesis{martinsson:Thesis:2016,
	    author     =     {Egil Martinsson},
	    title     =     {WTTE-RNN : Weibull Time To Event Recurrent Neural Network},
	    school     =     {Chalmers University Of Technology},
	    year     =     {2016},
	    }

Reach out to egil.martinsson[at]gmail.com if you have any questions. 
If you can, please open an issue so other can join in. 
Contributions/PR/Comments etc are very welcome!
