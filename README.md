# WTTE-RNN
A hackless machine-learning framework for churn- and time to event prediction. Forecasting problems as diverse as server monitoring to earthquake- and churn-prediction can be posed as the problem of predicting the time to an event. **churn_watch** is a data-science philosophy on how this should be done. The core technology is the WTTE-RNN algorithm ([blog post](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/), [master thesis](https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf))

<blockquote class="imgur-embed-pub" lang="en" data-id="0oVdKiv"><a href="//imgur.com/0oVdKiv"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

# ROADMAP
The goal is to create a forkable and easily deployable model framework. WTTE-RNN is the algorithm, churn_watch is the deployment - an opinionated idea about how churn-monitoring and reporting can be made beautiful and easy. 

## Implementations of the objective functions
The core technology is the objective functions. These can be used with any machine-learning algorithm. To spread the word we should implement and commit them to various ML-projects. 

* Tensorflow (Done but not implemented as raw op yet)
* MXnet
* Theano
* Keras
* TORCH
* H2o
* scikitFlow
* MLlib

## Auxiliary

To use the model one needs basic tte-transforms of raw data. To consume the models we need weibull related functions for the final output.
* Ready to run helper functions implemented in SQL, R, Python
  - get_time_to_event (calculates tte and censored tte)
  - get_is_censored
  - weibull hazard, chf, cdf, pdf, quantile, expected value etc (Python done). 

## Monitoring 
The WTTE-RNN is as much an ML-algorithm as a visual language to talk about this shape of data and our predictions.
* Plots (partly done)
* Shiny webapp or/and similar (partly done elsewhere)
* Integration. Slack/E-mail bots & summaries
* API 

## Models
To get this going we need at least one off-the-shelf deep-learning implementation that scales. Currently there's one that doesn't.
* Best-practices WTTE-RNN implementation

## Deployment
* Notebooks
* Containers
* IBM project?
* Cortana project?

# Licensing
* MIT-license. 
* Contributors should be encouraged to spread the word and help others implement the model.
