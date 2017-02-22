# GenerativeControl

Code for a generative controller for the AI Gym cartpole task.

The code in the root directory of this repository is set up to train sets of 
controllers on the AI Gym cartpole-v1 task in order to collect statistics about their learning
and performane. It generates a set of 10 runs, 125 episodes in length,
starting from a random seed given by the first commandline argument.

The task_transfer/ subdirectory contains a version of the code that
trains a single model and saves the weights, along with a second
program which loads the model and evaluates its performance for
different reward functions.