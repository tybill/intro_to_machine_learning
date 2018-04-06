<h1>Welcome to the Intro to Machine Learning Workshop for TrojanHacks Spring 2018</h1>

<h3>Please read the follwing instructions carefully before the workshop</h3>
<h2>Overview</h2>

This workshop will focus on deep learning. We will go over some basic concepts about deep learning and then build a classifier on the hand-written digit dataset (MNIST) with tensorflow.

<h2>Environment Setup</h2>

<p>An instruction on how to setting up the environment could be fond here <a href = "http://caisplusplus.usc.edu/blog/curriculum/environment_setup"> here </a>. Please note that you only have to install tensorflow, numpy and matplotlib for this workshop. Therefore, after completing step 7 in the instruction, simply skip step 8. In step 10, you do NOT need to run these two commands: <code class="language-bash">pip install pandas</code> and <code class="language-bash">pip install keras</code>. Pandas and Keras are also two very popular packages in machine learning. However, we will not cover these two libraries in this workshop.</p>

<p>After completing the above steps, you should be able to run the following lines of codes</p>
<code class="language-python">import matplotlib.pyplot as plt</code><br>
<code class="language-python">import numpy as np</code><br>
<br>
<code class="language-python">import tensorflow as tf</code><br>
<code class="language-python">mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)</code><br>
<code class="language-python">plt.imshow(mnist.train.images.reshape(mnist.train.labels.shape[0],28,28)[0].reshape(28,28), cmap='gray')</code><br>
<code class="language-python">plt.show()</code><br>

<p>This block of codes will load the mnist data set and will output a black and white image of the digit 7, which is the first image in the training set.</p>

<h2>Clone this repo into your own machine</h2>
<p>You should then clone this repo into your own machine. In this repo you can find two additional files <code class="highlighter-rouge">demo_incomplete.py</code> and <code class="highlighter-rouge">demo_incomplete.ipynb</code>. These are the skeleton code for the workshop. And we will be building on this to complete our classifier.</p>