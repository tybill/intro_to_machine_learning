<h1>Welcome to the Intro to Machine Learning Workshop for TrojanHacks Spring 2018</h1>

<h3>Please read the follwing instructions carefully before the workshop</h3>
<h2>Overvieww</h2>

This workshop will focus on deep learning. We will go over some basic concepts about deep learning and then build a classifier on the hand-written digit dataset (MNIST) with tensorflow.

<h2>Environment Setup</h2>

<p>An instruction on how to setting up the environment could be fond here <a href = "http://caisplusplus.usc.edu/blog/curriculum/environment_setup"> here </a>. Please note that you only have to install tensorflow, numpy and matplotlib for this workshop. Therefore, after completing step 7 in the instruction, simply skip step 8. In step 10, you do not need to run these two commands: <code class="language-bash">pip install pandas</code> and <code class="language-bash">pip install keras</code>. Pandas and Keras are also two very popular packages in machine learning. However, we will not cover these two libraries in this workshop.</p>

<p>After completing the above steps, you should be able to run the following lines of codesL</p>
<code class="language-python">
import matplotlib.pyplot as plt<br/>
import numpy as np<br/>

import tensorflow as tf<br/>
from tensorflow.examples.tutorials.mnist import input_data<br/>
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)<br/>	
</code>