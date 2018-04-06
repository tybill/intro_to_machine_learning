#importing libraries and loading in the data set
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#construct the training and testing sets
train_images = mnist.train.images.reshape(mnist.train.labels.shape[0],28,28)
train_labels = mnist.train.labels
test_images = mnist.test.images.reshape(mnist.test.labels.shape[0],28,28)
test_labels = mnist.test.labels

num_train = train_images.shape[0]

#Testing if loading is correct
print("Input dimensions: {}\nTesting Dimensions: {}".format(train_images.shape, test_images.shape[0]))

plt.imshow(train_images[0].reshape(28,28), cmap='gray')
plt.show()

#A function to compute the moving average
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class MNISTClassifier:
    
    ################### INIT: DO NOT CHANGE THIS PART OF THE CODE ###################
    def __init__(self, optimizer = 'GradientDescent'):
        self.learning_rate = 6e-2;
        self.build(optimizer)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    ################### END OF CODE #################################################
    
    ################### BUILDING THE NETWORK: TO COMPLETE ###########################
    def build(self, optimizer = 'GradientDescent'):
        #Placeholder for input and labels
        ############### TODO ########################################################

        
        ############### TODO ########################################################
        
        #Network Structures / Info:
        #Hyperparameter for x -> flatt -> fc_1 -> y_hat:
        #Learning rate: arround 6e-2
        #flatt -> fc_1: 728 -> 256 with relu activation
        #fc_1  -> y_hat: 256 -> 10
        #
        #Test Accuracy after 50 epoches: ~85%

        #Initiate Network Architectures
        ############### TODO ########################################################
        
        
        
        
        ############### TODO ########################################################
        
        #Initiate Loss Functions
        #Softmax with cross entropy loss
        ############### TODO ########################################################
        
        ############### TODO ########################################################
        
        #Initiate Optimizer
        #Gradient Descent or Adam Optimizer
        ############### TODO ########################################################
        
        
        
        
        ############### TODO ########################################################
        
        #Compute the accuracy
        ############### TODO ########################################################
        
        
        ############### TODO ########################################################
    ################### END OF CODE #################################################
    
    ################### TRAIN THE NETWORK: TO COMPLETE ##############################
    def train(self, x, y, num_epoch, batch_size):
        #Data Preprosessing
        ############### TODO ########################################################
        
        ############### TODO ########################################################
        
        losses = []
        accuracies = []
        for epoch in range(num_epoch):
            loss = 0
            accuracy = 0
            for iter in range(x.shape[0]//batch_size):
            #Do one iteration
            ########### TODO ########################################################
                
            ########### TODO ########################################################
            print("Epoch: {}\t| Loss: {}".format(epoch + 1, loss))
        
        #Plot the loss and accuracy for training
        plt.plot(running_mean(losses, 100))
        plt.title('training loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()    
        
        plt.plot(running_mean(accuracies, 100))
        plt.title('training accuracy')
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.show()
    ################### END OF CODE #################################################
    
    ################### TEST THE NETWORK: TO COMPLETE ###############################
    def test(self, x, y):
        #Data Preprosessing
        ############### TODO ########################################################
        
        ############### TODO ########################################################
        
        #Test
        ############### TODO ########################################################
        
        ############### TODO ########################################################
    ################### END OF CODE #################################################
    
    ################### DO ONE STEP: TO COMPLETE ####################################
    def run_single_step(self, x, sess):
        #Return 
        ############### TODO ########################################################
        
        ############### TODO ########################################################
    ################### END OF CODE #################################################
    
    ################### CLOSE SESSION: DO NOT CHANGE THIS PART OF THE CODE ##########
    def closeSession(self):
        self.sess.close()
        self.sess = None
    ################### END OF CODE #################################################

#Now lets train our network
####################### TODO ########################################################




####################### TODO ########################################################

