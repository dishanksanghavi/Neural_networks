# Sanghavi, Dishank Himasnhu
# 1001-761-070
# 2020_10_12
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        if not self.weights:
            new_Weights = tf.Variable(np.random.randn(self.input_dimension, num_nodes), trainable=True)
        else:
            new_Weights = tf.Variable(np.random.randn(self.weights[-1].shape[1], num_nodes), trainable=True)
         
        new_Bias = tf.Variable(np.random.randn(num_nodes,), trainable=True)
        

        self.weights.append(new_Weights)
        self.biases.append(new_Bias)
        self.activations.append(transfer_function)
        

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]
    
    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number]=weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number]=biases
        
    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        y_hat = tf.Variable(X)
        
        def activate(add_bias, y_hat):
            y_hat = y_hat.lower()
            if y_hat == 'sigmoid':
                return tf.nn.sigmoid(add_bias)
            elif y_hat == 'linear':
                return add_bias
            elif y_hat == 'relu':
                return tf.nn.relu(add_bias)
        
        for layer in range(len(self.weights)):
        
            weight_input = tf.matmul(y_hat, self.get_weights_without_biases(layer))
            add_bias = tf.add(weight_input, self.get_biases(layer))
            y_hat = activate(add_bias,self.activations[layer])
            #print(y_hat)
        return y_hat

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch o f data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        input_value=tf.data.Dataset.from_tensor_slices((X_train, y_train))
        input_value=input_value.batch(batch_size)
        for epoch in range(num_epochs):
            for count_step,(x,y) in enumerate(input_value):
                #print(x)
                with tf.GradientTape(persistent=True) as tape:
                    predict_y=self.predict(x)
                    loss=self.calculate_loss(y,predict_y)
                    for weight_value in range(len(self.weights)):
                        partial_loss_w,partial_loss_b=tape.gradient(loss,[self.weights[weight_value],self.biases[weight_value]])
                        self.weights[weight_value].assign_sub(alpha*partial_loss_w)
                        self.biases[weight_value].assign_sub(alpha*partial_loss_b)

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        new_matrix = self.predict(X)
        final=[]
        for i in new_matrix.numpy():  
            final.append(np.argmax(i))
        percent_error=round(1 - accuracy_score(list(y),final),3)
        return percent_error
        

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_c lasses].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        final=[]
        new_matrix=self.predict(X)
        for i in new_matrix.numpy():  
            final.append(np.argmax(i))
        confusion=tf.math.confusion_matrix(y,final)
        return confusion
