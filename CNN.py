# -*- coding: utf-8 -*-
"""
Created on Mon May 15 21:21:22 2017

@author: JianyuChen
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST/', one_hot=True)


print("-Training-set:\t\t{}".format(len(data.train.labels)))
print("-Test-set:\t\t{}".format(len(data.test.labels)))
print("-Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls=np.argmax(data.test.labels,axis=1)

# We know that MNIST images are 28 pixels in each dimension.
img_size=28
# Images are stored in one-dimensional arrays of this length.
img_size_flat=img_size*img_size
# Tuple with height and width of images used to reshape arrays.
img_shape=(img_size,img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels=1
# Number of classes, one class for each of 10 digits.
num_classes=10


def plot_images(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9
    
    #Create figure with 3x3 sub-plots.
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat):
        #plot image
        ax.imshow(images[i].reshape(img_shape),cmap='binary')
    
        #Show true and predicted classes.
        if cls_pred is None:
            xlabel="True:{0}".format(cls_true[i])
        else:
            xlabel="True:{0},Pred:{1}".format(cls_true[i],cls_pred[i])

            
        #Show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show
    
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
    
def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))
    
def new_conv_layer(input,num_input_channels,filter_size,
                   num_filters,use_pooling=True):
    
    shape=[filter_size,filter_size,num_input_channels,num_filters]

    weights=new_weights(shape=shape)
    biases=new_biases(length=num_filters)
    
    layer=tf.nn.conv2d(input=input,
                       filter=weights,
                       strides=[1,1,1,1],
                        padding='SAME')
    
    layer += biases
    
    if use_pooling:
        layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                            padding='SAME')
        
    layer=tf.nn.relu(layer)
    
    return layer,weights
    
def flatten_layer(layer):
    layer_shape=layer.get_shape()
    
    num_features=layer_shape[1:4].num_elements()
    
    layer_flat=tf.reshape(layer,[-1,num_features])
    
    return layer_flat,num_features
    
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
    
    
    
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
    
layer_flat, num_features = flatten_layer(layer_conv2)   
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session=tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size=64
total_iterations=0

def optimize(num_iterations):
    global total_iterations
    start_time=time.time()
    for i in range(total_iterations,
                   total_iterations+num_iterations):
        x_batch,y_true_batch=data.train.next_batch(train_batch_size)
        
        feed_dict_train={x:x_batch,
                         y_true:y_true_batch}
                         
        session.run(optimizer,feed_dict=feed_dict_train)
        
        if i%100 ==0:
            acc=session.run(accuracy,feed_dict=feed_dict_train)
            msg="Optimization Iteration:{0:>6},Training Accuracy:{1:>6.1%}"
            
            print(msg.format(i+1,acc))
            
    total_iterations=num_iterations
    end_time=time.time()
    
    time_dif=end_time-start_time
    
    print("Time usage:"+str(timedelta(seconds=int(round(time_dif)))))

    

def plot_example_errors(cls_pred,correct):
    incorrect=(correct==False)
    images=data.test.images[incorrect]
    cls_pred=cls_pred[incorrect]        
    cls_true=data.test.cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
  
def plot_confusion_matrix(cls_pred):
    cls_true=data.test.cls
    cm=confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks=np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    
    
test_batch_size=256
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    num_test=len(data.test.images)
    cls_pred=np.zeros(shape=num_test,dtype=np.int)
    
    i=0
    
    while i<num_test:
        j=min(i+test_batch_size,num_test)
        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
        
#随机分类准确度
#print_test_accuracy() 
#一次优化以后
#optimize(num_iterations=1)
#100次优化迭代之后
#optimize(num_iterations=99) # We already performed 1 iteration above
#optimize(num_iterations=9000) 
#print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)


#绘制卷积权重的函数
def plot_conv_weights(weights,input_channel=0):
    w=session.run(weights)
    w_min=np.min(w)
    w_max=np.max(w)
    num_filters=w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    
    fig, axes = plt.subplots(num_grids, num_grids)
    
    
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    