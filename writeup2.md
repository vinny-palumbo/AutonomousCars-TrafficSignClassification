# **Traffic Sign Classification** 
---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./writeup-images/training-set-sample.png "Training set sample"
[image2]: ./writeup-images/datasets-distribution.png "Datasets distribution"
[image3]: ./writeup-images/training-set-grayscale-sample.png "Grayscaling"
[image4]: ./writeup-images/accuracies-by-epoch.png "Accuracies throughout the epochs"
[image5]: ./writeup-images/images-from-web.png "Images from the web"

---
### Data Set Summary & Exploration

#### 1.Basic summary of the data set 

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

The code for this step is contained in the third and fourth code cells of the IPython notebook.  

Here is a sample of 5 traffic sign images from the training set

![alt text][image1]

Here is the distribution of the number of each traffic sign type in the training(blue), validation(green) and test(red) datasets

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data

The code for this step is contained in the fifth and sixth code cells of the IPython notebook.

I decided to convert the images to grayscale because the colors in the background (blue sky, green tree, etc.) should not be a factor in our model. Also, there is a strong correlation between the shape of a traffic sign and its colors (circle with a border: white and red, circle without a border: blue, triangle: white and red, diamond: white and yellow, etc.)

Here is a sample of 5 traffic sign images from the training set after grayscaling

![alt text][image3]

I could have used other preprocessing techniques to adjust the brightness of the images that were taken in darker light conditions.

#### 2. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 8th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description            					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU					|												|
| Dropout				| 75%											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x24	|
| RELU					|												|
| Dropout				| 75%											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24 					|
| Flatten				| outputs 600        							|
| Fully connected		| outputs 200        							|
| RELU					|												|
| Dropout				| 75%											|
| Fully connected		| outputs 100        							|
| RELU					|												|
| Dropout				| 75%											|
| Fully connected		| outputs 43        							|
| Softmax				|         										|


#### 3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the tenth cell of the ipython notebook. 

To train the model, I used the Adam Optimizer, which is know to have several advantages regarding the learning rate, over the more simple tf.train.GradientDescentOptimizer.

As for the hyperparameters, I've used a batch size of 128 so that the model trains faster and uses less memory, although it looses some accuracy in estimating the gradient. I've increased the number of epochs from 10 to 101, which increased the validation accuracy from 89% to 95%. Then, I decreased the learning rate by a factor of 3, from 0.001 to 0.0003, which increased the validation accuracy to 96%. Finally, I increased the filter depths of the convolutional layers, which increased the validation accuracy to 97%.

#### 4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th, 12th and 14th cells of the Ipython notebook.

My final model results are:
* training set accuracy of 100%
* validation set accuracy of 97.3%
* test set accuracy of 95.5%

Here is the evolution of the Training and Validation accuracies throughout the epochs:

![alt text][image4]

I've chosen the well-known architecture of LeNet-5 for this traffic sign classification problem. Since it's a convolutional neural network, "it recognizes visual patterns directly from pixel images with minimal preprocessing.  It can recognize patterns with extreme variability (such as handwritten characters), and with robustness to distortions and simple geometric transformations"(Yann LeCun). Moreover, convolutional neural networks are good for things that have statistical invariance (eg: a stop sign in the top-right of an image or a stop sign in the bottom-left of an image should both be classified as stop signs, so their position in the image is not meaningful). 

To make the model more robust and avoid overfitting the training set, I've added dropouts to the fully-connected layers first, and then also to the convolutional layers. I started with dropouts of 50%, but it was taking too long to train so I then set the dropouts to 75%. This gave me a validation accuracy of 91% after 10 epochs. I then increased the number of epochs to 101, and that improved the validation accuracy to 95%. I also decreased the learning rate by a factor of 3, from 0.001 to 0.0003, to optimize the loss minimisation, and that increased the validation accuracy to 96%. Finally, to make the model able to capture more complex patterns, I increased the depth of the first convolutional layer's filter from 6 to 12, and from 16 to 24 for the second convolutional layer's filter. This increased the validation accuracy to 97%.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

Qualitatively, all 5 images are pretty standard and should not have any issue being classified. There's just a small light-gray rectangle on the "Go straight or left" traffic sign (the second image), which is not ideal but shouldn't confuse the algorithm all that much since its edges are not that opaque. 

Quantitavely, the "Roundabout mandatory" traffic sign (the last image) could be difficult to classify because there are not a lot of images of that traffic sign type in the training set. Ideally, we would need to stabilize the distribution of our training set so that it is not biased towards the more numerous traffic sign types.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| No passing   									| 
| Go straight or left	| Go straight or left							|
| Speed limit (60km/h)	| Speed limit (60km/h)							|
| Yield 				| Yield							 				|
| Roundabout mandatory	| Roundabout mandatory 							|


The model was able to correctly guess all 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.5%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for outputting the top 5 softmax probabilities of each image is located in the 19th cell of the Ipython notebook.

For the first image, the model is sure that this is a "No passing" sign (probability of 1.0), and the image does contain a "No passing" sign. 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No passing   									| 
| 0.0     				| No passing for vehicles over 3.5 metric tons	|
| 0.0					| Ahead only									|
| 0.0      				| No entry					 					|
| 0.0					| Vehicles over 3.5 metric tons prohibited		|


For the second image, the model is sure that this is a "Go straight or left" sign (probability of 0.99999), and the image does contain a "Go straight or left" sign. 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999      			| Go straight or left							| 
| 1e-05    				| Stop  										|
| 0.0					| No passing for vehicles over 3.5 metric tons	|
| 0.0      				| Keep right 					 				|
| 0.0					| Roundabout mandatory 							|


For the third image, the model is sure that this is a "Speed limit (60km/h)" sign (probability of 0.99999), and the image does contain a "Speed limit (60km/h)" sign. 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999      			| Speed limit (60km/h)							| 
| 0.0001   				| Speed limit (80km/h)							|
| 0.0					| Speed limit (50km/h)							|
| 0.0      				| Speed limit (30km/h)			 				|
| 0.0					| No passing for vehicles over 3.5 metric tons	|


For the fourth image, the model is sure that this is a "Yield" sign (probability of 1.0), and the image does contain a "Yield" sign. 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield    										| 
| 0.0     				| Speed limit (20km/h)							|
| 0.0					| Speed limit (30km/h)							|
| 0.0      				| Speed limit (50km/h)			 				|
| 0.0					| Speed limit (60km/h) 							|


For the fifth image, the model is slightly more sure that this is a "Roundabout mandatory" sign (probability of 0.35193) rather than a "Keep right" sign (probability of 0.34193), and the image does contain a "Roundabout mandatory" sign. This "confusion" can be explained by the fact that there are about 8 times more images of "Keep right" sign in the training set than images of "Roundabout mandatory", and the model is biased towards the traffic sign types that are more numerous in the training set. Furthermore, one of the 3 arrows that form the "Roundabout mandatory" sign points right and is thus detected to be part of a "Keep right" sign.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.35193     			| Roundabout mandatory							| 
| 0.34193  				| Keep right									|
| 0.2165				| Slippery road									|
| 0.02638  				| Beware of ice/snow			 				|
| 0.02162				| Children crossing   							|
