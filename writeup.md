#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.jpg "Histogram"
[image2]: ./examples/generated_data.jpg "Generated data"
[image3]: ./examples/normalized_data.jpg "Normalized data"
[image4]: ./web_data/sign_1.png "Traffic Sign 1"
[image5]: ./web_data/sign_2.png "Traffic Sign 2"
[image6]: ./web_data/sign_3.png "Traffic Sign 3"
[image7]: ./web_data/sign_4.png "Traffic Sign 4"
[image8]: ./web_data/sign_5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/wildhemp/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410 
* The size of test set is 12630 
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the labels. Clearly, it is very unbalanced.

Checking the validation data set shows a similar picture, so it's not a problem with splitting. My assumption is, that this shows the relative frequency of different traffic signs in the real world. Exploring the different labels a bit seems to confirm this assumption. E.g. speed limit of 20 is much less frequent than 30 or 50.

![alt text][image1]

###Design and Test a Model Architecture

My model is largely based on the [baseline paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

##### 1. Generate additional data
First I generate additional data, because even though the model was doing quite well on the validation set, even with droupout applied, the model quickly overfit to the training values. I suspected that additional data would make it harder for the model to ovefit and will result in better validation and so test accuracy. The 'Traffic Sign Recognition' paper also uses it to make the model more robust.

To generate the data I applied some combination of:

* random translation of [-2, 2] on both axis
* random rotation by [-10, 10] degrees
* random scaling of [-10, 10] percents
* random shearing
* random brightness of [0.5, 1.5] percents

Applying all of the transformations at once and generating the same amount of data made the validation accuracy converge very slowly, compared to the training accuracy. One reason or this could be, that the data generated this way was not very representative of the validation data (and hence probably the real world data either). So instead of applying all of them, I tried to do some reasonable combination of them. E.g. shearing and scaling/rotation probably doesn't make much sense to do together.

An example of the original as well as generated data. The first image is the original, the second is scaled a little bit and the third is somewhat rotated.

![alt text][image2]

##### 2. Apply local and global normalization on the data

Beside the necessary global normalization I decided to apply local normalization as well on the images. Local normalization helps recognition by making edges more pronounced.

To make the normalization, the image is first converted to YUV, then the Y and U layers are normalized with the same local normalization technique. V is only global-normalized, the normalization technique I used is not effective on V and it also didn't seem to add much for the model quality.

An example of the 3 channels of the original image and the same 3 channels of the normalized image.

![alt text][image3]

The difference between the original and the normalized dataset is, that the edges are a little bit more pronounced in the latter case.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x108 (100 on Y, 4-4 on U-V each) 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x108 
| Dropout				| 60%
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x200      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x200 
| Dropout				| 60%
| Fully connected		| Multi-scale features 10292 in, 100 out									|
| Fully connected		|  100 in - 50 out									|
| Fully connected		|  50 in - 43 out									|
| Softmax				| with cross-entropy							|
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used AdamOptimizer.
Hyperparameters:

* Learning rate: 0.001
* Epochs: 25
* Batch size: 512
* Dropout on convolutional layers: 70%
* Dropout on the fully connected layer: 60%

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.94
* validation set accuracy of 99.07
* test set accuracy of 97.40

The first architecture I tried was the LeNet. This got a validation accuracy of ~89%. 
The main problem with this architecture was the feature sizes.

Increasing the feature sizes to 64 and 128 for the first and second conv layer respectively increased the validation accuracy to above 90%. 

Observing the training accuracy beside the validation, it was clear, that the model overfitted to the training data, so I introduced dropouts to the fully connected layers. This increased the accuracy to ~93%.

After this I added dropout to the convolutional layers as well. This increased the accuracy to ~94%.

After this I read the baseline paper, and tried the same feature sizes. This increased the accuracy by ~1%. Tested some other values, like additional fully connected layer and larger feature sizes, but those didn't make an observable difference.

Then I started experimentng with pre-processing the data. I found a local normalization algorithm on the internet, and started using it. I also converted the image to YUV and applied it on the Y and U layers (the baseline paper applies it on the Y only, but I did see some improvement on the U layers as well, and this is a different normalization technique). Although this didn't change the accuracy by much, it made the convergence of the model more stable.

Going back to the baseline paper I decided to try the multi-scale architecture. This made a big difference by increasing the validation accuracy to ~97%.

After tuning the hyperparameters, I started adding jittered data. One problem I observed here was, that if I applied all the transformations at once to the images, the model started to overfit to the training set (i.e. the validation accuracy converged much slower). So instead I applied some one by one and also added some combination at the end. This was chosen quite arbitrarily, although I find it somewhat logical (e.g. rotation is less important than scaling or translation or shearing, or translation and scaling together is probably helpful as well). I also tried to balance the size of the training set and not generate more than 6 image per original image.

Training the fine tuned model on all the data resulted in the final validation accuracy of 98.xx% and testing accuracy of 97.xx%

There are a couple of things which could improve the accuracy of the model further:

* Using a better local normalization technique and a different one for all the YUV channels. E.g. the baseline paper uses a different and probably better approach.
* Doing a feature extraction and using the resulting convolutional weights as starting weights for the model.
* Generating more data per image
* Generating new images for every epochs instead of in advance
* Adding noise to the images (I experimented a bit with this, but didn't have good results, not clear what was the problem)
* Using different activation functions, not ReLu.
* Maybe experimenting a bit more with dropout, the generated data might behave a bit differently in this regard

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]


The first image might be difficult to classify because it's shady and the light comes from behind.

The second one might be hard, because the top is not fully visible and also it's a little bit from the side and below.

The third one might be difficult, because the lower part is a bit shady and it's also a bit from the side.

The fourth image might be hard, because it is not a solid sign, but more like a canvas, and it also has some wrinkles. 

The fifth image has a man figure on it, which might confuse the model.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      		| Slippery road   									| 
| No vechicles    			| No vechicles 										|
| Bicycles crossing					| Bicycles crossing											|
| Road work	      		| Road work					 				|
| No entry			| No entry      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.40% accuracy.

I tried to balance the images chosen in a way, so that the model seen something similar, but it still has some weirdness on it. Obviously it's not very hard to find images what the model won't recognize. E.g. partially covered images where for example some leaves from a tree are obstructing some part of the traffic sign will make the model fail more often.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a slippery road sign (probability of 0.6778), and it's right. 

Top 5 probabilities for b'Slippery road' sign:

| Probability		|     Prediction				|
|:---------------------:|:---------------------------------------------:| 
| 0.6778		|  b'Slippery road'                                  |
| 0.1354		|  b'Bicycles crossing'                              |
| 0.0780		|  b'No passing for vehicles over 3.5 metric tons'   |
| 0.0512		|  b'Wild animals crossing'                          |
| 0.0434		|  b'Road work'                                      |

For the second image, the model is 99% sure it's a No vechicles sign and it contains a no vechicles sign.

Top 5 probabilities for b'No vehicles' sign:

| Probability		|     Prediction				|
|:---------------------:|:---------------------------------------------:| 
| 0.9999		|  b'No vehicles'                                    |
| 0.0000		|  b'Speed limit (30km/h)'                           |
| 0.0000		|  b'Yield'                                          |
| 0.0000		|  b'Speed limit (100km/h)'                          |
| 0.0000		|  b'Speed limit (120km/h)'                          |

For the third sign, the model is relatively (probability of .7255) sure it's a bicycles crossing sign, and the image contains that sign. Interestingly it also give some probability (.2739) for it to be a children crossing sign. The images indeed have some similarities in that they both contain people figures.

Top 5 probabilities for b'Bicycles crossing' sign:

| Probability		|     Prediction				|
|:---------------------:|:---------------------------------------------:| 
| 0.7255		|  b'Bicycles crossing'                              |
| 0.2739		|  b'Children crossing'                              |
| 0.0006		|  b'Slippery road'                                  |
| 0.0000		|  b'Beware of ice/snow'                             |
| 0.0000		|  b'Right-of-way at the next intersection'          |

For the fourth image, the model is 100% sure it's a Road work sign and the image contains that. I find this somewhat surprising, as I would have thought it to be at least less certain.

Top 5 probabilities for b'Road work' sign:

| Probability		|     Prediction				|
|:---------------------:|:---------------------------------------------:| 
| 1.0000		|  b'Road work'                                      |
| 0.0000		|  b'Children crossing'                              |
| 0.0000		|  b'Road narrows on the right'                      |
| 0.0000		|  b'Dangerous curve to the right'                   |
| 0.0000		|  b'Bumpy road'                                     |

For the fifth image the model again is 100% sure it's a No entry sign and it's right. Even the man figure wasn't enough to make the model uncertain.

Top 5 probabilities for b'No entry' sign:

| Probability		|     Prediction				|
|:---------------------:|:---------------------------------------------:| 
| 1.0000		|  b'No entry'                                       |
| 0.0000		|  b'Stop'                                           |
| 0.0000		|  b'Priority road'                                  |
| 0.0000		|  b'No passing for vehicles over 3.5 metric tons'   |
| 0.0000		|  b'No passing'                                     |


The model is very certain in some of it's predictions. My guess is the reason for this is, that those signs are relatively easy to recognize, they don't have complicated figures on them. So from that point of view I might have chosen some signs which are too easy to recognize. It seems, the features which should have made the model harder to recognize these didn't necessarily do their jobs.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
