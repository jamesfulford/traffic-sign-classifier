# **Traffic Sign Recognition**

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

[training_data_distribution]: ./artifacts/training_data_distribution.jpg "Training Data Distribution"
[validation_data_distribution]: ./artifacts/validation_data_distribution.jpg "Validation Data Distribution"
[test_data_distribution]: ./artifacts/test_data_distribution.jpg "Test Data Distribution"
[sample_image0]: ./sample_images/00000.jpg "Traffic Sign 0"
[sample_image1]: ./sample_images/00001.jpg "Traffic Sign 1"
[sample_image2]: ./sample_images/00002.jpg "Traffic Sign 2"
[sample_image3]: ./sample_images/00003.jpg "Traffic Sign 3"
[sample_image4]: ./sample_images/00004.jpg "Traffic Sign 4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Here is a link to my [project code](https://github.com/jamesfulford/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the labels are distributed. We can tell that classes 2 (50km/h speed limit), 1 (30km/h speed limit), and 13 (yield) (respectively) are the most common labels.

Training:
![Label distribution for training data][training_data_distribution]

The distribution is similar across training and test data, which is a good sign.

Validation:
![Label distribution for validation data][validation_data_distribution]

Test:
![Label distribution for validation data][validation_data_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I normalized the pixels of each image to be within (-1, 1) to improve algorithm performance. I did not go beyond this, as I thought grayscale might prove to be too damaging to results.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					|		Description								|
|:---------------------:|:---------------------------------------------:|
| Input					| 32x32x3 RGB image								|
| Layer 1				|:---------------------------------------------:|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling			| 2x2 area, 2x2 stride, outputs 14x14x10 		|
| Layer 2				|:---------------------------------------------:|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 area, 2x2 stride, outputs 5x5x16			|
| Flatten				| 5 * 5 * 16 = 400 nodes						|
| Layer 3				|:---------------------------------------------:|
| Fully connected		| 400 nodes in, 190 nodes out (52.5% reduction)	|
| Layer 4				|:---------------------------------------------:|
| Fully connected		| 190 nodes in, 90 nodes out (47.4% reduction)	|
| Layer 5				|:---------------------------------------------:|
| Fully connected		| 90 nodes in, 43 nodes out (47.8% reduction)	|
|:---------------------:|:---------------------------------------------:|

As you might have noticed, in layers 3-5, I tried to keep the reduction ratio roughly proportional. Ad-hoc trial-and-error showed this to be effective.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer over a maximum of 40 epochs (I saved every model that hit over .93 accuracy on the validation set, then manually picked the model with the highest validation set accuracy.) My machine could handle a batch size of 512 (didn't dare to go higher for some reason) and a lower learning rate of 0.001. When it came to dropout, I used a 0.5 keep/toss probability for training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 95%
* test set accuracy of 92.7%

(Looks like it didn't generalize all that well, despite the label distributions being similar. This looks like overfitting.)

* What was the first architecture that was tried and why was it chosen?

I started with the LeNet architecture. The general approach of using convolutional layers followed by fully-connected layers makes sense for determining traffic signs or written numbers alike, regardless of location in the image. In this sense, the two problems were similar enough that using the same architecture seemed viable.

* What were some problems with the initial architecture?

The LeNet architecture was built for a 32x32x1 input and 10 outputs. My input was 3 layers deep instead of 1, and my outputs were 43 in number, so I had to make those adjustments for the model to even run.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Considering that my input is 32x32x3 (three times larger), it made more sense to make changes to introduce more weights to account for the variances in data. So I added more depth to the first convolutional layer and added some more nodes in the final fully-connected layers, but the adjustment was not proportional to the increase in input to help with training (more weights implies more time and stochasticity, which is not helpful for ad-hoc model/parameter experimentation). I did not end up adding more layers, but was considering adding another fully-connected layer if I could reduce the output of the convolutional layers.

I added dropout to the fully connected layers to reduce overfitting. I presume it helped, but I still had a ways to go.

* Which parameters were tuned? How were they adjusted and why?

I tried tweaking the standard deviation of initial values for weights. That did not go well. Also tried reducing the probability of dropping out values, but eventually reverted back to 0.5.

Keeping an eye on how quickly validation accuracy converted over epochs, I either raised or lowered the epoch level. I figured it generally didn't hurt to have higher epochs if I had patience. Higher epoch counts are also needed to allow smaller learning rates (such as mine) to be properly applied during training.

What I found frustrating about this step is the time it takes to train the model over a couple epochs to find out if there was a noticeable impact. After doing this several times (at 5 epochs), I found that unlike before I can get considerably different results (+/- 5% validation accuracy at times). The non-determinism made it frustrating to determine if my change was a positive impact or just a statistical fluke.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A convolutional layer is good for several reasons for image processing in general. First, it reduces the number of weights to train compared to straight fully-connected networks. Second, it introduces a certain degree of disregard for the location of the data in the image. Considering that a sign is exactly the same regardless of its location in the image, the latter is an important property/feature to have.

A dropout layer is important to reduce overfitting. It teaches the neural network not to overly rely on any weights/nodes/connections during training, as they may be dropped at any time. I found this improved accuracy on the validation set, though lower values (0.3, for example) were of particular detriment.

If a well known architecture was chosen:
* What architecture was chosen?

(See first answer in this section) I started with the LeNet architecture.

* Why did you believe it would be relevant to the traffic sign application?

(See first answer in this section) The general approach of using convolutional layers followed by fully-connected layers makes sense for determining traffic signs or written numbers alike, regardless of location in the image. In this sense, the two problems were similar enough that using the same architecture seemed viable.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

While the accuracy went down from training to validation and from validation to test, I believe that since the drop in accuracy across these groups was not a large drop-off (all still over 90%) that the model is working reasonably well for my first ML model. It could have overfit much more than it did here.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

In general, the classifier correctly classified images it saw more frequently during training. This is not surprising.

Image 0: Class 11 (Right-of-way at the next intersection)
This image is somewhat less common (14th most common at 1170 instances in training data), so the classifier may not be familiar with the sign enough to classify it properly. Also, the background has some geometric and sharp differences which may resemble a more square sign, which may muddy the results enough.

![alt text][sample_image0]

Image 1: Class 40 (Roundabout mandatory)
This image is very uncommon (only 300 instances during training). Geometric shapes in background, plus white splotch in the middle of the sign may make it difficult.

![alt text][sample_image1]

Image 2: Class 39 (Keep left)
This image is rare in the training set (only 270 instances during training). Besides a light shading, this image is not otherwise difficult.

![alt text][sample_image2]

Image 3: Class 4 (Speed limit (70km/h))
This is a common image during training (7th most at 1770 instances), however two other signs can be partially seen at the top and bottom of the image.

![alt text][sample_image3]

Image 4: Class 21 (Double curve)
This is an uncommon image during training (270 instances) with a little bit of a sign peeking in the bottom of the image. Light shading may also be a difficulty.

![alt text][sample_image4]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

|	Image				|	Prediction									|
|:---------------------:|:---------------------------------------------:|
| Right-of-way			| Speed limit (30km/h)							|
| Roundabout mandatory	| Turn left ahead								|
| Keep left				| Keep left										|
| Speed limit (70km/h)	| Speed limit (70km/h)							|
| Double curve			| Double curve									|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The test set had an accuracy of 92%, which would have meant 4 or 5 correct guesses. However, my 5 images did not match the same distribution; 4 of the 5 images were relatively uncommon, so the classifier was less prepared to classify them all properly.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Oddly enough, my model is 100% confident in each prediction (others are on the order of 10^-28 or something similarly crazy). This does not seem right, but I cannot see what I'm doing wrong in my top_k tensor calculations, if anything.
