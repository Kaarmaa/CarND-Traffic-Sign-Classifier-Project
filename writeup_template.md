#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./datadist.png "Visualization"
[image2]: ./grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./data/ExtendedExamples/1.jpg "Traffic Sign 1"
[image5]: ./data/ExtendedExamples/2.jpg "Traffic Sign 2"
[image6]: ./data/ExtendedExamples/3.jpg "Traffic Sign 3"
[image7]: ./data/ExtendedExamples/4.jpg "Traffic Sign 4"
[image8]: ./data/ExtendedExamples/5.jpg "Traffic Sign 5"
[image9]: ./extendedset.png "Traffic Sign 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Kaarmaa/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

[Notebook: Cell 1]

I used the pandas/numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43



####2. Include an exploratory visualization of the dataset.

[Notebook: Cell 2-4]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the training, validation, and test sets.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

[Notebook: Cell 5]
As a first step, I decided to convert the images to grayscale because my first architecture that maintained color channels was not as accurate as the grayscale implementation. I ran into issues using the normal RGB2GRAY color conversion (not just the R / B channel swap), as the images were not converting correctly. This ultimately led me to use RGB2YUV and maintain luminance for the image.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data as suggested, to center the values at 0, with a range of +/- 1, in floating point.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

[Notebook: Cell 6]
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 Grayscale image   							| 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x32 				|
| Fully connected		     | outputs 128         									|
| RELU					             |												|
| Dropout					          |	prob 0.5											|
| Fully connected		     | outputs 64         									|
| RELU					             |												|
| Dropout					          |	prob 0.5											|
| Fully connected		     | outputs 43         									|
| Softmax				           |         									|
| One Hot				           | Final Classified Output        									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

[Notebook: Cell 7]
To train the model, I used an Adam optimizer, with a 128 batch and 20 Epochs. The learning rate was left at the standard 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

[Notebook: Cell 7]

My first attempts at the network architecture were a bit overkill. It turned out to be much deeper AND much wider than was actually necessary. After I realized this I began to iteratively take both the width and depth down for layers until I started to get a better fit with the data. As it is, the solution appears to be an overfit, and it should be toned down even further.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.970
* test set accuracy of 0.945

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 I honestly don't remember the full dimensions, but the connected layers were something on the order of 1000 and 500, which was great overkill, just in the interest of investigating what would happen.
* What were some problems with the initial architecture?
 Processing time was drastically hard to debug, even with high power GPUs on the local PC.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 To fix the overfit issue, the network was scaled down, as well as dropout functions being added to the fully connected layers.
* Which parameters were tuned? How were they adjusted and why?
 I messed around with batch size and epochs the most. Batch size at around 128 tended to yeld the best results, and on most runs, the validation accuracy stabilized between 15 and 20 epochs, so I left it at 20.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
 The convolutional layers are necessary to extract features for each sign. 
 The added dropout layers were an attempt to squash any overfit on the dataset. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image4] ![image5] ![image6] 
![image7] ![image8]

I picked all signs that are round, and have either similar shapes or colors, in an attempt to get the network to fail. I was, unfotunately, successful at this failure...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h	      		| Stop sign   									| 
| Ahead Only     			| Left Turn ahead 										|
| No Entry					| No Entry											|
| 30 km/h	      		| Roundabout mandatory					 				|
| End of all Speed...			| End of all Speed...      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares poorly to the accuracy on the test set. I believe this is due to overfitting the data.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

[Notebook: Cell 15]
![alt text][image9]

Correct or not, the net was always fairly certain of what classification was chosen, with the exception of the 30mph sign being misclassified as a roundabout. Looking back at the data, this sign has one of the highest occurance rates within the set, making me think that the data was overfit to this set, and the new sign was not able to be classified properly.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .89         			| Stop sign   									| 
| .99     				| Left Turn ahead 										|
| 1.00					| No Entry											|
| .27	      			| Roundabout mandatory					 				|
| .97				    | End of all Speed...      							|

