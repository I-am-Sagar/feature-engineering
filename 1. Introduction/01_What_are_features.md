## Chapter 1 - Introduction to Feature Engineering

Before we start this course, let us take a moment to appreciate the variety of information computers can comprehend today due to machine learning and artificial intelligence. We have systems that forecast weather, recommend songs and movies, understand our language and chat with us, provide suggestions, recognize faces and apply filters, enhance images, detect and cancel audio noise, and much more. _How has this been made possible?_ 

We all know computers only understand 1's and 0's. All types of information are eventually stored as bytes, on which we can perform arithmetic and logical operations. Computers do not know what language is, what images are, etc. These are __abstract__ ideas for the computer. 

For the computer, an image of 'cat' and of 'dog' is a bunch of 1's and 0's. The computer doesn't have the visual ability to identify what is in that image. Yet with the help of machine learning, we can identify different objects in images. Take another example of human language. When you ask, _What's the weather today like?_ to an AI assistant like Siri, Google Assistant, Alexa, etc., that question is a string input. A string is a sequence of characters and each character has a fixed unicode encoding of 2 bytes. So, for the computer, that sentence is just a sequence of bytes. _How can the computer understand the 'meaning' of the string?_ 

Let's try to understand this in the next section, where we discuss how a typical machine learning approach looks like. 

<hr>

### 1.1 - Machine Learning Approach

A typical machine learning approach looks like this. 

<img src='img/ML Approach.png' width="720">

**Step 1 - Define the problem statement**
We start by defining what we want our program to do. _What is the problem that we are trying to solve?_ On a very high level, problem statements may look like this:
1. I will give a bunch of images to the program, some of the cats and others of dogs. I want the program to segregate them into their respective folders. 
2. I will give the program a list of customer reviews of a particular product. The program should tell if each review is positive or negative. Then I want to understand the overall sentiment of the people regarding that product. 
3. I will give the past price data of a particular stock to the program and would wish to get the probability with which the stock price can go up or down tomorrow. 

**Step 2 - Collect and analyze the raw data**
I do not want to explicitly tell the program how to solve the problem. I wish the program to learn from the provided data and then produce the output accordingly based on what it learnt. I want to tell the computer in the learning/training phase that - these images are of the dogs and these are of the cats. Then I will show the computer an image and ask if it is a dog or a cat. So to train the program in the first place, we need _the raw data_. 

I am in the business of machine learning with you, only and only if you have some good raw data. 🙂  

1. For our *image classification problem*, we will need images of cats and dogs! 
2. For our *sentiment analysis problem*, we will need a list of customer reviews, along with the label - what you consider as positive and what as negative. 
3. For our *stock prediction problem*, we will need the past price data of the stock.  

_Who will collect this data?_ Well, there are teams in the corporate world that do this job. Gathering the _required data_ is a very time-consuming and expensive process. 

After the raw data is collected, we *filter and pre-process* it, so that our program is unimpacted by unnecessary variability. This filtered and pre-processed data is called **the cleaned data.** 

1. The images of cats and dogs should all be brought to the same resolution and file format. The number of pixels in the image then won't disturb our program. Either all should be coloured images or all should be in grayscale (black and white). Images in which the cat or dog is blurred or for whatever reason is not visible should be discarded. 
2. We should ensure no negative review is in the set with the positive ones and vice-versa. Too long and too short reviews should be removed.  
3. Usually, the financial data is the cleanest data one can get because it is already public and the processes are established to ensure its quality. 

**Step 3 - Come up with features**
Just look at the variety of data types we can have. 

1. Here, we have input as *Images*. 
2. Here, we are dealing with the *Text*. 
3. Stock Data is in *Numerical Format*. 

Analyzing numerical data is not that difficult because we can perform mathematical operations on them. The real issue arises with other data types like images, text, etc. So we need to somehow convert these data types that involve abstract ideas like language, visuals, audio, etc., to vectors on which we can train the computer using machine learning algorithms. 

A _mathematical model_ of data describes the relationships between different aspects of the data. For instance, a model that predicts stock prices might be a formula that maps a company’s earning history, past stock prices, and industry to the predicted stock price. A model that recommends music might measure the similarity between users (based on their listening habits) and recommend the same artists to users who have listened to the same songs.

Mathematical formulas relate numeric quantities to each other. But raw data is often not numeric. (The statement 'Alice bought _The Lord of the Rings trilogy_ on Wednesday' is not numeric, and neither is the review that she subsequently writes about the book.) There must be a piece that connects the two together. This is where features come in. 

A *feature* is a numeric representation of raw data. 

These features are fed to machine learning algorithms to make models. In this way, the algorithm is independent of the data type that passes as the input. It no longer matters if we want to classify text or images as long as we are able to convert them into a numeric representation. 

<img src="img/What is a feature.png" width="480"> 

In this approach, the accuracy of the machine learning model is highly dependent on the representation power of the features! If the numerical representation fails to capture the information of the raw data, the algorithm will not perform good. The number of features is also important. If there are not enough informative features, then the model will fail to perform the ultimate task. If there are too many features, or if most of them are irrelevant, then the model will be more difficult and tricky to train. Something might go awry in the training process that would impact the model’s performance.

Coming up with the right features that most accurately represent the raw data forms the basis of *feature engineering*. Feature engineering is the process of formulating the most appropriate features given the data, the model, and the task.

In a machine learning workflow, we pick not only the model, but also the features. This is a double-jointed lever, and the choice of one affects the other. Good features make the subsequent modeling step easy and the resulting model more capable of completing the desired task. Bad features may require a much more complicated model to achieve the same level of performance.

One last thing which I would want to add - coming up with good features require a good amount of domain knowledge. For example, if one has a fair understanding of image processing, they can design more ingenious image features. Similarly, if one is a linguist, they can design better text features. 

In the rest of this book, we will cover a little domain knowledge on each and every data type, different kinds of features and discuss their pros and cons for different types of data and models. 

**Step 4 - Decide the algorithm to train the model**
Once we have decided the features, we analyse which model will suit best for them and will produce most accurate results. 

If we plan to perform classification task, what will I use? Will I use *Logistic Regression*, or *Support Vector Machine*? Or should I use *Bayes' Decision Theory* or *Decision Tree*? 

Any machine learning course will focus on these questions. Discussing these things is beyond the scope of this book. For this, you might have to read the sequel of this book. 

**Step 5 - Test and evaluate the model on unseen data**
We check if the model meets the expectations and performs in a proper manner on unforeseen data. If it is not performing up to the mark, we will have to understand why and come up with the exact failure points for our model. 

**Step 6 - Based on evaluation, come up with strategies on how can we improve the model performance**
Most of the time, the model will not peform as expected in the first attempt. We then need to understand what is wrong based on the model evaluation and devise improvement strategies. We might have to arrive at more informative and better feature representation, decide to use a different machine learning algorithm altogether, reduce or increase the number of features used to perform the task, etc. 

We keep repeating this process in a loop till we are satisfied with the peformance of the model. 

<hr>

This book focuses on converting the raw data to accurate features. This is the step on which a data scientist or a machine learning engineer would spend most of their time. The details of the machine learning algorithms will be discussed in the next book. 

Without further ado, lets get started!

<hr>