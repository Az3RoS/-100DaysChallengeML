# Machine Learning

Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed. 

**_"The ability to learn"_** - for a machine, seems pretty intersting, huh!

Let’s try to understand Machine Learning in layman terms. Consider you are trying to toss a paper to a dustbin.
After first attempt, you realize that you have put too much force in it. After second attempt, you realize you are closer to target but you need to increase your throw angle. What is happening here is basically after every throw we are learning something and improving the end result. We are programmed to learn from our experience.
This implies that the tasks in which machine learning is concerned offers a fundamentally operational definition rather than defining the field in cognitive terms.

Within the field of data analytics, machine learning is used to devise complex models and algorithms that allow researchers, data scientists, engineers, and analysts to “produce reliable, repeatable decisions and results” and uncover “hidden insights” through learning from historical relationships and trends in the data set(input).

### Basic Difference in Machine Learning and Traditional Programming?
* **Traditional Programming :** We feed in DATA (Input) + PROGRAM (logic), run it on machine and get output.
* **Machine Learning :** We feed in DATA(Input) + Output, run it on machine during training and the machine creates its own program(logic), which can be evaluated while testing.

Now, you might be wondering whether Data science, analytics, and machine learning are all the same. Well they have their own way of doing things but to simply draw a line between them, let us look at the infographic.

<p align="center">
  <img src="https://www.simplilearn.com/ice9/article_detailed_content_img/data-science-data-analytics-machine-learning.jpg" alt="Pic Courtesy of Simplilearn">
</p>

If you want to know more about Data Science and how it differs, there is a great [Blog from Simplilearn](https://www.simplilearn.com/data-science-vs-data-analytics-vs-machine-learning-article).

There is another concept that you need to clear up before going forward. AI and machine learning are often used interchangeably, especially in the realm of big data. But these aren’t the same thing, and it is important to understand how these can be applied differently.
Artificial intelligence is a broader concept than machine learning, which addresses the use of computers to mimic the cognitive functions of humans. When machines carry out tasks based on algorithms in an “intelligent” manner, that is AI. Machine learning is a subset of AI and focuses on the ability of machines to receive a set of data and learn for themselves, changing algorithms as they learn more about the information they are processing.

<p align="center">
  <img src="https://www.mytectra.com/media/wysiwyg/Blog/deep-learning.png">
</p>

Now dont dwell into Deep Learning just now. It is the advanced part of Machine Learning which we will learn at a later stage. But just know that it is quite interesting(or I should rather say it is **AWSUM**).

Fun Fact: Do you know that you use machine learning algorithm dozens of time without even knowing it. Applications of Machine Learning include:

* **Web Search Engine:** One of the reasons why search engines like google, bing etc work so well is because the system has learnt how to rank pages through a complex learning algorithm.
* **Photo tagging Applications:** Be it facebook or any other photo tagging application, the ability to tag friends makes it even more happening. It is all possible because of a face recognition algorithm that runs behind the application.
* **Spam Detector:** Our mail agent like Gmail or Hotmail does a lot of hard work for us in classifying the mails and moving the spam mails to spam folder. This is again achieved by a spam classifier running in the back end of mail application.

Some other interesting uses are -
* **Speech Recognition (Natural Language Processing):** Alexa, Cortana, Siri, Google and a lot of new voice assistants.
* **Computer Vision:** Facial Recognition, Pattern Recognition, Character Recognition Techniques.
* **Google’s Self Driving Car:** Well. You can imagine what drives it actually. 
* **Amazon’s Product Recommendations:** Ever wondered how Amazon always has a recommendation that just tempts you to lighten your wallet. Well, that’s a Machine Learning Algorithm(s) called “Recommender Systems” working in the backdrop.
* **Youtube/Netflix:** They work just as above!
* **Data Mining / Big Data:** Data Mining and Big Data are just manifestations of studying and learning from data at a larger scale and you’ll find Machine Learning lurking nearby.
* **Stock Market/Housing Finance/Real Estate:** In order to better assess the market, namely “Regression Techniques”, for things as mediocre as predicting the price of a House, to predicting and analyzing stock market trends.

<p align="center">
  <img src="https://www.edureka.co/blog/wp-content/uploads/2018/03/Use-Case-What-is-Machine-Learning-Edureka-528x285.png">
</p>
Well did that peak your interest? There's more to come. Just the tip of the iceberg.


Let us begin then.

### How does Machine Learning Work?
Machine Learning algorithm is trained using a training data set to create a model. When new input data is introduced to the ML algorithm, it makes a prediction on the basis of the model.

The prediction is evaluated for accuracy and if the accuracy is acceptable, the Machine Learning algorithm is deployed. If the accuracy is not acceptable, the Machine Learning algorithm is trained again and again with an augmented training data set.

This is a basic workflow model from Edureka, as there are many other steps involved. But this one will do just fine to begin with.

<p align="center">
  <img src="https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/03/How-Machine-Learning-Works-What-is-Machine-Learning-Edureka-1.gif">
</p>

### Types of Machine Learning
**Machine learning** is an umbrella term covering lots of algorithms. It is sub-categorized to three types:

* **Supervised Learning** – _"Train Me!"_
* **Unsupervised Learning** – _"I am self sufficient in learning_"
* **Reinforcement Learning** – _"My life My rules! (Hit & Trial)"_

<p align="center">
  <img src="https://miro.medium.com/max/700/0*QYxNNYh6W9jO1b_-.png">
</p>


### Supervised Learning
Let’s say you are a real estate agent. Your business is growing, so you hire a bunch of new trainee agents to help you out. But there’s a problem — you can glance at a house and have a pretty good idea of what a house is worth, but your trainees don’t have your experience so they don’t know how to price their houses.
To help your trainees (and maybe free yourself up for a vacation), you decide to write a little app that can estimate the value of a house in your area based on it’s size, neighborhood, etc, and what similar houses have sold for.
So you write down every time someone sells a house in your city for 3 months. For each house, you write down a bunch of details — number of bedrooms, size in square feet, neighborhood, etc. But most importantly, you write down the final sale price:

| Bedrooms | Sq. feet | Neighborhood   | Sale Price |
| -------- | -------- | -------------- | ----------:|
| 3        | 2000     | *Metropolis*   | $250,000   |
| 2        | 2000     | *Gotham*       | $350,000   |
| 2        | 2000     | *Star City*    | $150,000   |
| 1        | 2000     | *Central City* | $78,000    |
| 4        | 2000     | *Blüdhaven*    | $150,000   |

Using that training data, we want to create a program that can estimate how much any other house in your area is worth:

| Bedrooms | Sq. feet | Neighborhood   | Sale Price |
| -------- | -------- | -------------- | ----------:|
| 3        | 2000     | *SmallVille*   | ???        |

This is called supervised learning. You knew how much each house sold for, so in other words, you knew the answer to the problem and could work backwards from there to figure out the logic. You feed your training data about each house into your machine learning algorithm. The algorithm is trying to figure out what kind of math needs to be done to make the numbers work out.

In **Supervised Learning**, you are letting the computer work out that relationship for you. And once you know what math was required to solve this specific set of problems, you could answer to any other problem of the same type!

So to be more specific, _"Supervised learning is the Data mining task of inferring a function from **labeled** training data."_




### Unsupervised Learning
In the previous example with the real estate agent, what if you didn’t know the sale price for each house? Even if all you know is the size, location, etc of each house, it turns out you can still do some really cool stuff. This is called **Unsupervised learning**.

_"Unsupervised learning is a branch of machine learning that learns from test data that has not been **labeled, classified or categorized**."_

Unsupervised machine learning helps to uncover previously unknown patterns in data. Most of the data we have today is unlabelled data. So this field of study is growing by a bit every day. The main difference between the two types is that supervised learning is done using a ground truth, or in other words, we have prior knowledge of what the output values for our samples should be. Therefore, the goal of supervised learning is to learn a function that, given a sample of data and desired outputs, best approximates the relationship between input and output observable in the data. Unsupervised learning, on the other hand, does not have labeled outputs, so its goal is to infer the natural structure present within a set of data points.

But if we represent the same in a picture, how would it be? Let's see for ourselves!

<p align="center">
  <img src="https://dataaspirant.files.wordpress.com/2014/09/george-clooney5.png">
</p>




### Reinforcement Learning
Reinforcement Learning is a type of Machine Learning which allows machines and software agents to automatically determine the ideal behaviour within a specific context, in order to maximize its performance.

The agent learns to achieve a goal by interacting with its environment. The agent receives rewards by performing correctly and penalties for performing incorrectly. The agent learns without intervention from a human and its goal is to maximize the total reward.

Getting a bit confusing? Let’s put a robot mouse in a maze.
The easiest context in which to think about reinforcement learning is in games with a clear objective and a point system.
Say we’re playing a game where our mouse is seeking the ultimate reward of cheese at the end of the maze (+1000 points), or the lesser reward of water along the way (+10 points). Meanwhile, robo-mouse wants to avoid locations that deliver an electric shock (-100 points).

<p align="center">
  <img src="https://pbs.twimg.com/media/DVRkSAhVAAAlceR.jpg:large">
</p>

We will discuss in details later. But here are few examples till date -

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*-mnyrNMXcCoqAj5svDpoXg.png">
</p>


There is a playlist of videos from [Simplilearn](https://www.youtube.com/watch?v=ukzFI9rgwfU&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy). You can have a look at them. They have explained it in simple terms.

So that was all an overview of things to come. Next, we will learn about the Machine Learning workflow.
We will start with a basic process flow and evolve as we progress. 

<p align="center">
  <img src="https://miro.medium.com/max/700/1*NS19lSaAj91ZC8b9l1yV_A.png">
</p>

So for starters lets say, Machine learning comprises of the following steps,
* **Data Collection:** The very first and the most important step is to collect relevant data corresponding to our problem statement. Accurate data collection is essential to maintaining the integrity of our machine learning project. The data set can be collected from various sources such as a file, database, sensor and many other such sources. We can also use some free data sets which are present on the internet. [Kaggle](http://www.kaggle.com) and [UCI Machine learning Repository](https://archive.ics.uci.edu/ml/index.php) are the two most-used repositories for practicing Machine Learning models.

* **Data Pre-Processing:** The collected data cannot be used directly for performing the analysis process as there might be a lot of missing data, extremely large values, unorganized text data or noisy data. Taking care of all the inconsistencies, errors and missing data in our dataset and converting it to a small clean dataset is called as data pre-processing. Please note that the data is categorized as -**Numeric** (income, age), **Categorical** (gender, nationality) and **Ordinal** (low/medium/high).
These are some of the basic pre — processing techniques that can be used to convert raw data:
    * **Conversion of data:** As we know that Machine Learning models can only handle numeric features, hence categorical and ordinal data must  be somehow converted into numeric features.
    * **Ignoring the missing values:** Whenever we encounter missing data in the data set then we can remove the row or column of data depending on our need. This method is known to be efficient but it shouldn’t be performed if there are a lot of missing values in the dataset.
    * **Filling the missing values:** Whenever we encounter missing data in the data set then we can fill the missing data manually, most commonly the mean, median or highest frequency value is used.
    * **Outliers detection:** There are some error data that might be present in our data set that deviates drastically from other observations in a data set. (Example: human weight = 800 Kg; due to mistyping of extra 0)
    
Here I would like to touch base on another important concept - **Feature Engineering**. The cleaned and processed data might not always provide you with the proper features needed to produce the desired output. Feature engineering is about creating new input features from your existing ones. Example - Let's say we already had a feature called **'num_schools'**, i.e. the number of schools within 5 miles of a property and **'median_school'**, i.e. the median quality score of those schools. However, we might suspect that what's really important is having many school options, but only if they are good. Well, to capture that interaction, we could simple create a new feature -
     **'school_score' = 'num_schools' x 'median_school'** 
     
According to a survey in [Forbes](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/#2ef9b7406f63), data scientists spend **80%** of their time on data preparation:
<p align="center">
  <img src="https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fgilpress%2Ffiles%2F2016%2F03%2FTime-1200x511.jpg">
</p>

The above mentioned steps data collection and pre-processing fall under the umbrella of **Exploratory Data Analysis(EDA)**. We will dwell into its details in due time. However, just know that EDA is the process of using statistical tools(such as graphs, measures of center and variatian) to investigate datasets in order to understand their important characteristics.
     
* **Model Training:** The pre-processed data is first divided into mainly two parts i.e. training and testing datasets in the train/test ratio of usually 70/30 or 80/20 for smaller datasets. Model training is the process by which a machine learning algorithm takes insights from the training dataset and learns specific parameters over the training period that will minimize the loss or how bad it performs on the training dataset. You can also train the classifier using training data set, tune the parameters using validation set and then test the performance of your classifier on unseen test data set. 

<p align="center">
  <img src="https://miro.medium.com/max/700/1*kpqurK-46RQxCllffLgM3w.png">
</p>

This is a good practice to divide the data set as:
    * **Training set:** The training set is the material through which the computer learns how to process information. Machine learning uses algorithms to perform the training part. A set of data used for learning, that is to fit the parameters of the classifier.
    * **Validation set:** Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. A set of unseen data is used from the training data to tune the parameters of a classifier.
    * **Test set:** A set of unseen data used only to assess the performance of a fully-specified classifier.

<p align="center">
  <img src="https://miro.medium.com/max/700/1*CeALK-1lzIWNJ7wN9DStlw.png">
</p>

* **Model Evaluation:** After the model is trained, it is then evaluated, using some evaluation metric, on the test dataset, which it has never seen before. Hence, the model tries to perform on the test dataset using only the knowledge gained from the training dataset. Once this is done we can develop a confusion matrix, this tells us how well our model is trained. A confusion matrix has 4 parameters, which are **True positives**, **True Negatives**, **False Positives** and **False Negative**. We prefer that we get more values in the True negatives and True positives to get a more accurate model. 

<p align="center">
  <img src="https://miro.medium.com/max/386/1*GMlSubndVt3g7FmeQjpeMA.png">
</p>

The size of the Confusion matrix completely depends upon the number of classes.
    * **True positives:** These are cases in which we predicted TRUE and our predicted output is correct.
    * **True negatives:** We predicted FALSE and our predicted output is correct.
    * **False positives:** We predicted TRUE, but the actual predicted output is FALSE.
    * **False negatives:** We predicted FALSE, but the actual predicted output is TRUE.
Some of the most common evaluation metrics are Accuracy score, F1 Score, Mean Absolute Error (MAE) and Mean Squared Error (MSE).

* **Performance Improvement:** The performance of the model can further be improved on both the training and testing datasets using various techniques like, cross-validation, hyper-parameter tuning or by trying out multiple machine learning algorithms and using the one which performs the best or even better, by using ensembling methods which combine the results from multiple algorithms. The goal is to improve the accuracy and also looking at the confusion matrix to try to increase the number of true positives and true negatives.


Without further ado, let us start with our first Machine Learning Algorithm.
[Machine Learning Algorithms - Playing with Data]()