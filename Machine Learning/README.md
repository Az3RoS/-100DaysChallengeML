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


### Unsupervised Learning
