# Hate-Speech-Detection-in-Tweets

## Problem Category: 
Text Classification

## Classification Models Used: 
Decision Trees and Neural Networks

## Objective: 
Comparing the performance of various machine learning models in multi-class text classification

Project Tasks:

1. Data Collection
2. Exploratory Data Analysis
3. Baseline Model Selection (Random Forest or Gradient Boosting?)
4. Artificial Neural Networks for Text Classification
5. Comparion of Different models using macro-average F1 score.

# 1. Data Collection

Twitter data was collected from three different sources and combined to form a larger dataset.

1. https://data.world/crowdflower/hate-speech-identification
2. https://www.kaggle.com/vkrahul/twitter-hate-speech/activity#train_E6oV3lV.csv
3. http://github.com/zeerakw/hatespeech

All tweets were classified as either hate speech, offensive or neutral. 


# 2. Exploratory Data Analysis

Following features were removed from the tweet during preprocessing:

1.	Username: Usernames were limited to a maximum length of 15 characters after ‘@' symbol.  
2.	Web address: Web addresses started either with "https://” or directly as “www…".
3.	Retweet handle: Keyword 'RT' short for re-tweet.
4.	Punctuations: Punctuations are most common feature in the online texts and do not convey any meaning to the machine learning classifiers. 
5.	Emoticons: Emoji could be useful in the right context but, dropped for this study.
6.	Numbers: Numeric data did not add to any text classification relevance.
7.	Stop words: Stop words are common words with high occurrence but carry little information for the text classification tasks, e.g., words like a, an and the.
8. All uppercase characters were converted to the lowercases.

The final data set consisted of 6580, 6585 and 6585, hate, offensive and neutral tweets, repectively.

The data set was split into 2:1 training to test set ratio. For neural networks the training set is futher split in 2:1 ratio for training and validation test set.

The input tweet text was first tokenized, where the sentence was split into words and each unique word was assigned an index starting with 1 (index 0 was reserved for zero padding). The training dataset had vocabulary size of 18,602 words.

For this study, 300-dimensional pre-trained GloVe model was used for the vector representations of the tweets and the tokens. The dataset had about 400,000 unique words and a unique vector for each word. An empty embedding matrix (18,602 x 300) was created. Each word from the training set was queried against the GloVe vocabulary and corresponding dense vectors were stored in the embedding matrix. For words with no match in GloVe corpus were initialized with zero vectors.

# 3. Baseline Model Selection

The input features were modified by converting the word embeddings into document embedding such that each document vectors belonged to one of the class categories. The document embedding per tweet was computed by simply averaging the 300-dimensional word vectors (excluding the zero vectors corresponding to zero-paddings) of the tweet. Using grid search, the best fit random forest and gradient boosting model was selected based with minimum 5-fold stratified cross-validation error.

Without the loss of too much performance, random forests are comparable to the gradient boosting trees. In terms of word vectorization, TD-IDF character N-grams were superior than any other input features. Since the other classifiers (neural networks) were trained using word level vectorization, word-unigram + gradient boosting model was selected as the baseline model for this study. 

# 4. Artificial Neural Networks for Text Classification

Multilayer neural network was trained on the twitter dataset and had relatively good performance only about 2% less than the macro F1-score of the baseline model. 1D covnets performed better than MLPs and at the level of the baseline model. LSTM neural networks outperformed all other classifiers by a slight margin with an accuracy score of 87%. A hybrid model combining the 1D covnets and LSTM model also performed in the average range of other neural network classifiers. In terms of computational power 1D covnets performed well and did not require long computation time. In terms of accuracy score, all classifiers had almost similar performance. CNN and gradient boosting methods were similar in terms of performance but had three times more speed. Same was true when comparing LSTM with hybrid model. When considering both the performance metrics and the speed of execution, CNNs had advantage over other classifiers. 
