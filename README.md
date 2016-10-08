<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
# Exploiting convolution neural network features along with hand crafted features in SVM.
In this experiment, we use input of softmax layer in [convolution nueral network][1] as valuable features, and we combine these features with lexicon features and classify tweets' sentiment using SVM.

# Preprocessing
We use SemEval 2013 dataset in our experiment. After tokenizing tweets with [ark-tweet-nlp][2], we replace Numerics, URLs, User mentions and Punctuations with NUM, URL, USR and PUN respectively. You can find the preprocessed tweets.

# Feature Extracting
We use five lexicon:
- Hashtag Sentiment Lexicons
- Sentiment140 Lexicons
- NRC Emotion Lexicon
- Bing Liu's Lexicon
- MPQA Subjectivity Lexicon

In each Lexicon we compute below features for negative and positive tokens of each tweet:
- Number of token with score not equal to zero
- Maximum score of tokens
- Totald score of tokens
- Score of last token

You can find the preprocessed tweets.

# Dependencies
The code requires:
- Python (2.7)
- Numpy, Scipy
- Theano
- Keras
- Sci-kit learn

# Running
For classification simply run:
> python classifier.py

# Results



[1]: http://arxiv.org/abs/1408.5882
[2]: http://www.cs.cmu.edu/~ark/TweetNLP/
