# Exploiting convolution neural network features along with hand crafted features in SVM.
In this experiment, we use input of softmax layer in [convolution nueral network][1] as valuable features, and we combine these features with lexicon features and classify tweets' sentiment using SVM.

# Preprocessing
We use SemEval 2013 dataset in our experiment. After tokenizing tweets with [ark-tweet-nlp][2], we replace Numerics, URLs, User mentions and Punctuations with NUM, URL, USR and PUN respectively. You can download the preprocessed tweets.

# Feature Extracting
We use five lexicon:
*Hashtag Sentiment Lexicons
*Sentiment140 Lexicons
*NRC Emotion Lexicon
*Bing Liu's Lexicon
*MPQA Subjectivity Lexicon
In each Lexicon we compute:
-

[1]: http://arxiv.org/abs/1408.5882
[2]: http://www.cs.cmu.edu/~ark/TweetNLP/
