# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import math
import random

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # Collapse duplicate words first so each unique word only costs one indexer lookup,
        # and hit the indexer's dicts directly rather than through index_of/add_and_get_index
        # (each of which does two hash lookups where one .get() suffices).
        word_counts = Counter(word.lower() for word in sentence)
        objs_to_ints = self.indexer.objs_to_ints
        features = Counter()
        if add_to_indexer:
            ints_to_objs = self.indexer.ints_to_objs
            for word, count in word_counts.items():
                idx = objs_to_ints.get(word)
                if idx is None:
                    idx = len(objs_to_ints)
                    objs_to_ints[word] = idx
                    ints_to_objs[idx] = word
                features[idx] = count
        else:
            for word, count in word_counts.items():
                idx = objs_to_ints.get(word)
                if idx is not None:
                    features[idx] = count
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        words = [word.lower() for word in sentence]
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i + 1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
            if idx != -1:
                features[idx] += 1
        return features


class BetterFeatureExtractor(FeatureExtractor):
    """
    Unigram bag-of-words features weighted by TF-IDF. Document frequencies are computed once
    over the training corpus (passed in at construction time), then each sentence's raw term
    counts are scaled by tf * idf, where tf is the term's relative frequency within the sentence
    and idf = log(num_train_docs / doc_freq[term]).
    """
    def __init__(self, indexer: Indexer, train_exs: List[SentimentExample]=None):
        self.indexer = indexer
        self.doc_freq = Counter()
        self.num_docs = 0
        if train_exs is not None:
            self._fit_idf(train_exs)

    def _fit_idf(self, train_exs: List[SentimentExample]):
        self.num_docs = len(train_exs)
        for ex in train_exs:
            words_in_doc = set(word.lower() for word in ex.words)
            for word in words_in_doc:
                idx = self.indexer.add_and_get_index(word)
                self.doc_freq[idx] += 1

    def get_indexer(self):
        return self.indexer

    def _idf(self, idx: int) -> float:
        df = self.doc_freq[idx]
        if df == 0 or self.num_docs == 0:
            return 0.0
        return math.log(self.num_docs / df)

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        raw_counts = Counter()
        for word in sentence:
            word = word.lower()
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
            if idx != -1:
                raw_counts[idx] += 1

        features = Counter()
        num_terms = sum(raw_counts.values())
        for idx, count in raw_counts.items():
            tf = count / num_terms
            weight = tf * self._idf(idx)
            if weight != 0:
                features[idx] = weight
        return features


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: Counter, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * count for idx, count in features.items())
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: Counter, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def _sigmoid(self, z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * count for idx, count in features.items())
        return 1 if self._sigmoid(score) > 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    num_epochs = 10
    initial_lr = 1.0
    weights = Counter()

    # Precompute features once so we don't re-extract them every epoch
    train_feats = [feat_extractor.extract_features(ex.words, add_to_indexer=True) for ex in train_exs]

    random.seed(0)
    indices = list(range(len(train_exs)))
    for epoch in range(num_epochs):
        random.shuffle(indices)
        lr = initial_lr / (1 + epoch)
        for i in indices:
            ex = train_exs[i]
            features = train_feats[i]
            score = sum(weights[idx] * count for idx, count in features.items())
            prediction = 1 if score > 0 else 0
            if prediction != ex.label:
                update = lr if ex.label == 1 else -lr
                for idx, count in features.items():
                    weights[idx] += update * count

    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    num_epochs = 10
    initial_lr = 0.5
    weights = Counter()

    # Precompute features once so we don't re-extract them every epoch
    train_feats = [feat_extractor.extract_features(ex.words, add_to_indexer=True) for ex in train_exs]

    def sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    random.seed(0)
    indices = list(range(len(train_exs)))
    for epoch in range(num_epochs):
        random.shuffle(indices)
        lr = initial_lr / (1 + epoch)
        for i in indices:
            ex = train_exs[i]
            features = train_feats[i]
            score = sum(weights[idx] * count for idx, count in features.items())
            prob = sigmoid(score)
            error = ex.label - prob
            for idx, count in features.items():
                weights[idx] += lr * error * count

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer(), train_exs)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model