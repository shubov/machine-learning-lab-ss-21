import os
import re
import codecs
import math
from .base_interface import BaseClass
import numpy as np


class NaiveBayesClassifier(BaseClass):
    def __init__(self, num_classes: int):
        """

        :param num_classes: number of unique classes in the dataset
        """
        self._classes = np.arange(num_classes) if num_classes is not None and num_classes > 0 else None
        self._priors = {}
        self._conditionals = {}

    def learn(self, features, targets):
        if self._classes is None:
            self._classes = np.unique(targets)

        log_num_elements = np.log(len(features))
        class_occurrences = np.bincount(targets)

        for cn in self._classes:
            log_class_occurrences = np.log(class_occurrences[cn] + 0.0000000001)

            # occurrences priors: frequency
            self._priors[cn] = log_class_occurrences - log_num_elements

            indices = np.argwhere(targets == cn)

            occurrences_in_class = np.sum(features[indices])

            # calculate conditionals
            self._conditionals[cn] = {}
            for feat_idx in range(features.shape[-1]):
                occurrences = np.sum(features[indices, feat_idx])
                log_occurrences = math.log(occurrences + 0.0000000001)
                log_occurrences_in_class = math.log(occurrences_in_class + 0.0000000001)
                self._conditionals[cn][feat_idx] = log_occurrences - log_occurrences_in_class

    def infer(self, features):
        results = []
        for bins in features:
            _, predicted_class = self._predict(bins)
            results.append(predicted_class)
        return np.stack(results)

    def _predict(self, binary_features):
        """

        :param binary_features:
        :return: (1) list of scores for each class, (2) predicted class
        """
        scores = {}
        for cn in self._classes:
            scores[cn] = self._priors[cn]
            for feat_idx in range(binary_features.shape[-1]):
                if binary_features[feat_idx] > 0:
                    scores[cn] += self._conditionals[cn][feat_idx]
        return scores, max(scores, key=scores.get)


class NaiveBayesClassifierPaths:
    def __init__(self):
        self.min_count = 1
        self.vocabulary = {}
        self.num_docs = 0
        self.classes = {}
        self.priors = {}
        self.conditionals = {}

    def train(self, path):
        """Train the model

        Keyword arguments:

        path -- path to the location of the train dataset
        """
        for class_name in os.listdir(path):
            self.classes[class_name] = {"doc_counts": 0, "term_counts": 0, "terms": {}}
            path_class = os.path.join(path, class_name)

            for doc_name in os.listdir(path_class):
                terms = self.tokenize_file(os.path.join(path_class, doc_name))
                self.num_docs += 1
                self.classes[class_name]["doc_counts"] += 1

                self.build_vocabulary(class_name, terms)

        for class_name in self.classes:
            self.calculate_priors(class_name)
            self.calculate_conditionals(class_name)

    def test(self, path):
        """Test the model

        Keyword arguments:

        path -- path to the location of the test dataset

        :returns: (list, list) (1)List of the true classes of the files, (2)List of predicted classes
        """
        true_class = []
        predicted_class = []

        for class_name in self.classes:

            for doc_name in os.listdir(os.path.join(path, class_name)):
                doc_path = os.path.join(path, class_name, doc_name)
                result_class = self.predict(doc_path)

                true_class.append(class_name)

                predicted_class.append(result_class)

        return true_class, predicted_class

    def predict(self, path):
        """Predict the class of a document

        Keyword arguments:

        path -- path to the document

        :returns: (string) Name of the class with the maximum score
        """
        tokens = self.tokenize_file(path)
        scores = self.calculate_scores(tokens)
        return max(scores, key=scores.get)

    def tokenize_str(self, doc):
        return re.findall(r'\b\w\w+\b', doc)  # return all words with #characters > 1

    def tokenize_file(self, doc_file):  # reading document, encoding and tokenizing
        with codecs.open(doc_file, encoding='latin1') as doc:
            doc = doc.read().lower()
            _header, _blank_line, body = doc.partition('\n\n')
            return self.tokenize_str(body)  # return all words with #characters > 1

    def build_vocabulary(self, class_name, terms):
        """Fill vocabulary

        Keyword arguments:

        class_name -- string
            Class name for the terms
        terms -- list
            List of words to fill the vocabulary with
        """
        for term in terms:
            self.classes[class_name]["term_counts"] += 1
            if term not in self.vocabulary:
                self.vocabulary[term] = 1
                self.classes[class_name]["terms"][term] = 1
            else:
                self.vocabulary[term] += 1
                if term not in self.classes[class_name]["terms"]:
                    self.classes[class_name]["terms"][term] = 1
                else:
                    self.classes[class_name]["terms"][term] += 1

    def calculate_conditionals(self, class_name):
        """Calculate conditional probabilities

        Keyword arguments:

        class_name -- string
            Class name to calculate conditional probabilities for
        """
        self.conditionals[class_name] = {}
        c_dict = self.classes[class_name]['terms']
        c_len = sum(c_dict.values())

        for term in self.vocabulary:
            t_ct = 1.
            t_ct += c_dict[term] if term in c_dict else 0.
            self.conditionals[class_name][term] = math.log(t_ct) - math.log(c_len + len(self.vocabulary))

    def calculate_priors(self, class_name):
        """Calculate prior probabilities for a class

        Keyword arguments:

        class_name -- string
            Class name to calculate prior probabilities for
        """
        self.priors[class_name] = math.log(self.classes[class_name]['doc_counts']) - math.log(self.num_docs)

    def calculate_scores(self, tokens):
        """Calculate scores for the list of tokens

        Keyword arguments:

        tokens -- list
            List of tokens

        :returns: (string) Name of the class with the maximum score
        """
        scores = {}
        for class_num, class_name in enumerate(self.classes):
            scores[class_name] = self.priors[class_name]
            for term in tokens:
                if term in self.vocabulary:
                    scores[class_name] += self.conditionals[class_name][term]
        return scores
