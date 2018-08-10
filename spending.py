import csv
import math
import copy
import operator
import pickle
from pathlib import Path
import shutil
import sys
import os
import re
import getopt

class SpendingNB:
    def __init__(self, file_path, USE_CACHED_CLASSIFIER):
        self.USE_CACHED_CLASSIFIER = USE_CACHED_CLASSIFIER
        self.train_file_name = './history_archive/cc_history_041018_073118.csv'
        self.test_file_name = file_path

        self.bag_of_words_unigram = {}
        self.vocabulary = {}
        self.category_counts = {}

        # Pickle files
        self.b_of_w_pickle = './model_cache/bag_of_words.pickle'
        self.voc_pickle = './model_cache/vocabulary.pickle'
        self.cat_cts_pickle = './model_cache/category_counts.pickle'

    def addExample(self, description, category):
        tokens = description.split()
        category_dict = {}

        if category in list(self.bag_of_words_unigram.keys()):
            category_dict = self.bag_of_words_unigram[category]

        for tok in tokens:
            category_dict[tok] = category_dict.get(tok, 0) + 1
            self.vocabulary[tok] = self.vocabulary.get(tok, 0) + 1

        self.bag_of_words_unigram[category] = category_dict
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

    def train(self):
        cache_check = 0
        if self.USE_CACHED_CLASSIFIER:
            my_file = Path(self.b_of_w_pickle)
            if my_file.is_file():
                with open(self.b_of_w_pickle, 'rb') as b_of_w_pickle:
                    self.bag_of_words_unigram = pickle.load(b_of_w_pickle)
                    cache_check += 1

            my_file = Path(self.voc_pickle)
            if my_file.is_file():
                with open(self.voc_pickle, 'rb') as voc_pickle:
                    self.vocabulary = pickle.load(voc_pickle)
                    cache_check += 1

            my_file = Path(self.cat_cts_pickle)
            if my_file.is_file():
                with open(self.cat_cts_pickle, 'rb') as cat_cts_pickle:
                    self.category_counts = pickle.load(cat_cts_pickle)
                    cache_check += 1

            if cache_check == 3:
                print('Found existing cached classifier!')
                return
        elif not self.USE_CACHED_CLASSIFIER or cache_check < 3:
            print('Creating new classifier.')
            with open(self.train_file_name) as csvfile:
                myreader = csv.reader(csvfile, delimiter=',')
                for purchase in myreader:
                    [date, cost, star, category, description] = purchase
                    self.addExample(description, category)
    
    def classify(self, purchase_desc):
        vocabulary_size = len(list(self.vocabulary.keys()))
        total_category_count = sum(self.category_counts.values())
        
        category_word_counts = {cat : sum(cat_dict.values()) for cat,cat_dict in self.bag_of_words_unigram.items()}
        category_log_probabilities = {cat : counts/total_category_count for cat,counts in self.category_counts.items()}

        category_denominators = {cat : float(wc + vocabulary_size) for cat,wc in category_word_counts.items()}

        category_prob_results = copy.deepcopy(category_log_probabilities)
        for word in purchase_desc.split():
            if word in list(self.vocabulary.keys()):
                category_numerators = {cat : float(cat_dict.get(word, 0) + 1) for cat,cat_dict in self.bag_of_words_unigram.items()}
                
                for category in category_log_probabilities.keys():
                    category_prob_results[category] += math.log(category_numerators[category])
                    category_prob_results[category] -= math.log(category_denominators[category])

        return(max(category_prob_results.items(), key=operator.itemgetter(1))[0])
    
    def test(self):
        correct = 0
        wrong = 0
        with open(self.test_file_name) as csvfile:
            myreader = csv.reader(csvfile, delimiter=',')
            for purchase in myreader:
                [date, cost, star, category, description] = purchase
                self.addExample(description, category)

                pred_category = self.classify(description)
                # print('actual:', category, '\tpredicted:', pred_category)

                if pred_category == category:
                    correct += 1
                else:
                    wrong += 1

        breadcrumbs = self.test_file_name.split('/')
        base_file_name = breadcrumbs[len(breadcrumbs) - 1]
        shutil.move(self.test_file_name, os.path.join('./history_archive/', base_file_name))

        accuracy = float(correct) / (correct + wrong)
        print('Accuracy: ' + str(accuracy * 100) + '% (' +  str(correct) + '/' + str(correct + wrong) + ')')
        print()
        return accuracy

def main():
    try:
        USE_CACHED_CLASSIFIER = False
        (options, args) = getopt.getopt(sys.argv[1:], 'c')
        if ('-c','') in options:
            USE_CACHED_CLASSIFIER = True

        if len(args) < 1:
            print('Usage: python3 spending.py -c (optional) <file.csv>')
            return
            
        file_path = args[0]
        my_file = Path(file_path)

        if not my_file.is_file():
            print('No file found.')
            return

        classifier = SpendingNB(file_path, USE_CACHED_CLASSIFIER)
        classifier.train()
        classifier.test()

        # Caching
        with open(classifier.b_of_w_pickle, 'wb') as b_of_w_pickle:
            pickle.dump(classifier.bag_of_words_unigram, b_of_w_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(classifier.voc_pickle, 'wb') as voc_pickle:
            pickle.dump(classifier.vocabulary, voc_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(classifier.cat_cts_pickle, 'wb') as cat_cts_pickle:
            pickle.dump(classifier.category_counts, cat_cts_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    except getopt.GetoptError as optionError:
        print(optionError)
    
if __name__ == '__main__':
    main()