#!/usr/bin/env python3

import random
from collections import defaultdict
import pickle

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.moses import MosesDetokenizer


detokenizer = MosesDetokenizer()


class MaximumRetriesReachedError(Exception):
    pass


class Chain(object):
    def __init__(self, link_size=3, seed=None):
        assert link_size >= 1
        random.seed(seed)
        self._link_size = link_size
        self._text_count = 0
        self._tuple_to_nxt = defaultdict(set)
        self._tuple_to_prev = defaultdict(set)
        self._tuple_to_text_id = defaultdict(set)
        self._starts = set()
        self._ends = set()
        self._seen_sentences = set()

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def _update_key(self, key, word):
        return key[-(self._link_size-1):] + (word,)

    def add_text(self, text, text_id):
        text = [tuple(word_tokenize(sent)) for sent in sent_tokenize(text)]
        for sent in text:
            self._seen_sentences.add(
                    detokenizer.detokenize(sent, return_str=True))
        words = ()
        for sent in text:
            if len(sent) >= self._link_size:
                self._starts.add(sent[:self._link_size])
                self._ends.add(sent[-self._link_size:])
            for word in sent:
                prev = words
                words = self._update_key(words, word)
                if len(prev) < self._link_size:
                    continue
                self._tuple_to_nxt[prev].add(words)
                self._tuple_to_prev[words].add(prev)
                self._tuple_to_text_id[prev].add(text_id)
        self._tuple_to_text_id[words].add(text_id)
        return self
    
    def make_sentences(self, count=1, retries=100):
        for _ignored in range(retries):
            words = random.choice(tuple(self._starts))
            key = words
            sent_count = 0
            prev_sentence_end = 0
            while True:
                try:
                    key = random.choice(tuple(self._tuple_to_nxt[key]))
                except IndexError:
                    break
                words = words + key[-1:]
                if key in self._ends:
                    if detokenizer.detokenize(words[prev_sentence_end:],
                            return_str=True) in self._seen_sentences:
                        break
                    prev_sentence_end = len(words)
                    sent_count += 1
                    if sent_count == count:
                        return detokenizer.detokenize(words, return_str=True)
        raise MaximumRetriesReachedError()


def main():
    pickled = 'interface.pickle'
    try:
        c = Chain.load(pickled)
    except FileNotFoundError:
        with open('interface.txt', 'r') as f:
            string = f.read()
        c = Chain().add_text(string, 0)
        c.save(pickled)
    print(c.make_sentences(3))


if __name__ == '__main__':
    main()
