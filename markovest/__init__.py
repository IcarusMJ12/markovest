#!/usr/bin/env python3

import random
import re
from collections import defaultdict
import pickle
from builtins import open

from ansi.colour.rgb import rgb8
from ansi.colour.fx import reset
import nltk


# this is really ghetto, but that's how `nltk` works... and it escapes venv
nltk.download('perluniprops')  # noqa
nltk.download('punkt')  # noqa


from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import moses


# serious shenanigans here -- we want `re.search` to ignore ansi color
# https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
ansi_escape = re.compile(r'\x1b[^m]*m')
re_search = re.search


def re_strip_search(haystack, needle):
    return re_search(haystack, ansi_escape.sub('', needle))


re.search = re_strip_search
detokenizer = moses.MosesDetokenizer()


class MaximumRetriesReachedError(Exception):
    pass


class Word(str):
    __slots__ = ['text_ids']

    def __new__(cls, string, *argv):
        return super(Word, cls).__new__(cls, string)

    def __init__(self, string, text_ids):
        self.text_ids = text_ids

    def colorize(self):
        rgb = [0, 0, 0]
        for text_id in self.text_ids:
            rgb[text_id] = 0xff
        return rgb8(*rgb) + self


class Chain(object):
    def __init__(self, link_size=2, seed=None):
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

    def add_text(self, text, text_id=0):
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

    def make_sentence(self, retries=100):
        return self.make_sentences(retries=retries)

    def make_sentences(self, count=1, retries=100):
        for _ignored in range(retries):
            key = random.choice(tuple(self._starts))
            words = [Word(w, self._tuple_to_text_id[key]) for w in key]
            sent_count = 0
            prev_sentence_end = 0
            while True:
                try:
                    key = random.choice(tuple(self._tuple_to_nxt[key]))
                except IndexError:
                    break
                words += [Word(key[-1], self._tuple_to_text_id[key])]
                if key in self._ends:
                    if detokenizer.detokenize(
                            words[prev_sentence_end:],
                            return_str=True) in self._seen_sentences:
                        break
                    prev_sentence_end = len(words)
                    sent_count += 1
                    if sent_count == count:
                        return reset(
                            detokenizer.detokenize([
                                w.colorize() for w in words], return_str=True))
        raise MaximumRetriesReachedError()
