#!/usr/bin/env python3

import random
import re
from collections import defaultdict
import pickle
from builtins import open

from ansi.colour.rgb import rgb256
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
    __slots__ = ['text_id', 'intensity', 'threshold']

    def __new__(cls, string, *argv, **kw):
        return super(Word, cls).__new__(cls, string)

    def __init__(self, string, text_id, intensity=1, threshold=1):
        self.text_id = text_id
        self.intensity = intensity
        self.threshold = threshold

    def colorize(self):
        rgb = [0, 0, 0]
        if self.intensity < 0.5:
            rgb = [0xaa, 0xaa, 0xaa]
        elif self.intensity < self.threshold:
            rgb = [0x55, 0x55, 0x55]
        if self.intensity > 0.01:
            rgb[self.text_id] = 0xff
        else:
            rgb = [0xff, 0xff, 0xff]
        return rgb256(*rgb) + self


class Chain(object):
    _VERSION = 1

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
        # not-really tf-idf, because it's equally-weighted
        # frequency per document / overall frequency
        self._tfs = []  # per document
        # inverse *total* frequency
        # this is normalized by including a phony document that has one of each
        # possible word
        self._itf = defaultdict(lambda: 0)
        # will contain a dict of (doc_id, tf/itf) tuples
        # tf/itf ranges from 0 to <1
        self._weighings = None

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def _update_key(self, key, word):
        return key[-(self._link_size-1):] + (word,)

    def _update_word_frequency(self, text):
        tf = defaultdict(lambda: 0)
        total = 0
        for sent in text:
            for word in sent:
                if len(word) < 3:
                    continue
                word = word.lower()
                total += 1
                tf[word] += 1
        for key in tf.keys():
            tf[key] /= total
            self._itf[key] += tf[key]
        self._tfs.append(tf)

    def add_text(self, text, text_id=0):
        text = [tuple(word_tokenize(sent)) for sent in sent_tokenize(text)]
        self._update_word_frequency(text)
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

    def _maybe_build_weighings(self):
        if self._weighings is not None:
            return
        self._weighings = defaultdict(lambda: (0, 0))
        adjustment = 0.00001
        for key in self._itf.keys():
            self._itf[key] += adjustment
        for key in self._itf.keys():
            candidates = [self._tfs[i][key] for i in range(len(self._tfs))]
            highest = max(candidates)
            self._weighings[key] = (candidates.index(highest),
                                    highest/self._itf[key])

    def make_sentence(self, retries=10000, threshold=0.9):
        return self.make_sentences(retries=retries, threshold=threshold)

    def make_sentences(self, count=1, retries=10000, threshold=0.9):
        self._maybe_build_weighings()
        for retry in range(retries):
            covered_texts = defaultdict(lambda: 0)
            key = random.choice(tuple(self._starts))
            words = [Word(w, *self._weighings[w.lower()], threshold=threshold)
                     for w in key]
            for word in words:
                covered_texts[word.text_id] = max(covered_texts[word.text_id],
                                                  word.intensity)
            sent_count = 0
            prev_sentence_end = 0
            while True:
                try:
                    key = random.choice(tuple(self._tuple_to_nxt[key]))
                except IndexError:
                    break
                word = Word(key[-1], *self._weighings[key[-1].lower()],
                            threshold=threshold)
                words += [word]
                covered_texts[word.text_id] = max(covered_texts[word.text_id],
                                                  word.intensity)
                if key in self._ends:
                    if detokenizer.detokenize(
                            words[prev_sentence_end:],
                            return_str=True) in self._seen_sentences:
                        break
                    prev_sentence_end = len(words)
                    sent_count += 1
                    if sent_count == count:
                        if len(list(filter(
                                lambda x: x > threshold,
                                covered_texts.values()))) < len(self._tfs):
                            break
                        return reset(
                            detokenizer.detokenize([
                                w.colorize() for w in words], return_str=True))
        raise MaximumRetriesReachedError()
