#!/usr/bin/env python3


from mock import call, sentinel, MagicMock
import pytest

from markovest import Chain, MaximumRetriesReachedError


SHORTEST_TEXT = 'This is a sentence.'
SHORTEST_COLORED_TEXT = '\x1b[0m\x1b[31mThis \x1b[31mis \x1b[31ma ' + \
                        '\x1b[31msentence\x1b[31m.\x1b[0m'
ADDED_TEXT = 'a sentence. And this is another.'
COMBINED_COLORED_TEXT = '\x1b[0m\x1b[31mThis \x1b[31mis \x1b[31ma ' + \
                        '\x1b[31msentence\x1b[33m. \x1b[32mAnd ' + \
                        '\x1b[32mthis \x1b[32mis \x1b[32manother' + \
                        '\x1b[32m.\x1b[0m'


@pytest.fixture
def pickle(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr('markovest.pickle', m)
    return m


@pytest.fixture
def fopen(monkeypatch):
    m = MagicMock()
    m.return_value.__enter__.return_value = sentinel.f
    m.return_value.__exit__.return_value = False
    monkeypatch.setattr('markovest.open', m)
    return m


@pytest.fixture
def ch():
    return Chain().add_text(SHORTEST_TEXT)


def test_instantiate():
    Chain()


def test_add_text(ch):
    pass


def test_max_retries(ch):
    with pytest.raises(MaximumRetriesReachedError):
        ch.make_sentence()


def test_save(pickle, fopen, ch):
    ch.save(sentinel.fname)
    pickle.dump.assert_has_calls([call(ch, sentinel.f)])


def test_load(pickle, fopen):
    Chain.load(sentinel.fname)
    pickle.load.assert_has_calls([call(sentinel.f)])


def test_success(ch, monkeypatch):
    monkeypatch.setattr(ch, '_seen_sentences', [])
    sent = ch.make_sentence()
    assert sent == SHORTEST_COLORED_TEXT, print(
            repr(sent) + ' != ' + repr(SHORTEST_COLORED_TEXT))


def test_index_error(ch, monkeypatch):
    monkeypatch.setattr(ch, '_seen_sentences', [])
    with pytest.raises(MaximumRetriesReachedError):
        ch.make_sentences(5)


def test_combined_text(ch, monkeypatch):
    ch.add_text(ADDED_TEXT, 1)
    monkeypatch.setattr(ch, '_seen_sentences', [])
    monkeypatch.setattr(ch, '_starts', (('This', 'is', 'a'),))
    sent = ch.make_sentences(2)
    assert sent == COMBINED_COLORED_TEXT, print(
            repr(sent) + ' != ' + repr(COMBINED_COLORED_TEXT))
