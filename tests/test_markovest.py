#!/usr/bin/env python3


from mock import call, sentinel, MagicMock
import pytest

from markovest import Chain, MaximumRetriesReachedError


SHORTEST_TEXT = 'This is a sentence.'


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
    assert ch.make_sentence() == SHORTEST_TEXT


def test_index_error(ch, monkeypatch):
    monkeypatch.setattr(ch, '_seen_sentences', [])
    with pytest.raises(MaximumRetriesReachedError):
        ch.make_sentences(5)
