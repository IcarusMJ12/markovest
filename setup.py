#!/usr/bin/env python

from distutils.core import setup

setup(name='markovest',
      version='0.0.1',
      description='The markovest of Markov chains, with color-coding and '
                  'tf-idf-based sentence ratings.',
      author='Igor Kaplounenko',
      author_email='megawidget@gmail.com',
      url='https://github.com/megawidget/markovest',
      packages=['markovest'],
      scripts=['bin/markovest'])
