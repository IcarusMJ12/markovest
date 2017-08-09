.PHONY: default install test

default: install

test:
	python3 -m tox

install:
	sudo python3 -m pip install -r requirements.txt
	sudo python3 setup.py install
