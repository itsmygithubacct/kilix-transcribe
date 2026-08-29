PYTHON ?= python3

.PHONY: check
check:
	$(PYTHON) tools/check_design.py
	$(PYTHON) -m unittest discover -s tests -v
