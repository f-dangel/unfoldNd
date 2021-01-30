.DEFAULT: help

help:
	@echo "install"
	@echo "        Install unfoldNd and dependencies"
	@echo "uninstall"
	@echo "        Unstall unfoldNd"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "test-light"
	@echo "        Run pytest on the light part of test and report coverage"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "conda-env"
	@echo "        Create conda environment 'unfoldNd' with dev setup"
	@echo "darglint-check"
	@echo "        Run darglint (docstring check) on the project"
	@echo "pydocstyle-check"
	@echo "        Run pydocstyle (docstring check) on the project"

.PHONY: install

install:
	@pip install -r requirements.txt
	@pip install .

.PHONY: uninstall

uninstall:
	@pip uninstall unfoldNd

.PHONY: install-test

install-test:
	@pip install -r requirements/test.txt

.PHONY: test test-light

test:
	@pytest -vx --run-optional-tests=expensive --cov=unfoldNd test

test-light:
	@pytest -vx --cov=unfoldNd test

.PHONY: install-lint

install-lint:
	@pip install -r requirements/lint.txt

.PHONY: black black-check

black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

.PHONY: flake8

flake8:
	@flake8 .

.PHONY: darglint-check

darglint-check:
	@darglint --verbosity 2 unfoldNd

.PHONY: pydocstyle-check

pydocstyle-check:
	@pydocstyle --count .

.PHONY: conda-env

conda-env:
	@conda env create --file .conda_env.yml
