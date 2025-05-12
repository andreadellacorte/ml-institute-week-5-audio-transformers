#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ml-institute-week-5-audio-transformers
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: install
install:
	@if [ -z "$(mode)" ]; then \
		echo "Usage: make install mode=--gpu|--cpu"; \
		exit 1; \
	fi
	./setup.sh $(mode)

.phony: activate
activate:
	@source .venv/bin/activate

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src
	isort --check --diff src
	black --check src

## Format source code with black
.PHONY: format
format:
	isort src
	black src

## Run tests
.PHONY: test
test:
	python -m pytest tests

PID_FILE := script.pid
LOG_FILE := output.log

.PHONY: nohup
nohup:
	@if [ -z "$(script)" ]; then \
		echo "Usage: make nohup script=your_script.py"; \
		exit 1; \
	fi
	nohup python $(script) > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)

.PHONY: stop
stop:
	@pkill -f python
	@rm -f $(PID_FILE)

.PHONY: logs
logs:
	tail -f $(LOG_FILE)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
