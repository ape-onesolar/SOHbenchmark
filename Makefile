virtualenv: ## Create virtual environment
	#if [ ! -d venv ]; then pip install virtualenv; virtualenv venv -p python3.10; fi
	if [ ! -d venv ]; then python3.11 -m venv venv; fi
	echo "To activate the new environment run this: 'source venv/bin/activate' (in Linux) or this: 'source venv/Scripts/activate' (in Windows)"

format: ## Format code using black
	#format code
	if [ -d venv ]; then . venv/*/activate; fi && \
	black --line-length 120 **.python
