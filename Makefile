setup:
	pip install -r requirements.txt

run-cli:
	python agent.py

run-ui:
	python app.py

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

test:
	python -m pytest tests/
