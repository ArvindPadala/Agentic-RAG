setup:
	pip install -r requirements.txt

run-cli:
	python agent.py

run-ui:
	python app.py

test:
	python -m unittest discover tests/
