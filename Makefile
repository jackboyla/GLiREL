
test:
	pytest


dev:
	pip install -e .
	pip install -r requirements.txt
	pip install ipdb
	git clone https://github.com/thunlp/DocRED.git