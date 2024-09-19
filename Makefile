
test:
	pytest


dev:
	pip install -e .
	pip install -r requirements.txt
	pip install ipdb

docred_download:
	gdown --folder 1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw
	mv DocRED/train_distant.json data/re-docred/train_distant.json
