install-dependencies:
	conda create --name off-clustering
	conda activate off-clustering
	pip install -r requirements.txt
	git submodule init
	git submodule update
	pip install -r teaching_ml_2023/requirements.txt


install-python-dependencies:
	git submodule init
	git submodule update
	pip install -r requirements.txt
	pip install -r teaching_ml_2023/requirements.txt