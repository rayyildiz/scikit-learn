initialize:
	pip install scikit-learn
	pip install pydotplus
	pip install numpy
	pip install matplotlib

seperator:
	@echo "**********************************************************"

run1:
	@echo "Running Hello World example"
	python3 hello-world-scikit.py

run2:
	@echo "Running Labradors And Greyhunds example"
	python3 labs-greyhunds.py

run3:
	@echo "Running iris problem example"
	python3 iris-problem.py

run4:
	@echo "Running iris problem with KNeighbors"
	python3 iris-knodes.py

run5:
	@echo "Custom classifier example"
	python3 custom-classifier.py

run-all:
	$(MAKE) run1
	$(MAKE) seperator
	$(MAKE) run2
	$(MAKE) seperator
	$(MAKE) run3
	$(MAKE) seperator
	$(MAKE) run4
	$(MAKE) seperator
	$(MAKE) run5
	$(MAKE) seperator
