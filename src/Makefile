all:
	python3 main.py

view: all
	xdg-open tree_graph/decision_tree_train.pdf
	xdg-open tree_graph/decision_tree_prune.pdf

check:
	python3 entropy_test.py
	python3 gini_impurity_test.py 
