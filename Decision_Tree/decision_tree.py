import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from Decision_Tree import reprocessing
import pickle

X, y, encode_saver = reprocessing.get_data()

# training
model = DecisionTreeClassifier()
model.fit(X, y)

# save model
f = open('models/play_decision_models', 'wb')
pickle.dump(model, f)
f.close()

# export tree
import graphviz
exporter = export_graphviz(model)
graph = graphviz.Source(exporter)
graph.render('show_tree/play_decision')






