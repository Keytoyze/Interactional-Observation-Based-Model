import ultra, sys, json, os

def read_data(data_set, features, labels):
    list_num = len(data_set.initial_list)
    for i in range(list_num):
        list_size = len(data_set.initial_list[i])
        for x in range(list_size):
            doc_id = data_set.initial_list[i][x]
            if doc_id >= 0:
                features.append(data_set.features[doc_id])
                labels.append(data_set.labels[i][x])

data_dir = sys.argv[1]
train_set = ultra.utils.read_data(data_dir, "train", 0)
test_set = ultra.utils.read_data(data_dir, "test", 0)
valid_set = ultra.utils.read_data(data_dir, "valid", 0)

features = []
labels = []
read_data(train_set, features, labels)
read_data(test_set, features, labels)
read_data(valid_set, features, labels)

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier as Classifier

features = np.array(features)
labels = np.array(labels)

print("label shape: %s" % str(labels.shape))
print("feature shape: %s" % str(features.shape))

forest = Classifier(n_estimators=250, random_state=0, n_jobs=48)
forest.fit(features, labels)

print("score: %f" % (forest.score(features, labels)))

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(30):
    print("%d. feature %d (%f±%f)" % (f + 1, indices[f], importances[indices[f]], std[indices[f]]))

with open(os.path.join(data_dir, "importance.json"), "w") as fi:
    json.dump(indices.tolist(), fi)