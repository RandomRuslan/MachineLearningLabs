import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    dataset = pd.read_csv("shuttle.csv", header=None).values.astype(np.int32, copy=False)
    count = len(dataset)
    print("Size", "CART\t\t\t", "Random forest", sep="\t")
    for train_part in range(60, 100, 10):
        data_train = dataset[0:int(count*train_part/100)]
        data_test = dataset[int(count*train_part/100 + 1):]

        tree = DecisionTreeClassifier() #CART
        tree = tree.fit(data_train[:, :-1], data_train[:, -1])
        tree = tree.score(data_test[:, :-1], data_test[:, -1])

        forest = RandomForestClassifier(n_estimators=100) #Random forest
        forest = forest.fit(data_train[:, :-1], data_train[:, -1])
        forest = forest.score(data_test[:, :-1], data_test[:, -1])

        if(tree == 1.0):    #formatted output
            tree=str(tree)+'\t'*3
        print(str(train_part)+'%', tree, forest, sep='\t\t')

main()
