import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class ComplimentPicker:

    def __init__(self, csv_filename):

        feature_names = ['Eyes', 'Nose', 'Mouth', 'Lips']

        self.compliments = pd.read_csv(csv_filename,)

        self.kNN = KNeighborsClassifier(n_neighbors=1)

        self.kNN.fit(X=self.compliments[feature_names].values,
                     y=self.compliments[['Text']].values
                     )

    def pick_compliment(self, feature_vector):
        # TODO: add randomizer to deal with synonym compliments, for example based on random summand on probabilities.
        return self.kNN.predict(feature_vector)
