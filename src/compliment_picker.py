import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class ComplimentPicker:

    def __init__(self, csv_filename, personality):

        feature_names = ['Eyes', 'Nose', 'Mouth', 'Lips']

        self.compliments = self._get_compliments_for_personality(pd.read_csv(csv_filename), personality, feature_names)

        self.kNN = KNeighborsClassifier(n_neighbors=1)
        self.kNN.fit(X=self.compliments[feature_names].values,
                     y=self.compliments[['Text']].index.values
                     )
    def _get_compliments_for_personality(self, df, personality, feature_names):
        temp = feature_names.copy()
        temp.insert(0, 'Text')
        compliments = df[df['Personality']==personality]
        compliments = compliments[temp]
        return compliments

    def pick_compliment(self, feature_vector):
        probs = self.kNN.predict_proba(feature_vector)
        noise = np.random.uniform(0, 0.5, probs.shape)
        probs = probs+noise
        compliment = self.compliments['Text'].loc[np.argmax(probs)]
        return compliment


def main():
    picker = ComplimentPicker('../data/compliment_database.csv', 'neutral')
    feat_vector = np.random.randint(10, size=(1,4))

    compliment = picker.pick_compliment(feat_vector)
    print(f'The compliment is: {compliment}')
if __name__ == '__main__':
    main()