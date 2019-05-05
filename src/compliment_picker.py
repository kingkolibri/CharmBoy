import os

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class ComplimentPicker:

    def __init__(self, csv_filename):

        self.feature_names = ['female',	'male',	'eyes',	'eyes-blue',	'eyes-brown',	'eyes-green',	'nose',	'smile',
                              'lips',	'hair',	'bracelet',	'glasses',	'style']

        self.compliments = pd.read_csv(csv_filename)

    def pick_compliment(self, feature_vector, personality=None):

        if personality is None:
            compliments_filtered = self.compliments
        else:
            compliments_filtered = self.compliments[self.compliments['personality'].str.contains(personality)]
        compliments_filtered = compliments_filtered.reset_index()

        kNN = KNeighborsClassifier(n_neighbors=5)
        kNN.fit(X=compliments_filtered[self.feature_names].values,
                y=compliments_filtered[['text', 'face_to_make']].index.values
                )

        probs = kNN.predict_proba(feature_vector)
        probs[0] = [prob+0.1*np.random.random_sample() if prob > 0 else prob for prob in probs[0]]

        compliment = compliments_filtered['text'].loc[np.argmax(probs)]
        return compliment


def test():
    picker = ComplimentPicker('../data/compliment_database.csv')
    feat_vector = np.random.randint(10, size=(1, 13))
    print(feat_vector)

    compliment = picker.pick_compliment(feat_vector, personality=None)

    tts = gTTS(text=compliment, lang='en')
    tts.save("../data/charmboy.mp3")
    os.system("mpg321 ../data/charmboy.mp3")

    print('The compliment is: {}'.format(compliment))


if __name__ == '__main__':
    test()
