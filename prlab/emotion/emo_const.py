"""
This file contains some wide used variable that common in emotion
"""

# FER/FERPLUS labels and id map
ferlabels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
ferlbl2id = {val: pos for pos, val in enumerate(ferlabels)}

emowlabels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear']
emowlabels = [o.lower() for o in emowlabels]
emowlbl2id = {val: pos for pos, val in enumerate(emowlabels)}
