from tensorflow import keras
import pandas as pd
from sklearn.externals import joblib


new_vars = None
with open('vars.txt', 'r') as f:
    new_vars = f.read().split(' ')

blue_gold, red_gold, blue_minions, red_minions, blue_kills, blue_death, blue_assists, red_assists = new_vars


scaler = joblib.load('scaler1') 
loaded = keras.models.load_model('model1')

topredict = pd.DataFrame(data={'blueKills': [blue_kills],
                               'blueDeaths': [blue_death],
                               'blueTotalGold': [blue_gold],
                               'blueAssists': [blue_assists],
                               'blueTotalMinionsKilled':[blue_minions],
                               'redTotalGold':[red_gold],
                               'redTotalMinionsKilled':[red_minions],
                               'redAssists':[red_assists]})


topredict = scaler.transform(topredict)

print(loaded.predict(x=topredict))