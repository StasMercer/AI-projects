import numpy as np
import pandas as pd
import argparse


def distance(x, y):
    sum = 0

    # Vypocet suctu rozdelov atributov
    for xi, yi in zip(x, y):
        sum += (xi - yi) **2
    
    # Odmocnina suctu
    return sum ** (1/2)

def knn(dataframe, target_class, y, N=3):
    df_vals = dataframe.drop(target_class, axis=1)

    # Vypocet vzdalenosti pre kazdy element v datasete a neznamy element
    df_vals['distance'] = df_vals.apply(lambda x: distance(x.to_numpy(), y), axis=1)
    # print(df_vals)

    # Zistime indexy N elementov z najmensou vzdalenostou
    smallest_indexes = df_vals.nsmallest(N, 'distance').index

    # Zistime triedy indexov
    nearest = dataframe.loc[smallest_indexes][target_class]
    # print(nearest)
    
    #Najdeme najcastiejsie triedy a vyberjeme maximalnu z nych 
    most_frequent = nearest.value_counts().index[0]
    return most_frequent


# Parser argumentov
parser = argparse.ArgumentParser(description='KNN classifier')

parser.add_argument('path', metavar='path', type=str, help='path to dataset')
parser.add_argument('target_class', metavar='target_class', type=str, help='target field to classify')
parser.add_argument('data_path', metavar='data_path', type=str, help='data to classify')
parser.add_argument('N', metavar='N', type=int, help='Neighbours count')

args = parser.parse_args()

dataset_path = args.path
target_class = args.target_class
data_path = args.data_path
N = args.N

if N < 1:
    raise ValueError('Invalid neighbours count')

with open(data_path) as f:

    data = []
    df = pd.read_csv(dataset_path)

    for line in f.readlines():
        elem = line.strip().split(',')

        if len(elem) != len(df.columns)-1:
            raise ValueError(f'Invalid features length. expected {len(df.columns)-1} got {len(elem)}')
        
        data.append(list(map(float, elem)))
    
    for elem in data:
        # klasifikacia kazdeho prikladu
        predicted = knn(df, target_class, np.array(elem), N)
        print(f"{elem} -> {predicted}")

