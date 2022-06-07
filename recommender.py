import pandas as pd
import numpy as np

from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

import json
import re

from typing import Union

class Recommender:
    def __init__(self, X: 'np.array', columns: list[str] = ['name', 'rating', 'wrkh', 'address', 'sector', 'spec', 'has_alc', 'has_park', 'has_delivery', 'coord']) -> None:
        """
        Initialize the Recommender.
        :param X: np.array
            Item matrix
        :param columns: list[str]
            The names of columns in the dataset. The json returned from recommendations will contain these names as keys
        """

        self.X = X
        self.vectorizer = Vectorizer()
        self.columns = columns

    def __similarity(self, q: list, x: list, metric: str = 'neighbors') -> float:
        """
        Compute similarity using one of the proposed methods, given by the metric parameter.
        :param q: list
            The first vector.
        :param x: list
            The second vector.
        :param metric: str
            The method to use for calculating similarity. One of: 'cosine', 'dot', 'euclidean'
        returns: float
            The similarity value
        """

        if metric == 'cosine':
            return distance.cosine(q, x)
        if metric == 'dot':
            return np.dot(q, x)
        if metric == 'euclidean':
            return distance.euclidean(q, x)
    
    def __neighbors_similarity(self, X, y: list, n: int) -> tuple['np.array', 'np.array']:
        """
        Compute similarity using K Nearest Neighbor.
        :param X:
            The array of items
        :param y: list
            The query for which to find the neighbor
        :param n: int
            Number of neighbors. Used as NearestNeighbors' n_neighbors
        returns: tuple of two np.arrays
            Return the recommended items and their indices in the dataset
        """

        knn = NearestNeighbors(n_neighbors=n)
        knn.fit(X)
        neighbors = knn.kneighbors([y], return_distance=False)[0]

        X = np.array(X)

        return X[neighbors], neighbors

    def recommend(self, y: list, n: int, metric: str = 'neighbors', return_indices=True) -> 'np.array':
        """
        Return n items similar to the query item y.
        :param y: list
            The query item.
        :param n: int
            Number of similar items to return.
        :param metric: str
            The method to use. One of: 'cosine', 'dot', 'euclidean', 'neighbors'. In case of 'neighbors' uses K Nearest Neighbors with n neighbors.
        returns: np.array
            The array of recommendations
        """

        if metric != 'neighbors':
            items = [
                {'data': x, 'similarity': self.__similarity(x, y, metric=metric)}
                for x in self.X
            ]

            recommendations = sorted(np.shuffle(items), key=lambda k: k['similarity'], reverse=True)[:n]

            return [r['data'] for r in recommendations]
        
        similarity, indices = self.__neighbors_similarity(self.X, y, n)

        if return_indices:
            return similarity, indices
        
        return similarity
    
    def format_recommendation(self, R: 'np.array', logging: bool = False) -> list[dict]:
        """
        Return a formated recommendation either in extended form or simpified, based on the logging argument.
        :param R: np.array
            The matrix of recommendations
        :param logging: bool
            Flag for returning simplified form (default: False) or extended form (True)
        returns: list of dicts
            A dictionary of recommendations with the information used for recommendation as elements of each item
        """

        res = []

        if logging:
            res = [{
                    'alcohol': x[0],
                    'delivery': x[1],
                    'parking': x[2],
                    'Fast food': x[3],
                    'Restaurant': x[4],
                    'Japanese': x[5],
                    'Cafe': x[6],
                    'Italian': x[7],
                    'Grill': x[8],
                    'Romanian / Moldavian': x[9],
                    'Bar': x[10],
                    'European': x[11],
                    'Other cuisine': x[12],
                    'Canteen': x[13]
                } for x in R]
        else:
            specialities = ['Fast Food', 'Restaurant', 'Japanese', 'Cafe', 'Italian', 'Grill', 'Romanian / Moldavian', 'Bar', 'European', 'Other cuisines', 'Canteen']

            res = [{
                'alcohol': x[0],
                'delivery': x[1],
                'parking': x[2],
                'speciality': specialities[np.argwhere(x[3:])[0][0]]
            } for x in R]
        
        return res

    def parse_recommendations(self, indices: 'np.array', dataset: 'np.array') -> list:
        """
        Parse to json info about the given recommendations
        :param indices: np.array
            array of indices of the recommended items
        :param dataset: np.array
            dataset - matrix of available items
        returns: list
        """
        df = dataset.whole_df

        res = [{
            self.columns[j]: df.iloc[i][self.columns[j]] for j in range(len(self.columns))
        } for i in indices]

        return res
    
    def recommend_and_parse(self, dataset: 'np.array', y: Union['json', list], n: int, metric: str = 'neighbors', json_input: bool = True) -> list:
        """
        Perform a recommendation and return a dictionary of recommended items with the necessary information
        :param dataset: np.array
            dataset - matrix of available items
        :param y: list or json
            if json_input is True, y is a json object, else y is a list of a vectorized item
        :param n: int
            the number of recommendations to return
        :param metric: str
            the metric to use for finding recommendations
        :param json_input: bool
            set y argument's type
        return: list
        """

        if json_input:
            y = self.vectorizer.vectorize(y)

        _, indices = self.recommend(y, n, metric)

        return self.parse_recommendations(indices, dataset)
    
    @staticmethod
    def show_recommendation(recommendation, show_working_hours, show_coordinates):
        coordinates = recommendation['coord'].split(',')
        x = coordinates[0].strip()[:-1]
        y = coordinates[1].strip()

        working_hours = f"""
        Working hours: {recommendation['wrkh']}""" if show_working_hours else ''
        directions = f"""
        Directions: https://www.google.com/maps/search/{x},+{y}/@{x},{y}""" if show_coordinates else ''
        alcohol = """
        Alcohol: yes""" if recommendation['has_alc'] in ['y', 'yes'] else ''
        delivery = """
        Delivery: yes""" if recommendation['has_delivery'] in ['y', 'yes'] else ''
        parking = """
        Parking: yes""" if recommendation['has_park'] in ['y', 'yes'] else ''

        return f"""
        Name: {recommendation['name']}
        Rating: {recommendation['rating']}/5
        Speciality: {recommendation['spec']}
        Address: {recommendation['address']}{working_hours}{directions}{alcohol}{delivery}{parking}
        """
    

class Dataset:
    def __init__(self, df: 'pandas.DataFrame', columns: list[str] = ['spec', 'has_alc', 'has_delivery', 'has_park']) -> None:
        """
        Initialize the Dataset class
        :param df: pandas.DataFrame
            The DataFrame from which to build the dataset
        :param columns: list[str]
            The column names of the dataframe
        """

        self.whole_df = df
        self.df = df[columns]
        self.vectorizer = Vectorizer()
        self.X = None

    def __to_dict(self, x: list) -> dict:
        """
        Convert item into a simplified dictionary form for further processing
        :param x: list
            item to be converted
        returns: dict
            dictionary representing the item
        """

        return {
            'alcohol': x[1],
            'delivery': x[2],
            'parking': x[3],
            'speciality': x[0]
        }
    
    def vectorize_sample(self, x: list) -> 'np.array':
        """
        Vectorize a single item
        :param x: list
            Item to be vectorized
        returns: np.array
            The vectorized item
        """

        obj = json.dumps(self.__to_dict(x))

        return self.vectorizer.vectorize(obj)
    
    def vectorize(self):
        """
        Vectorize the given dataset
        """

        ds = []

        for _, x in self.df.iterrows():
            ds.append(self.vectorize_sample(x))

        self.X = np.array(ds)   

    def __len__(self):
        if self.X is not None:
            return len(self.X)
        
        return 0
    
    def __str__(self):
        if self.X is not None:
            return f'<Dataset: {self.X.shape}>'
        
        return f'<Dataset: (empty)>'


class Vectorizer:
    def __init__(self) -> None:
        """
        Initialize the Vectorizer class.
        """

        # item form:
        # [Fast food, Restaurant, Japanese, Cafe, Italian, Grill, Romanian / Moldavian,
        # Bar, European, Other cuisine, Canteen, Alcohol, Delivery, Parking]
        self.item = []
        self.columns = []

    def __vectorize_speciality(self) -> None:
        """
        Vectorize the speciality key of the json object. Appends the truth values for each speciality to the item array.
        """

        specialities = ['Fast food', 'Restaurant', 'Japanese', 'Cafe', 'Italian', 'Grill', 'Romanian / Moldavian', 'Bar', 'European', 'Other cuisine', 'Canteen']

        for speciality in specialities:
            self.item.append(1 if self.obj['speciality'] == speciality else 0)


    def __vectorize_binary(self, key: str) -> int:
        """
        A unified function to calculate and append to the item array binary valued key-value pairs of the json object. Intended for 'alcohol', 'delivery' and 'parking'.
        :param key: str
            The key of the dictionary to select the correct feature of the item. One of 'alcohol, 'delivery' or 'parking'
        returns: int
            The vectorized form of the binary feature
        """

        x = 0 # initialize to 'doesn't matter' option

        if self.obj[key].lower() in ['n', 'no']:
            x = 1
        elif self.obj[key].lower() in ['y', 'yes']:
            x = 2
        
        self.item.append(x)      

    def vectorize(self, obj: 'json') -> 'np.array':
        """
        Parse the json and vectorize it.
        :param obj:
            The json object
        returns: int
            Vectorize the given json object
        """

        self.item = []
        self.columns = []
        self.obj = json.loads(obj)

        self.__vectorize_binary('alcohol')
        self.__vectorize_binary('delivery')
        self.__vectorize_binary('parking')
        self.__vectorize_speciality()

        return np.array(self.item)

# Helper functions
def process_database(path: str, columns: list[str] = ['spec', 'has_alc', 'has_delivery', 'has_park']) -> np.array:
    '''
    Create the DataFrame from the database at the given path and vectorize it
    :param path: str
        The path to the database
    :param columns: list[str]
        A list of column names to be selected from the database for vectorization
    :returns np.array
        The vectorized database 
    '''

    df = pd.read_excel(path)
    dataset = Dataset(df, columns=columns)
    dataset.vectorize()

    return dataset