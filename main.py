from flask import Flask, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

import json
import pandas as pd #cargar los daots
import numpy as np #operaciones matriciales
import nltk  as nltk #libreria de procesamiento de lenguaje natural
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, make_response
import joblib
app = Flask(__name__)
api = Api(app)
CORS(app)

def recommender(ingredients):
    '''Generate recipe recommendations from ingredient data'''
    # TODO Load model instead of training.
    vectorizer = joblib.load("./vectorizer.pkl")

    recipe_ingredients = pd.read_csv(
        "https://raw.githubusercontent.com/jabrahamvm/Recipe-Recommender/nutrition/parsed_ds.csv", 
        encoding="UTF-8",
        nrows=5000)
    recipe_ingredients = recipe_ingredients.dropna()

    tfidf_ingredients = joblib.load("./tfidf_recipe.pkl")
    ingredients_v = vectorizer.transform([ingredients])

    similarity_list = cosine_similarity(ingredients_v, tfidf_ingredients)
    sorted_indexes = np.argsort(similarity_list[0])[::-1]
    return json.dumps(
        recipe_ingredients['recipe_name'].iloc[sorted_indexes].values[0:20].tolist())

class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running, please type /recommender?ingredients=ingredient1,ingredient2,ingredient3'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}

class Recommender(Resource):
    def get(self):
        ingredients = request.args.get('ingredients')
        return jsonify({'data': recommender(ingredients)})

api.add_resource(status, '/')
api.add_resource(Recommender, '/recommender')

if __name__ == '__main__':
    app.run()