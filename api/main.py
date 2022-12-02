import json
import pandas as pd 
import numpy as np 
import nltk  as nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import flask
import pickle

app = flask.Flask(__name__)

with open("vectorizer.pickle", "rb") as f:
    vectorizer = pickle.load(f)

with open("recipes_encodings.pickle", "rb") as f:
    tfidf_ingredients = pickle.load(f)

recipe_ingredients = pd.read_csv(
        "https://raw.githubusercontent.com/jabrahamvm/Recipe-Recommender/nutrition/parsed_ds.csv", 
        encoding="UTF-8",
        nrows=5000)
recipe_ingredients = recipe_ingredients.dropna()

@app.route("/")
def test_flask():
    '''Test a flask response.'''
    ingredients = flask.request.args.get("ingredients")
    if ingredients:
        return flask.make_response(recommender(ingredients))
    else:
        return f"No ingredients were provided. Please enter ingredients."
    
def recommender(ingredients):
    '''Generate recipe recommendations from ingredient data'''
    # TODO Load model instead of training.
    
    ingredients_v = vectorizer.transform([ingredients])

    similarity_list = cosine_similarity(ingredients_v, tfidf_ingredients)
    sorted_indexes = np.argsort(similarity_list[0])[::-1]
    recipe_ids = recipe_ingredients["recipe_id"].iloc[sorted_indexes].values[0:20].tolist()
    recipes_names = recipe_ingredients["recipe_name"].iloc[sorted_indexes].values[0:20].tolist()
    my_dict = [{"id": recipe_ids[i], "name":recipes_names[i]} for i in range(len(recipe_ids))]
    #return json.dumps(my_dict)
    return flask.jsonify(my_dict)

def recommenderAPI(request):
    '''HTTP request to sent a recommendation'''
    ingredients = request.args.get("ingredients")
    print(ingredients)
    if ingredients:
        return flask.make_response(recommender(ingredients))
    else:
        return f"No ingredients were provided. Please enter ingredients."

@app.route('/recipes/<int:recipe_id>')
def getRecipe(recipe_id):
    return flask.jsonify(recipe_ingredients[recipe_ingredients["recipe_id"] == recipe_id].to_dict(orient="records")[0])

app.run(host="0.0.0.0", debug=True)
