import json
import pandas as pd 
import numpy as np 
import nltk  as nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import flask

app = flask.Flask(__name__)

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
    vectorizer = TfidfVectorizer()
    recipe_ingredients = pd.read_csv(
        "https://raw.githubusercontent.com/jabrahamvm/Recipe-Recommender/nutrition/parsed_ds.csv", 
        encoding="UTF-8",
        nrows=5000)
    recipe_ingredients = recipe_ingredients.dropna()
    df_ing_parsed = recipe_ingredients["ingredients_parsed"].values.astype("U")

    tfidf_ingredients = vectorizer.fit_transform(df_ing_parsed)
    ingredients_v = vectorizer.transform([ingredients])

    similarity_list = cosine_similarity(ingredients_v, tfidf_ingredients)
    sorted_indexes = np.argsort(similarity_list[0])[::-1]
    return json.dumps(
        recipe_ingredients['recipe_name'].iloc[sorted_indexes].values[0:20].tolist())

def recommenderAPI(request):
    '''HTTP request to sent a recommendation'''
    ingredients = request.args.get("ingredients")
    if ingredients:
        return flask.make_response(recommender(ingredients))
    else:
        return f"No ingredients were provided. Please enter ingredients."

app.run(host="0.0.0.0", debug=True)
