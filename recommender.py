import nltk
import pandas as pd
import string
import re

def ingredients_parser(text):
    if text == "":
        return ValueError("The string is empty")
    
    parsed_ingredients = " ".join(text.split("^"))
    
    # Make everything lower case.
    pass