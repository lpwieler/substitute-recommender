import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from difflib import get_close_matches


st.title('Substitute Recommender')

@st.cache(allow_output_mutation=True)
def load_model(model='SRM'):
    return Word2Vec.load(f'./models/{model}.model')

@st.cache()
def load_ingredient_list():
    return pd.read_pickle('./data/ingredient_list.pkl')

def remove_same_ingredients(ingredient, substitutes_list, remove_count=3, similarity_score=0.8):
    ingredients_to_remove = get_close_matches(ingredient, substitutes_list, remove_count, similarity_score)
    cleaned_substitutes_list = [x for x in substitutes_list if x not in ingredients_to_remove]
    return cleaned_substitutes_list

def remove_same_substitutes(substitutes_list_without_same_ingredients):
    cleaned_substitutes_list = substitutes_list_without_same_ingredients
    
    for substitute in substitutes_list_without_same_ingredients:
        if substitute in cleaned_substitutes_list:
            cleaned_substitutes_list = remove_same_ingredients(substitute, cleaned_substitutes_list, remove_count=len(cleaned_substitutes_list), similarity_score=0.8)
            cleaned_substitutes_list.append(substitute)
        
    return cleaned_substitutes_list

def find_substitute(ingredient, wv_topn=30, suggested_substitutes=10, sort_by='score'):
    ingredient = ingredient.strip().replace(' ', '_').lower()
    similar_substitutes = model.wv.most_similar(ingredient, topn=wv_topn)

    df_substitutes = pd.DataFrame(similar_substitutes, columns = ['ingredient', 'similarity'])
    
    substitutes_list = df_substitutes['ingredient'].to_list()
    substitutes_list_without_same_ingredients = remove_same_ingredients(ingredient, substitutes_list, remove_count=wv_topn, similarity_score=0.8)
    substitutes_list_without_same_substitutes = remove_same_substitutes(substitutes_list_without_same_ingredients)
    df_substitutes['ingredient'] = pd.Series(substitutes_list_without_same_substitutes)
    
    df_substitutes = df_substitutes[['ingredient', 'similarity']]
    df_substitutes['nr_substitute'] = np.arange(start=1, stop=(wv_topn+1), step=1)

    df_substitutes_index = df_substitutes.set_index('nr_substitute')

    df_substitutes_final = df_substitutes_index.replace('_', ' ', regex=True)
    
    possible_substitutes = ingredient_list.merge(df_substitutes_final, on='ingredient', how='inner')
    possible_substitutes['score'] = possible_substitutes.apply(lambda row: round(row.frequency * 20 ** (10 * row.similarity) / 10 ** 6), axis=1)
    possible_substitutes = possible_substitutes.sort_values(by=[sort_by], ascending=False)

    r = len(possible_substitutes)
    possible_substitutes['nr_substitute'] = np.arange(start=1, stop=(r+1), step=1)
    possible_substitutes = possible_substitutes.set_index('nr_substitute')
    
    return possible_substitutes.head(suggested_substitutes)


model = load_model()
ingredient_list = load_ingredient_list()

sort_by = st.sidebar.selectbox('Sort criteria',('score', 'frequency', 'similarity'))
suggested_substitutes = st.sidebar.slider('Amount of suggested substitutes',0, 30, 10)
wv_topn = st.sidebar.slider('Number of top-N similar keys',0, 50, 30)

ingredient = st.text_input('Ingredient')

if ingredient:
    substitutes = find_substitute(ingredient, wv_topn, suggested_substitutes, sort_by)
    st.subheader('Recommended Substitutes')
    st.table(substitutes)


