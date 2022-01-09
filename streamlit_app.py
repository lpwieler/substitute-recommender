import streamlit as st
import pandas as pd
import numpy as np
import math
import os
from gensim.models import Word2Vec
from difflib import get_close_matches
from libs.simple_image_download import simple_image_download
from func_timeout import func_set_timeout, FunctionTimedOut
from googletrans import LANGUAGES, LANGCODES
from google.cloud import translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

google_cloud_project = "projects/substitute-recommender/locations/global"

simple_image = simple_image_download()
translator = translate.TranslationServiceClient()

image_timeout = 3
default_image = "https://socialistmodernism.com/wp-content/uploads/2017/07/placeholder-image.png"

@st.cache(allow_output_mutation=True)
def load_model(model='SRM'):
    return Word2Vec.load(f'./models/{model}.model')

@st.cache()
def load_ingredient_list():
    return pd.read_pickle('./data/ingredient_list.pkl')

@func_set_timeout(image_timeout)
def image_url(ingredient):
    print(f'Searching image for ingredient "{ingredient}"...')
    return simple_image.urls(ingredient, 1, extensions={'.jpg'})[0]

@st.cache()
def search_image(ingredient):
    try:
        return image_url(ingredient)
    except FunctionTimedOut:
        return default_image

@st.cache()
def translate(word, language, mode="to_language"):
    if language == LANGUAGES["en"]:
        return word
    
    if mode == "to_language":
        dest=LANGCODES[language]
        src="en"
    elif mode == "to_english":
        dest="en"
        src=LANGCODES[language]
    else:
        return word

    print(f'Translating word "{word}" from {LANGUAGES[src]} to {LANGUAGES[dest]}...')

    translated_word = translator.translate_text(
        request={
            "parent": google_cloud_project,
            "contents": [word],
            "mime_type": "text/plain",
            "source_language_code": src,
            "target_language_code": dest
        }
    ).translations[0].translated_text

    return translated_word

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

language = st.sidebar.selectbox('', [(v) for _, v in LANGUAGES.items()], index=21)

score_translated = translate('score', language)
frequency_translated = translate('frequency', language)
similarity_translated = translate('similarity', language)

sorter_mapping = {}
sorter_mapping[score_translated] = 'score'
sorter_mapping[frequency_translated] = 'frequency'
sorter_mapping[similarity_translated] = 'similarity'

sort_key = st.sidebar.selectbox(translate('Sort criteria', language),(score_translated, frequency_translated, similarity_translated))
sort_by = sorter_mapping[sort_key]

suggested_substitutes = st.sidebar.slider(translate('Amount of suggested substitutes', language), 0, 30, 10)
wv_topn = st.sidebar.slider(translate('Number of top-N similar keys', language), 0, 50, 30)

show_table = st.sidebar.checkbox(translate('Show table', language), True)
show_images = st.sidebar.checkbox(translate('Show images', language), False)

st.subheader(translate('Find Ingredient Substitutions', language))

ingredient = st.text_input("", placeholder=translate("Ingredient", language))

if ingredient:
    ingredient = translate(ingredient, language, mode="to_english")

    try:
        substitutes = find_substitute(ingredient, wv_topn, suggested_substitutes, sort_by)
    except:
        st.warning(translate(f'Invalid ingredient "{ingredient}"', language))
        st.stop()

    substitutes_list = substitutes['ingredient'].to_list()

    if language != LANGUAGES["en"]:
        substitutes['ingredient'] = substitutes['ingredient'].apply(lambda ingredient: translate(ingredient, language))

        substitutes.rename(
            columns={
                'ingredient': translate('ingredient', language),
                'frequency': translate('frequency', language),
                'similarity': translate('similarity', language),
                'score': translate('score', language)
            },
            inplace=True
        )

    if show_images:
        st.image(search_image(ingredient), width=150)

    st.subheader(translate('Recommended Substitutes', language))

    if show_table:
        st.table(substitutes)

    if show_images:
        progress_bar = st.progress(0)
        progress_step = math.floor(100 / len(substitutes.index))
        images = []
        captions = substitutes[translate('ingredient', language)].to_list()

        for index, row in substitutes.iterrows():
            images.append(search_image(substitutes_list[index - 1]))
            progress_bar.progress(progress_step * index)

        progress_bar.empty()

        st.image(images, width=100, caption=captions)
