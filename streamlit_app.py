import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import google.auth
from gensim.models import Word2Vec
from difflib import get_close_matches
from libs.simple_image_download import simple_image_download
from func_timeout import func_set_timeout, FunctionTimedOut
from googletrans import Translator as GoogleFreeTranslator, LANGUAGES, LANGCODES
from google.cloud.translate import TranslationServiceClient as GoogleCloudTranslator

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

_, project_id = google.auth.default()

google_cloud_project = f'projects/{project_id}/locations/global'

simple_image = simple_image_download()

google_translator_provider = os.getenv('GOOGLE_TRANSLATOR_PROVIDER', "cloud")

google_cloud_translator, google_free_translator = None, None

if google_translator_provider == "cloud":
    google_cloud_translator = GoogleCloudTranslator()
elif google_translator_provider == "free":
    google_free_translator  = GoogleFreeTranslator()

multi_language_support = True if (google_cloud_translator or google_free_translator) else False

image_timeout = 3
default_image = "https://socialistmodernism.com/wp-content/uploads/2017/07/placeholder-image.png"

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(model='SRM'):
    return Word2Vec.load(f'./models/{model}.model')

@st.cache(show_spinner=False)
def load_ingredients():
    return pd.read_pickle('./data/ingredients.pkl')

@st.cache(show_spinner=False)
def create_ingredient_list(df_ingredients):
    return [x.replace(' ', '_') for x in df_ingredients['ingredient'].to_list()]

@func_set_timeout(image_timeout)
def image_url(ingredient):
    print(f'Searching image for ingredient "{ingredient}"...')
    return simple_image.urls(ingredient, 1, extensions={'.jpg'})[0]

@st.cache(show_spinner=False)
def search_image(ingredient):
    try:
        return image_url(ingredient)
    except FunctionTimedOut:
        return default_image

def translate_word(word, src, dest):
    if google_translator_provider == "cloud":
        return google_cloud_translator.translate_text(
            request={
                "parent": google_cloud_project,
                "contents": [word],
                "mime_type": "text/plain",
                "source_language_code": src,
                "target_language_code": dest
            }
        ).translations[0].translated_text
    elif google_translator_provider == "free":
        return google_free_translator.translate(word, dest, src).text
    else:
        return word

@st.cache(show_spinner=False)
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

    return translate_word(word, src, dest)

def get_query_param(key, query_params, default=""):
    return query_params[key][0] if key in query_params else default

def find_ingredient(ingredient):
    ingredient_list = create_ingredient_list(df_ingredients)

    if ingredient in ingredient_list:
        print(f'Found exact match for ingredient "{ingredient}"')
        return ingredient
    else:
        matched_ingredient =  (get_close_matches(ingredient, ingredient_list, n=1, cutoff=0.8) or [None])[0]
        if matched_ingredient:
            print(f'Found close match "{matched_ingredient}" for ingredient "{ingredient}"')
            return matched_ingredient
        else:
            raise Exception(f'Did not find close match for ingredient "{ingredient}"')

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

@st.cache(show_spinner=False)
def find_substitute(ingredient, wv_topn=30, suggested_substitutes=10, sort_by='score'):
    ingredient = find_ingredient(ingredient.strip().replace(' ', '_').lower())
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
    
    possible_substitutes = df_ingredients.merge(df_substitutes_final, on='ingredient', how='inner')
    possible_substitutes['score'] = possible_substitutes.apply(lambda row: round(row.frequency * 20 ** (10 * row.similarity) / 10 ** 6), axis=1)
    possible_substitutes = possible_substitutes.sort_values(by=[sort_by], ascending=False)

    r = len(possible_substitutes)
    possible_substitutes['nr_substitute'] = np.arange(start=1, stop=(r+1), step=1)
    possible_substitutes = possible_substitutes.set_index('nr_substitute')
    
    return possible_substitutes.head(suggested_substitutes)


model = load_model()
df_ingredients = load_ingredients()

query_params = st.experimental_get_query_params()

if 'initial_query_params' not in st.session_state:
    st.session_state['initial_query_params'] = st.experimental_get_query_params()
    print(f'Initial query params: {st.session_state["initial_query_params"]}')

initial_query_params = st.session_state['initial_query_params']

default_values = {
    'sort_by': get_query_param('sort_by', initial_query_params),
    'suggested_substitutes': int(get_query_param('suggested_substitutes', initial_query_params, 10)),
    'wv_topn': int(get_query_param('wv_topn', initial_query_params, 30)),
    'show_table': get_query_param('show_table', initial_query_params, 'true').lower() == 'true',
    'show_images': get_query_param('show_images', initial_query_params, 'false').lower() == 'true',
    'ingredient': get_query_param('ingredient', initial_query_params)
}

if multi_language_support:
    default_values['language'] = get_query_param('language', initial_query_params)

## Sidebar content

# Language
if multi_language_support:
    language_default = default_values['language']
    language_list = [(v) for _, v in LANGUAGES.items()]
    language_index = 21

    if language_default:
        if language_default in language_list:
            language_index = language_list.index(language_default)

    language = st.sidebar.selectbox('', language_list, index=language_index)
    query_params['language'] = language
else:
    language = LANGUAGES["en"]

# Settings
score_translated = translate('score', language)
frequency_translated = translate('frequency', language)
similarity_translated = translate('similarity', language)

sorter_mapping = {}
sorter_mapping[score_translated] = 'score'
sorter_mapping[frequency_translated] = 'frequency'
sorter_mapping[similarity_translated] = 'similarity'

sort_by_default = default_values['sort_by']
sorter_list = list(sorter_mapping.values())
sorter_index = 0

if sort_by_default:
    if sort_by_default in sorter_list:
        sorter_index = sorter_list.index(sort_by_default)

sort_key = st.sidebar.selectbox(translate('Sort criteria', language),(score_translated, frequency_translated, similarity_translated), sorter_index)
sort_by = sorter_mapping[sort_key]
query_params['sort_by'] = sort_by

suggested_substitutes = st.sidebar.slider(translate('Amount of suggested substitutes', language), 0, 30, default_values['suggested_substitutes'])
query_params['suggested_substitutes'] = suggested_substitutes

wv_topn = st.sidebar.slider(translate('Number of top-N similar keys', language), 0, 50, default_values['wv_topn'])
query_params['wv_topn'] = wv_topn

show_table = st.sidebar.checkbox(translate('Show table', language), default_values['show_table'])
query_params['show_table'] = show_table

show_images = st.sidebar.checkbox(translate('Show images', language), default_values['show_images'])
query_params['show_images'] = show_images

## Main page content

st.subheader(translate('Find Ingredient Substitutions', language))

ingredient = st.text_input("", placeholder=translate("Enter Ingredient", language), value=default_values['ingredient'])

if ingredient:
    with st.spinner(translate('Searching substitutes...', language)):
        query_params['ingredient'] = ingredient

        ingredient_english = translate(ingredient, language, mode="to_english")

        try:
            substitutes = find_substitute(ingredient_english, wv_topn, suggested_substitutes, sort_by)
        except Exception as error:
            print(f'find_substitute failed with error: {error}')
            st.warning(translate(f'Invalid ingredient "{ingredient_english}"', language))
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
            st.image(search_image(ingredient_english), width=150)

        st.subheader(translate('Recommended Substitutes', language))

        if not show_table and not show_images:
            st.info(translate('All views are disabled. Table or image view needs to be enabled to see results.', language))

        if show_table:
            st.table(substitutes)

    if show_images:
        with st.spinner(translate('Loading images...', language)):
            progress_bar = st.progress(0)
            progress_step = math.floor(100 / len(substitutes.index))
            images = []
            captions = substitutes[translate('ingredient', language)].to_list()

            for index, row in substitutes.iterrows():
                images.append(search_image(substitutes_list[index - 1]))
                progress_bar.progress(progress_step * index)

            progress_bar.empty()

            st.image(images, width=100, caption=captions)

st.experimental_set_query_params(**query_params)
