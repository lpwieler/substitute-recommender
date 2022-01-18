import streamlit as st
import pandas as pd
import numpy as np
import os
import google.auth
import libs.session_state as session_state
from gensim.models import Word2Vec
from difflib import get_close_matches
from dask import compute, delayed
from libs.simple_image_download import simple_image_download
from func_timeout import func_set_timeout, FunctionTimedOut
from googletrans import Translator as GoogleFreeTranslator, LANGUAGES, LANGCODES
from google.cloud.translate import TranslationServiceClient as GoogleCloudTranslator

logger = st._LOGGER

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

_, project_id = google.auth.default()

google_cloud_project = f'projects/{project_id}/locations/global'

simple_image = simple_image_download()

google_translator_provider = os.getenv('GOOGLE_TRANSLATOR_PROVIDER', 'cloud')

google_cloud_translator, google_free_translator = None, None

if google_translator_provider == 'cloud':
    google_cloud_translator = GoogleCloudTranslator()
elif google_translator_provider == 'free':
    google_free_translator  = GoogleFreeTranslator()

multi_language_support = True if (google_cloud_translator or google_free_translator) else False

translation_replacements = {
    'sort criteria': 'sort-criteria',
    'view settings': 'view-settings',
    'piment d espelette': 'piment d`espelette'
}

image_replacements = {
    'agave nectar': 'food agave nectar',
    'apple': 'food apple',
    'blackberry': 'food blackberry',
    'citrus': 'food citrus',
    'curacao': 'curacao drink',
    'dressing': 'salate dressing',
    'fat': 'food fat',
    'lean beef': 'ingredient lean beef',
    'liver': 'food liver',
    'mace': 'food mace',
    'mirin': 'mirin sauce',
    'oil': 'food oil',
    'squash': 'food squash',
    'raspberry': 'food raspberry',
    'rocket': 'food rocket',
    'sweetener': 'sugar substitute sweetener'
}

image_search_timeout = 5
default_image = 'https://socialistmodernism.com/wp-content/uploads/2017/07/placeholder-image.png'

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(model=os.getenv('MODEL', 'SRM')):
    return Word2Vec.load(f'./models/{model}.model')

@st.cache(show_spinner=False)
def load_ingredients():
    return pd.read_pickle('./data/ingredients.pkl')

@st.cache(show_spinner=False)
def create_ingredient_list(df_ingredients):
    return [x.replace(' ', '_') for x in df_ingredients['ingredient'].to_list()]

def find_replacement(word, replacements):
    return replacements[word.lower()] if word.lower() in replacements.keys() else word

@func_set_timeout(image_search_timeout)
def image_url(ingredient):
    logger.info(f'Searching image for ingredient "{ingredient}"...')
    search_query = find_replacement(ingredient, image_replacements)
    return simple_image.urls(search_query, 1, extensions={'.jpg'})[0]

@st.cache(show_spinner=False, max_entries=1000)
def search_image(ingredient):
    try:
        return image_url(ingredient)
    except FunctionTimedOut:
        return default_image

def translate_word(word, src, dest):
    if google_translator_provider == 'cloud':
        return google_cloud_translator.translate_text(
            request={
                'parent': google_cloud_project,
                'contents': [find_replacement(word, translation_replacements)],
                'mime_type': 'text/plain',
                'source_language_code': src,
                'target_language_code': dest
            }
        ).translations[0].translated_text
    elif google_translator_provider == 'free':
        return google_free_translator.translate(find_replacement(word, translation_replacements), dest, src).text
    else:
        return word

@st.cache(show_spinner=False)
def translate(word, language, src=None, dest=None):
    src = src or 'en'
    dest = dest or LANGCODES[language]

    if src == dest:
        return word

    logger.info(f'Translating word "{word}" from {LANGUAGES[src]} to {LANGUAGES[dest]}...')

    return translate_word(word, src, dest)

def get_query_param(key, query_params, default=''):
    return query_params[key][0] if key in query_params else default

@st.cache(show_spinner=False)
def find_ingredient(ingredient):
    ingredient_list = create_ingredient_list(df_ingredients)

    if ingredient in ingredient_list:
        logger.info(f'Found exact match for ingredient "{ingredient}"')
        return ingredient
    else:
        matched_ingredient =  (get_close_matches(ingredient, ingredient_list, n=1, cutoff=0.85) or [None])[0]
        if matched_ingredient:
            logger.info(f'Found close match "{matched_ingredient}" for ingredient "{ingredient}"')
            return matched_ingredient
        else:
            logger.info(f'Did not find match for "{ingredient}" in ingredient list')
            return ingredient

def remove_same_ingredients(ingredient, substitutes_list, remove_count=3, similarity_score=0.8):
    ingredients_to_remove = get_close_matches(ingredient, substitutes_list, remove_count, similarity_score)
    cleaned_substitutes_list = [x for x in substitutes_list if x not in ingredients_to_remove]
    return cleaned_substitutes_list

def remove_same_substitutes(substitutes_list):
    cleaned_substitutes_list = substitutes_list
    
    for substitute in substitutes_list:
        if substitute in cleaned_substitutes_list:
            cleaned_substitutes_list = remove_same_ingredients(substitute, cleaned_substitutes_list, remove_count=len(cleaned_substitutes_list), similarity_score=0.8)
            cleaned_substitutes_list.append(substitute)
        
    return cleaned_substitutes_list

def calculate_score(frequency, similarity, similarity_weight):
    return round(frequency * similarity_weight ** (10 * similarity) / 10 ** (similarity_weight / 5))

@st.cache(show_spinner=False)
def find_substitutes(ingredient, wv_topn=100, suggested_substitutes=10, similarity_weight=20, sort_by='similarity'):
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

    if possible_substitutes.empty:
        raise Exception(f'Did not find any substitutes for ingredient "{ingredient}"')

    possible_substitutes['score'] = possible_substitutes.apply(lambda row: calculate_score(row.frequency, row.similarity, similarity_weight), axis=1)
    possible_substitutes = possible_substitutes.sort_values(by=[sort_by], ascending=False)

    r = len(possible_substitutes)
    possible_substitutes['nr_substitute'] = np.arange(start=1, stop=(r+1), step=1)
    possible_substitutes = possible_substitutes.set_index('nr_substitute')
    
    return possible_substitutes.head(suggested_substitutes)


model = load_model()
df_ingredients = load_ingredients()

query_params = st.experimental_get_query_params()

session = session_state.get(initial_query_params=None, previous_language=None, current_values=None)

if not session.initial_query_params:
    session.initial_query_params = query_params.copy()
    logger.info(f'Initial query params: {session.initial_query_params}')

initial_query_params = session.initial_query_params

default_values = {
    'sort_by': get_query_param('sort_by', initial_query_params),
    'suggested_substitutes': int(get_query_param('suggested_substitutes', initial_query_params, 10)),
    'wv_topn': int(get_query_param('wv_topn', initial_query_params, 100)),
    'similarity_weight': int(get_query_param('similarity_weight', initial_query_params, 20)),
    'show_table': get_query_param('show_table', initial_query_params, 'true').lower() == 'true',
    'show_images': get_query_param('show_images', initial_query_params, 'true').lower() == 'true',
    'ingredient': get_query_param('ingredient', initial_query_params)
}

if multi_language_support:
    default_values['language'] = get_query_param('language', initial_query_params, LANGUAGES['en'])

if not session.current_values:
    session.current_values = default_values

## Sidebar content

# Language
if multi_language_support:
    default_language = default_values['language']

    if not session.previous_language:
        session.previous_language = default_language

    previous_language = session.previous_language

    language_list = [(v) for _, v in LANGUAGES.items()]
    language_index = 21

    if default_language in language_list:
        language_index = language_list.index(default_language)

    language = st.sidebar.selectbox('', language_list, index=language_index)
    query_params['language'] = [language]
    session.current_values['language'] = language

    if language != previous_language:
        current_ingredient = session.current_values['ingredient']

        if current_ingredient:
            translated_ingredient = translate(
                current_ingredient,
                language,
                src=LANGCODES[previous_language],
                dest=LANGCODES[language]
            )
            query_params['ingredient'] = [translated_ingredient]
            session.current_values['ingredient'] = translated_ingredient
    
        default_values = session.current_values
        session.previous_language = language
        session.initial_query_params = query_params

        st.experimental_rerun()
else:
    language = LANGUAGES['en']

# View settings
st.sidebar.subheader(translate('View Settings', language))

show_images = st.sidebar.checkbox(translate('Show images', language), default_values['show_images'])
query_params['show_images'] = [str(show_images).lower()]
session.current_values['show_images'] = show_images

show_table = st.sidebar.checkbox(translate('Show table', language), default_values['show_table'])
query_params['show_table'] = [str(show_table).lower()]
session.current_values['show_table'] = show_table

# Parameter settings
st.sidebar.subheader(translate('Parameter Settings', language))

score_translated = translate('score', language)
frequency_translated = translate('frequency', language)
similarity_translated = translate('similarity', language)

sorter_mapping = {}
sorter_mapping[score_translated] = 'score'
sorter_mapping[frequency_translated] = 'frequency'
sorter_mapping[similarity_translated] = 'similarity'

sort_by_default = default_values['sort_by']
sorter_list = list(sorter_mapping.values())
sorter_index = 2

if sort_by_default:
    if sort_by_default in sorter_list:
        sorter_index = sorter_list.index(sort_by_default)

sort_key = st.sidebar.selectbox(translate('Sort criteria', language), (score_translated, frequency_translated, similarity_translated), sorter_index)
sort_by = sorter_mapping[sort_key]
query_params['sort_by'] = [sort_by]
session.current_values['sort_by'] = sort_by

suggested_substitutes = st.sidebar.slider(translate('Amount of suggested substitutes', language), 0, 30, default_values['suggested_substitutes'])
query_params['suggested_substitutes'] = [suggested_substitutes]
session.current_values['suggested_substitutes'] = suggested_substitutes

wv_topn = st.sidebar.slider(translate('Number of top-N similar keys', language), 0, 200, default_values['wv_topn'])
query_params['wv_topn'] = [wv_topn]
session.current_values['wv_topn'] = wv_topn

similarity_weight = st.sidebar.slider(translate('Similarity weight', language), 0, 50, default_values['similarity_weight'])
query_params['similarity_weight'] = [similarity_weight]
session.current_values['similarity_weight'] = similarity_weight

## Main page content

st.subheader(translate('Find Ingredient Substitutions', language))

ingredient = st.text_input('', placeholder=translate('Enter Ingredient', language), value=default_values['ingredient'])

if ingredient:
    query_params['ingredient'] = [ingredient]
    session.current_values['ingredient'] = ingredient

    ingredient_english = translate(ingredient, language, src=LANGCODES[language], dest='en')

    try:
        ingredient_from_list = find_ingredient(ingredient_english.strip().replace(' ', '_').lower())
        substitutes = find_substitutes(ingredient_from_list, wv_topn, suggested_substitutes, similarity_weight, sort_by).copy(deep=True)
    except Exception as error:
        logger.warning(error)
        st.warning(f'{translate("Invalid ingredient", language)} "{ingredient}"')
        st.stop()

    substitutes_list = substitutes['ingredient'].to_list()

    if language != LANGUAGES['en']:
        ingredient_substitutes = substitutes['ingredient'].apply(lambda ingredient: delayed(translate, traverse=False)(ingredient, language))
        substitutes['ingredient'] = compute(ingredient_substitutes.to_list())[0]

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
        with st.spinner(translate('Loading ingredient...', language)):
            st.image(search_image(ingredient_from_list), width=150)

    st.subheader(translate('Recommended Substitutes', language))

    if not show_table and not show_images:
        st.info(translate('All views are disabled. Table or image view needs to be enabled to see results.', language))

    if show_images:
        with st.spinner(translate('Loading ingredients...', language)):
            images = []
            captions = substitutes[translate('ingredient', language)].to_list()

            for index, row in substitutes.iterrows():
                images.append(delayed(search_image, traverse=False)(substitutes_list[index - 1]))

            st.image(compute(images)[0], width=100, caption=captions)

    if show_table:
        st.table(substitutes)

st.experimental_set_query_params(**query_params)
