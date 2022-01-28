import streamlit as st
st.set_page_config(layout='wide')
with open('styles/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import re
import os
import warnings; warnings.simplefilter('ignore')
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

udemy = pd.read_csv('data/udemy.csv')
st.header('Recommendation System')
with st.form(key='my_form', clear_on_submit=True):
    submit_home = st.form_submit_button('Home')
    search_string = st.text_input('Search Here', max_chars=55)
    submit_button = st.form_submit_button('Search')

def extract_best_indices(m, topk, mask=None):
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1]
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask)
    best_index = index[mask][:topk]
    return best_index
STOPWORDS = set(stopwords.words('english'))
def tokenizer(nltk, min_words = 4, max_words = 200, stopwords = 'english', lemmatize=True):
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(nltk)]
    else:
        tokens = [w for w in word_tokenize(nltk)]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words and w not in stopwords)]
    return tokens

token_stop = tokenizer(''.join(STOPWORDS), lemmatize=False)
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
tfidf_mat = vectorizer.fit_transform(udemy['description'].values.astype('U'))
def recommender(search_string, tfidf_mat):
    def get_recommendations_tfidf(nltk, tfidf_mat):
        tokens = [str(tok) for tok in tokenizer(nltk)]
        vec = vectorizer.transform(tokens)
        mat = cosine_similarity(vec, tfidf_mat)
        best_index = extract_best_indices(mat, topk=7)
        return best_index
    best = get_recommendations_tfidf(search_string, tfidf_mat)
    return udemy[['Course Name','Short Description','Original rating','Categories','Difficulty']].iloc[best]

try:
    content_based = recommender(search_string, tfidf_mat)
    content = pd.DataFrame(content_based)
    st.write('You Searched for {}'.format(search_string))
    st.subheader('Recommended Courses')
    for i in range(7):
        st.metric(label=content['Difficulty'].iloc[i], value=content['Course Name'].iloc[i], delta=content['Original rating'].iloc[i].astype(str))
    topics_maybe_interested = content_based['Categories']
    topics_maybe_interested = pd.DataFrame(topics_maybe_interested)
    topics_maybe_interested['newcol'] = topics_maybe_interested['Categories']
    topics_maybe_interested.dropna()
    tmi = topics_maybe_interested['newcol'].str.split(',', expand=True).stack().value_counts()
    tmi = tmi.to_dict()
    topics_mi = [*tmi]
    st.subheader('Topics or Categories you maybe Interested in')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(topics_mi[0])
    with col2:
        st.write(topics_mi[1])
    with col3:
        st.write(topics_mi[2])

except ValueError:
    df = pd.read_csv('data/Udemy_Clean.csv')
    rating_counts = df[df['No_of_Ratings'].notnull()]['No_of_Ratings'].astype(int)
    rating_averages = df[df['Overall_Rating'].notnull()]['Overall_Rating'].astype(int)
    c = rating_averages.mean()
    m = rating_counts.quantile(0.95)
    col_list = ['Title', 'Overall_Rating', 'No_of_Ratings', 'Category']
    qualified = df[(df['No_of_Ratings'] >= m)
                  & (df['No_of_Ratings'].notnull())
                  & (df['Overall_Rating'].notnull())][col_list]
    qualified['No_of_Ratings'] = qualified['No_of_Ratings'].astype(int)
    qualified['Overall_Rating'] = qualified['Overall_Rating'].astype(int)
    def weighted_rating(x):
        v = x['No_of_Ratings']
        r = x['Overall_Rating']
        return (v/(v+m) * r) + (m/(m+v) * c)
    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('weighted_rating', ascending=False).head(250)
    # popularity based #
    popularity_based = qualified.head(5)
    st.subheader('Trending Courses')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('🔥 Trending this week', value=popularity_based['Title'].iloc[0], delta=popularity_based['No_of_Ratings'].iloc[0].astype(str))
    col2.metric('🔥 Trending this week', value=popularity_based['Title'].iloc[1], delta=popularity_based['No_of_Ratings'].iloc[1].astype(str))
    col3.metric('🔥 Trending this week', value=popularity_based['Title'].iloc[2], delta=popularity_based['No_of_Ratings'].iloc[2].astype(str))
    col4.metric('🔥 Trending this week', value=popularity_based['Title'].iloc[3], delta=popularity_based['No_of_Ratings'].iloc[3].astype(str))
    col5.metric('🔥 Trending this week', value=popularity_based['Title'].iloc[4], delta=popularity_based['No_of_Ratings'].iloc[4].astype(str))

    rating_counts_stars = df[df['No_of_Ratings'].notnull()]['No_of_Ratings'].astype(float)
    rating_averages_stars = df[df['Overall_Rating'].notnull()]['Overall_Rating'].astype(float)
    c_stars = rating_averages_stars.mean()
    m_stars = rating_counts_stars.quantile(0.95)
    col_list_stars = ['Title', 'Overall_Rating', 'No_of_Ratings', 'Category']
    qualified_stars = df[(df['No_of_Ratings'] >= m)
                  & (df['No_of_Ratings'].notnull())
                  & (df['Overall_Rating'].notnull())][col_list]
    qualified_stars['No_of_Ratings'] = qualified_stars['No_of_Ratings'].astype(float)
    qualified_stars['Overall_Rating'] = qualified_stars['Overall_Rating'].astype(float)
    def weighted_rating_stars(x_stars):
        v_stars = x_stars['No_of_Ratings']
        r_stars = x_stars['Overall_Rating']
        return (v_stars/(v_stars+m_stars) * r_stars) + (m_stars/(m_stars+v_stars) * c_stars)
    qualified_stars['weighted_rating_stars'] = qualified_stars.apply(weighted_rating_stars, axis=1)
    qualified_stars = qualified_stars.sort_values('weighted_rating_stars', ascending=False).head(250)
    # trending courses #
    trending_courses = qualified_stars.head(5)
    st.subheader('Most Popular Courses')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=trending_courses['Overall_Rating'].iloc[0].astype(str), value=trending_courses['Title'].iloc[0], delta=trending_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=trending_courses['Overall_Rating'].iloc[1].astype(str), value=trending_courses['Title'].iloc[1], delta=trending_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=trending_courses['Overall_Rating'].iloc[2].astype(str), value=trending_courses['Title'].iloc[2], delta=trending_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=trending_courses['Overall_Rating'].iloc[3].astype(str), value=trending_courses['Title'].iloc[3], delta=trending_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=trending_courses['Overall_Rating'].iloc[4].astype(str), value=trending_courses['Title'].iloc[4], delta=trending_courses['No_of_Ratings'].iloc[4].astype(str))

    temp = df.apply(lambda x: pd.Series(x['Category']),axis=1).stack().reset_index(level=1, drop=True)
    temp.name = 'category'
    df_cat = df.drop('Category', axis=1).join(temp)
    
    def make_toplist(genre, percentile=0.85):
        dataframe = df_cat[df_cat['category'] == genre]
        rating_counts = dataframe[dataframe['No_of_Ratings'].notnull()]['No_of_Ratings'].astype(float)
        rating_averages = dataframe[dataframe['Overall_Rating'].notnull()]['Overall_Rating'].astype(float)
        cg = rating_averages.mean()
        mg = rating_counts.quantile(percentile)
        col_list_g = ['Title', 'Overall_Rating', 'No_of_Ratings', 'category']
        qualified_g = dataframe[(dataframe['No_of_Ratings'] >= mg)
                               & (dataframe['No_of_Ratings'].notnull())
                               & (dataframe['Overall_Rating'].notnull())][col_list_g]
        qualified_g['No_of_Ratings'] = qualified_g['No_of_Ratings'].astype(float)
        qualified_g['Overall_Rating'] = qualified_g['Overall_Rating'].astype(float)

        qualified_g['weighted_rating_g'] = qualified.apply(lambda xg: (xg['No_of_Ratings']/(xg['No_of_Ratings']+mg) * xg['Overall_Rating']) + (mg/(mg+xg['No_of_Ratings']) * cg), axis=1)
        qualified_g = qualified_g.sort_values('weighted_rating_g', ascending=False).head(250)
        return qualified_g
    # top courses on development
    development_top_courses = make_toplist('Development').head(5)
    st.subheader('Courses for IT & Software Development')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=development_top_courses['Overall_Rating'].iloc[0].astype(str), value=development_top_courses['Title'].iloc[0], delta=development_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=development_top_courses['Overall_Rating'].iloc[1].astype(str), value=development_top_courses['Title'].iloc[1], delta=development_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=development_top_courses['Overall_Rating'].iloc[2].astype(str), value=development_top_courses['Title'].iloc[2], delta=development_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=development_top_courses['Overall_Rating'].iloc[3].astype(str), value=development_top_courses['Title'].iloc[3], delta=development_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=development_top_courses['Overall_Rating'].iloc[4].astype(str), value=development_top_courses['Title'].iloc[4], delta=development_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # office and productivity
    office_top_courses = make_toplist('Office Productivity').head(5)
    st.subheader('Courses for Office & Productivity')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=office_top_courses['Overall_Rating'].iloc[0].astype(str), value=office_top_courses['Title'].iloc[0], delta=office_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=office_top_courses['Overall_Rating'].iloc[1].astype(str), value=office_top_courses['Title'].iloc[1], delta=office_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=office_top_courses['Overall_Rating'].iloc[2].astype(str), value=office_top_courses['Title'].iloc[2], delta=office_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=office_top_courses['Overall_Rating'].iloc[3].astype(str), value=office_top_courses['Title'].iloc[3], delta=office_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=office_top_courses['Overall_Rating'].iloc[4].astype(str), value=office_top_courses['Title'].iloc[4], delta=office_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # marketing
    market_top_courses = make_toplist('Marketing').head(5)
    st.subheader('Courses on Marketing')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=market_top_courses['Overall_Rating'].iloc[0].astype(str), value=market_top_courses['Title'].iloc[0], delta=market_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=market_top_courses['Overall_Rating'].iloc[1].astype(str), value=market_top_courses['Title'].iloc[1], delta=market_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=market_top_courses['Overall_Rating'].iloc[2].astype(str), value=market_top_courses['Title'].iloc[2], delta=market_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=market_top_courses['Overall_Rating'].iloc[3].astype(str), value=market_top_courses['Title'].iloc[3], delta=market_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=market_top_courses['Overall_Rating'].iloc[4].astype(str), value=market_top_courses['Title'].iloc[4], delta=market_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # business
    business_top_courses = make_toplist('Business').head(5)
    st.subheader('Courses for Business')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=business_top_courses['Overall_Rating'].iloc[0].astype(str), value=business_top_courses['Title'].iloc[0], delta=business_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=business_top_courses['Overall_Rating'].iloc[1].astype(str), value=business_top_courses['Title'].iloc[1], delta=business_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=business_top_courses['Overall_Rating'].iloc[2].astype(str), value=business_top_courses['Title'].iloc[2], delta=business_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=business_top_courses['Overall_Rating'].iloc[3].astype(str), value=business_top_courses['Title'].iloc[3], delta=business_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=business_top_courses['Overall_Rating'].iloc[4].astype(str), value=business_top_courses['Title'].iloc[4], delta=business_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # design
    design_top_courses = make_toplist('Design').head(5)
    st.subheader('Courses on Design')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=design_top_courses['Overall_Rating'].iloc[0].astype(str), value=design_top_courses['Title'].iloc[0], delta=design_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=design_top_courses['Overall_Rating'].iloc[1].astype(str), value=design_top_courses['Title'].iloc[1], delta=design_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=design_top_courses['Overall_Rating'].iloc[2].astype(str), value=design_top_courses['Title'].iloc[2], delta=design_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=design_top_courses['Overall_Rating'].iloc[3].astype(str), value=design_top_courses['Title'].iloc[3], delta=design_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=design_top_courses['Overall_Rating'].iloc[4].astype(str), value=design_top_courses['Title'].iloc[4], delta=design_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # finance
    finance_top_courses = make_toplist('Finance & Accounting').head(5)
    st.subheader('Courses for Finance & Accounting')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=finance_top_courses['Overall_Rating'].iloc[0].astype(str), value=finance_top_courses['Title'].iloc[0], delta=finance_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=finance_top_courses['Overall_Rating'].iloc[1].astype(str), value=finance_top_courses['Title'].iloc[1], delta=finance_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=finance_top_courses['Overall_Rating'].iloc[2].astype(str), value=finance_top_courses['Title'].iloc[2], delta=finance_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=finance_top_courses['Overall_Rating'].iloc[3].astype(str), value=finance_top_courses['Title'].iloc[3], delta=finance_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=finance_top_courses['Overall_Rating'].iloc[4].astype(str), value=finance_top_courses['Title'].iloc[4], delta=finance_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # photo
    photo_top_courses = make_toplist('Photography & Video').head(5)
    st.subheader('Courses for Photo and Video Production')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=photo_top_courses['Overall_Rating'].iloc[0].astype(str), value=photo_top_courses['Title'].iloc[0], delta=photo_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=photo_top_courses['Overall_Rating'].iloc[1].astype(str), value=photo_top_courses['Title'].iloc[1], delta=photo_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=photo_top_courses['Overall_Rating'].iloc[2].astype(str), value=photo_top_courses['Title'].iloc[2], delta=photo_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=photo_top_courses['Overall_Rating'].iloc[3].astype(str), value=photo_top_courses['Title'].iloc[3], delta=photo_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=photo_top_courses['Overall_Rating'].iloc[4].astype(str), value=photo_top_courses['Title'].iloc[4], delta=photo_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # personality development
    perdev_top_courses = make_toplist('Personal Development').head(5)
    st.subheader('Courses for Personality Development')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=perdev_top_courses['Overall_Rating'].iloc[0].astype(str), value=perdev_top_courses['Title'].iloc[0], delta=perdev_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=perdev_top_courses['Overall_Rating'].iloc[1].astype(str), value=perdev_top_courses['Title'].iloc[1], delta=perdev_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=perdev_top_courses['Overall_Rating'].iloc[2].astype(str), value=perdev_top_courses['Title'].iloc[2], delta=perdev_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=perdev_top_courses['Overall_Rating'].iloc[3].astype(str), value=perdev_top_courses['Title'].iloc[3], delta=perdev_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=perdev_top_courses['Overall_Rating'].iloc[4].astype(str), value=perdev_top_courses['Title'].iloc[4], delta=perdev_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # music
    music_top_courses = make_toplist('Music').head(5)
    st.subheader('Courses for Music')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=music_top_courses['Overall_Rating'].iloc[0].astype(str), value=music_top_courses['Title'].iloc[0], delta=music_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=music_top_courses['Overall_Rating'].iloc[1].astype(str), value=music_top_courses['Title'].iloc[1], delta=music_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=music_top_courses['Overall_Rating'].iloc[2].astype(str), value=music_top_courses['Title'].iloc[2], delta=music_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=music_top_courses['Overall_Rating'].iloc[3].astype(str), value=music_top_courses['Title'].iloc[3], delta=music_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=music_top_courses['Overall_Rating'].iloc[4].astype(str), value=music_top_courses['Title'].iloc[4], delta=music_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # academic
    teaching_top_courses = make_toplist('Teaching & Academics').head(5)
    st.subheader('Courses for Academics')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=teaching_top_courses['Overall_Rating'].iloc[0].astype(str), value=teaching_top_courses['Title'].iloc[0], delta=teaching_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=teaching_top_courses['Overall_Rating'].iloc[1].astype(str), value=teaching_top_courses['Title'].iloc[1], delta=teaching_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=teaching_top_courses['Overall_Rating'].iloc[2].astype(str), value=teaching_top_courses['Title'].iloc[2], delta=teaching_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=teaching_top_courses['Overall_Rating'].iloc[3].astype(str), value=teaching_top_courses['Title'].iloc[3], delta=teaching_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=teaching_top_courses['Overall_Rating'].iloc[4].astype(str), value=teaching_top_courses['Title'].iloc[4], delta=teaching_top_courses['No_of_Ratings'].iloc[4].astype(str))
    # health
    health_top_courses = make_toplist('Health & Fitness').head(5)
    st.subheader('Courses on Health')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label=health_top_courses['Overall_Rating'].iloc[0].astype(str), value=health_top_courses['Title'].iloc[0], delta=health_top_courses['No_of_Ratings'].iloc[0].astype(str))
    col2.metric(label=health_top_courses['Overall_Rating'].iloc[1].astype(str), value=health_top_courses['Title'].iloc[1], delta=health_top_courses['No_of_Ratings'].iloc[1].astype(str))
    col3.metric(label=health_top_courses['Overall_Rating'].iloc[2].astype(str), value=health_top_courses['Title'].iloc[2], delta=health_top_courses['No_of_Ratings'].iloc[2].astype(str))
    col4.metric(label=health_top_courses['Overall_Rating'].iloc[3].astype(str), value=health_top_courses['Title'].iloc[3], delta=health_top_courses['No_of_Ratings'].iloc[3].astype(str))
    col5.metric(label=health_top_courses['Overall_Rating'].iloc[4].astype(str), value=health_top_courses['Title'].iloc[4], delta=health_top_courses['No_of_Ratings'].iloc[4].astype(str))
