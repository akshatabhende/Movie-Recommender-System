#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System

# ![](https://i.kym-cdn.com/entries/icons/original/000/026/825/movies-tiles.jpg)

# # Recommender System

# In our Movie recommender system, we will be implementing Simple, Content and Collaborative filtering. Combining all these models we will be building the final model i.e. Hybrid filtering. Here, we will be using two dataset Full Dataset and Small Dataset.
# 
# - Full Dataset:- Is made up of 26,000,000 ratings and 750,000 tag applications submitted by 270,000 users to 45,000 movies. Includes tag genome data across 1,100 tags, with 12 million related ratings.
# 
# - Small Dataset:- Comprises of 100,000 reviews and 1,300 tag apps added by 700 users to 9,000 movies.
# 
# 
# We will build our Simple Recommender using Full Dataset movies, while the small dataset will be used by all personalized recommendation systems like Collaborative Recommender System, Content Based Recommender and Hybrid Recommender (due to the computing power I possess being very limited). Let us build our simple recommendation system as a first step.

# # Simple Recommender
# 
# The Simple Recommender provides every user with generalized recommendations based on movie popularity and sometimes genre. The underlying principle behind this recommender is that movies that are more popular and more critically acclaimed would be more likely to be enjoyed by the average viewer.  This model does not offer personalized recommendations based on the user.
# 
# This model's implementation is highly trivial. All we need to do is sort our movies based on ratings and popularity and display our list of top films. As an added step, to get the top movies of a specific genre, we can pass into a genre argument.

# In[1]:


import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')

movie_data = pd. read_csv("movies_metadata.csv")
movie_data.head(10)


# In[2]:


movie_data.shape


# In[3]:


movie_data.describe()


# In[4]:


movie_data.info()


# We will be dropping few features that are not useful for our recommender system

# In[5]:


# Dropping unnecessary columns
movie_data.drop(['homepage', 'adult', 'overview', 'poster_path', 'tagline', 'video'], axis=1)


# ### Features of our dataset
# 
# 
# * **belongs_to_collection:** Name of the franchise the movie belongs to.
# * **budget:** The budget of the movie in dollars.
# * **genres:** A stringified list of dictionaries that list out all the genres associated with the movie.
# * **id:** The ID of the move.
# * **imdb_id:** The IMDB ID of the movie.
# * **original_language:** The language in which the movie was originally shot in.
# * **original_title:** The original title of the movie.
# * **popularity:** The Popularity Score assigned by TMDB.
# * **production_companies:** A stringified list of production companies involved with the making of the movie.
# * **production_countries:** A stringified list of countries where the movie was shot/produced in.
# * **release_date:** Theatrical Release Date of the movie.
# * **revenue:** The total revenue of the movie in dollars.
# * **runtime:** The runtime of the movie in minutes.
# * **spoken_languages:** A stringified list of spoken languages in the film.
# * **status:** The status of the movie (Released, To Be Released, Announced, etc.)
# * **title:** The Official Title of the movie.
# * **vote_average:** The average rating of the movie.
# * **vote_count:** The number of votes by users, as counted by TMDB.

# Genre in our dataset is in form of list of dictionaries.So we need to convert it into list form 

# In[6]:


import ast
from ast import literal_eval

#converted list of dictionaries to list
movie_data['genres'] = movie_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] 
                                                                       if isinstance(x, list) else [])


# We will be using the TMDB Ratings to come up with our **Top Movies Chart.** For that we will use IMDB's *weighted rating* formula to construct top chart. Mathematically, it is represented as follows:
# 
# Weighted Rating (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$
# 
# where,
# * *v* is the number of votes for the movie
# * *m* is the minimum votes required to be listed in the chart
# * *R* is the average rating of the movie
# * *C* is the mean vote across the whole report
# 
# The next step is to determine an appropriate value for *m*, the minimum votes required to be listed in the chart. We will use **95th percentile** as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.
# 
# The next step is to decide a suitable value for m, the minimum votes needed to appear in the chart. We are going to be using 95th percentile as our limit. In other words, for a movie to be included in the charts, it must have more votes in the list than at least 95 percent of the movies.
# 
# Building our overall Top 250 Chart and we will then define a function to build charts for a particular genre.

# In[7]:


vote_count = movie_data[movie_data['vote_count'].notnull()]['vote_count'].astype('int')
vote_average = movie_data[movie_data['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_average.mean()
C


# In[8]:


m = vote_count.quantile(0.95)
m


# In[9]:


movie_data['year'] = pd.to_datetime(movie_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[10]:


top_movies = movie_data[(movie_data['vote_count'] >= m) & (movie_data['vote_count'].notnull()) & (movie_data['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
top_movies['vote_count'] = top_movies['vote_count'].astype('int') # converted float to integer
top_movies['vote_average'] = top_movies['vote_average'].astype('int')
top_movies.shape


# Therefore a movie has to have at least **434 votes** on TMDB to qualify to be considered for the list. We also see that on a scale of 10, the average rating for a movie on TMDB is **5.244**. **2274** movies are qualified to be on our list.

# In[11]:


#applying the IMDB's weighted rating formula 
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[12]:


top_movies['weighted_rating'] = top_movies.apply(weighted_rating, axis=1)


# In[13]:


top_movies = top_movies.sort_values('weighted_rating', ascending=False).head(250)


# ## Top Movies

# In[14]:


top_movies.head(15)


# We observe that at the very top of our list are three Christopher Nolan movies, **Inception**, **The Dark Knight** and **Interstellar**. A strong bias of TMDB users towards specific genres and directors is also seen in the table.
# 
# Let us create our feature now, which constructs charts for different genres. 

# In[15]:


s = movie_data.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
genre_movie = movie_data.drop('genres', axis=1).join(s)


# In[16]:


genre_movie.head()


# In[17]:


def simple_recommender(genre, percentile=0.95):
    movie_data2 = genre_movie[genre_movie['genre'] == genre]
    vote_count = movie_data2[movie_data2['vote_count'].notnull()]['vote_count'].astype('int')
    vote_average = movie_data2[movie_data2['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_average.mean()
    m = vote_count.quantile(percentile)
    
    top_movies = movie_data2[(movie_data2['vote_count'] >= m) & (movie_data2['vote_count'].notnull()) & (movie_data2['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    top_movies['vote_count'] = movie_data2['vote_count'].astype('int')
    top_movies['vote_average'] = top_movies['vote_average'].astype('int')
    
    top_movies['wr'] = top_movies.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    top_movies = top_movies.sort_values('wr', ascending=False).head(250)
    
    return top_movies


# Let's see our system in practice by showing the Top 15 Romance Movies (Romance hardly featured anywhere in our Generic Top Chart despite being one of the most common genres of movies).

# ## Top Romance Movies

# In[18]:


simple_recommender('Romance').head(15)


# The top romance movie according to our metrics is **Forrest Gump**.Followed by bollywood's movie **Dilwale Dulhania Le Jayenge**.

# ### Data Pre Processing for Persolinzed Recommender Engines
# 
# To build our personalized recommender, we will need to merge our current dataset with the crew and the keyword datasets. Let us prepare this data as our first step.
# Also, as mentioned in the introduction, we will be using a subset of all the movies available to us due to limiting computing power available.

# In[19]:


credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
links = pd.read_csv('links.csv')
links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[20]:


credits.info()
keywords.info()


# The **credits.csv** file contains the cast and crew information of the movie set, while **keywords.csv** contains the keywords used to describe the movie

# In[21]:


movie_data = movie_data.drop([19730, 29503, 35587])
movie_data['id'] = movie_data['id'].astype('int')

small_data = movie_data[movie_data['id'].isin(links)]
small_data.shape

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')


# In[22]:


# mergeing credits and keywords dataset with our small dataset
small_data = small_data.merge(credits, on='id')
small_data = small_data.merge(keywords, on='id')


# In[23]:


small_data.shape


# In[24]:


small_data.info()


# We can observe that cast,crew and keywords columns have been added to our small dataset

# In[25]:


small_data


# Now we've got our cast, crew, genres and credits in one data frame. Let's wrangle this a bit more with the following axioms: 
# 
# 1) **Crew** :We're only going to select the director from the team as our element, as the others don't add too much to the movie taste.
# 
# 2) **Cast**: Choosing Cast is a little trickier. Lesser recognized actors and small roles have no real impact on the perception of a film by people. So we only have to pick the main characters and their respective actors. We will randomly pick the top 3 actors appearing in the credits list.

# In[26]:


#Converting columns which are in the form of list of dictionaires to list

small_data['cast'] = small_data['cast'].apply(literal_eval)
small_data['crew'] = small_data['crew'].apply(literal_eval)
small_data['keywords'] = small_data['keywords'].apply(literal_eval)
small_data['cast_size'] = small_data['cast'].apply(lambda x: len(x))
small_data['crew_size'] = small_data['crew'].apply(lambda x: len(x))


# In[27]:


#defining a function to get directors from the crew list

def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[28]:


small_data['director'] = small_data['crew'].apply(director)
small_data['cast'] = small_data['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
small_data['cast'] = small_data['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
small_data['keywords'] = small_data['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
small_data['cast'] = small_data['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
small_data['director'] = small_data['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
small_data['director'] = small_data['director'].apply(lambda x: [x,x])


# In[29]:


small_data = small_data.reset_index()
titles = small_data['title']
indices = pd.Series(small_data.index, index=small_data['title'])


# We must do a little pre-processing of our keywords before we put them to good use. We measure the frequency counts of each keyword appearing in the dataset as a first step.

# In[30]:


c = small_data.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
c.name = 'keyword'
c = c.value_counts()
c[:5]


# Keywords occur in frequencies of between 1 and 610. We have no use for keywords which only occur once. And these can be eliminated . We will finally convert every word to its stem so that words like Dogs and Dog are considered the same.

# In[31]:


c = c[c > 1]


# In[32]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english') #reduces to root word
stemmer.stem('running')


# In[33]:


def keywords(x):
    words = []
    for i in x:
        if i in c:
            words.append(i)
    return words


# In[34]:


small_data['keywords'] = small_data['keywords'].apply(keywords)
small_data['keywords'] = small_data['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
small_data['keywords'] = small_data['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# Now that we are done with data pre processing we can move on to our next recommender engine which is **"Content Based Recommender System"**

# # Content Based Recommender System

# <img src="https://miro.medium.com/max/1026/1*BME1JjIlBEAI9BV5pOO5Mg.png" alt="Drawing" style="width: 300px;"/>
# 

# We suffered from some major limitations in the simple recommendation engine. It offers the same suggestion to everyone, no matter the user's personal taste. If our top 15 list were to be looked at by a person who loves romantic films (and hates action), he/she would probably not like any of the films.
# 
# To personalise our recommendations more, We are going to build an engine that takes in a movie that a user currently likes as input. Then it analyzes the contents (storyline, genre, cast, director etc.) of the movie to find out other movies which have similar content. Then it ranks similar movies according to their similarity scores and recommends the most relevant movies to the user.Since we will be using contents to build this engine, this is also known as **Content Based Filtering.**
# 
# 

# In[35]:


small_data.columns


# In this dataset, we observe that a movie has a lot of additional detail. We don't need them all. So as our feature set (the "Content" of the movie), we choose keywords, cast, genres and director column to use.
# 
# To do that we will combined all those features in one column

# In[36]:


small_data['combined_features'] = small_data['keywords'] + small_data['cast'] + small_data['director'] + small_data['genres']
small_data['combined_features'] = small_data['combined_features'].apply(lambda x: ' '.join(x))
small_data['combined_features']


# Now, we need to represent the combined features as vectors. So we will be using TfidfVectorizer() class from sklearn.feature_extraction.text library to do that.

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(small_data['combined_features'])


# #### Cosine Similarity
# 
# We will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. Mathematically, it is defined as follows:
# 
# $cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $
# 
# Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. Therefore, we will use sklearn's **linear_kernel** instead of cosine_similarities since it is much faster.

# In[38]:


from sklearn.metrics.pairwise import linear_kernel
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[39]:


cosine_similarity


# We now have a pairwise cosine similarity matrix for all the movies in our dataset. The next step is to write a function that returns the most similar movies based on the cosine similarity score.

# In[40]:


def get_recommendations(title):
    index = indices[title]
    simlarity_scores = list(enumerate(cosine_similarity[index]))
    simlarity_scores = sorted(simlarity_scores, key=lambda x: x[1], reverse=True)
    simlarity_scores = simlarity_scores[1:31]
    movie_indices = [i[0] for i in simlarity_scores]
    return titles.iloc[movie_indices]


# In[41]:


get_recommendations('The Dark Knight').head(10)


# The recommendations seem to have acknowledged other Christopher Nolan movies (due to the director's heavy weighting) and placed them as top recommendations. 

# In[42]:


get_recommendations('Pulp Fiction').head(10)


# **Popularity and Ratings**
# 
# One thing we see in our recommendation system is that, regardless of ratings and popularity, it recommends movies. It is true that **Batman and Robin** compared to **The Dark Knight** have a lot of similar characters, but it was a bad movie that should not be recommended to anyone.
# 
# To achieve that, we will add a function to remove terrible movies and return movies which are successful and have had a strong critical response.
# 
# Based on similarity scores, We will take the top 50 movies and measure the vote for the 75th percentile movie. Then using this as the $m$ value, we will use the IMDB formula to calculate the weighted rating of each movie, as we did in the Simple Recommender System.

# In[43]:


def content_based_recommendations(title):
    index = indices[title]
    similarity_scores = list(enumerate(cosine_similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:51]
    movie_indices = [i[0] for i in similarity_scores]
    
    movies = small_data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.75)
    top_movies = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    top_movies['vote_count'] = top_movies['vote_count'].astype('int')
    top_movies['vote_average'] = top_movies['vote_average'].astype('int')
    top_movies['weighted_rating'] = top_movies.apply(weighted_rating, axis=1)
    top_movies = top_movies.sort_values('weighted_rating', ascending=False).head(10)
    return top_movies


# In[44]:


content_based_recommendations('The Dark Knight')


# We can clearly observe that movies recommended to us now have good rating and strong critical response. Also the earlier recommendation of **Batman and Robin** is not present as it has terrible vote_average ,i.e, 4.

# In[45]:


content_based_recommendations('Pulp Fiction')


# # Collaborative Filtering Recommender System
# 
# <img src="https://miro.medium.com/max/345/1*x8gTiprhLs7zflmEn1UjAQ.png" alt="Drawing" style="width: 400px;"/>
# 
# Our content-based recommender suffers from some extreme constraints. It can only recommend movies that are **close** to a certain movie. That is, it is not capable of capturing tastes across genres and making recommendations.
# 
# Also, it is does not really gives us personal recommendations as it doesn't take into consideration the users personal taste and biases of that user.
# 
# In this recommender system, We will therefore use a technique called **Collaborative Filtering** to make more personalized suggestions to Movie Lovers. Collaborative filtering is based on the concept that it is possible to consider a users similar to me to predict which movies I would like which those users have already watched, but I have not.
# 
# We will be implementing two different technique for Collabrative Filtering Recommender Engine:
#  
#  1) Using Scikit Learn's Surprise Library
#  
#  2) Using Pearson's Correlation

# ### Using Scikit Learn's Surprise Library
# 
# We will be using the **Surprise** library that uses extremely powerful algorithms like **Singular Value Decomposition (SVD)** to minimise RMSE (Root Mean Square Error) and give great recommendations.

# In[46]:


ratings = pd.read_csv("ratings.csv")
ratings.head()


# In[47]:


ratings.rating.value_counts()


# From here we can see rating of 4.0 has highest value counts. This means more people rated the movie 4.0.

# In[48]:


ratings.info()


# In[49]:


ratings.isnull().sum()


# In[50]:


from surprise import Reader, Dataset

reader = Reader()

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# In[51]:


from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.25)


# In[52]:


from surprise import SVD, accuracy
svd = SVD()

from surprise.model_selection import cross_validate
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# For our case, we get a mean Root Mean Sqaure Error of 0.8975 which is more than good enough. Let us now train the dataset and do some predictions.

# In[53]:


svd.fit(trainset)


# In[54]:


predictions = svd.test(testset)


# Let us pick user 30 and check the ratings s/he has given.

# In[55]:


ratings[ratings['userId'] == 30]


# In[56]:


svd.predict(30, 302, 3)


#  We get an average prediction of 3.941 for film with ID 302. One surprising aspect of this recommender method is that what the film is (or what it contains) doesn't matter. It operates solely on the basis of an allocated film ID, and attempts to predict ratings based on how the other users predicted the film.

# ### Collaborative Filtering using Pearson Correlation
# 

# Pearson’s Coefficient of correlation is used to get statistical relationship between two variables. In collaborative filtering recommender system, we are going to use Pearson’s correlation to get us the correlation coefficient between similar movies which will help us to get more personalized movie recommendation.

# In[57]:


ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'], axis = 1)
ratings.head()


# In[58]:


user_ratings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
user_ratings.head()


# In[59]:


#removes movies which has less than 10 users who rated it and fill Nan with 0
user_ratings = user_ratings.dropna(thresh=10,axis=1).fillna(0)


# In[60]:


user_ratings


# In[61]:


item_similarity = user_ratings.corr(method='pearson')
item_similarity.head()


# In[62]:


def collaborative_recommender(movie_name,user_rating):
    similarity_score = item_similarity[movie_name]*(user_rating-2.5)
    similarity_score = similarity_score.sort_values(ascending=False)
    
    return similarity_score


# In[63]:


collaborative_recommender("17 Again (2009)", 5). head(10)


# In[64]:


my_rating = [("17 Again (2009)", 5), ("101 Dalmatians (1996)", 3), ("(500) Days of Summer (2009)",2)]
similar_movies = pd.DataFrame()

for movie,rating in my_rating:
    similar_movies = similar_movies.append(collaborative_recommender(movie,rating), ignore_index = True)
    
similar_movies
similar_movies.sum().sort_values(ascending=False).head(10)


# # Hybrid Recommender System

# Hybrid recommender system will brings together the techniques we have implemented in the engines based on the content based and the collaborative filter recommender system. Consideration we need to take to implement hybrid system are as follows: 
# 
# - **Input**: User ID and the Movie 
# 
# - **Output Title**: Related films sorted by that particular user based on predicted ratings.

# In[65]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[66]:


id_map = pd.read_csv('links.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(small_data[['title', 'id']], on='id').set_index('title')

#Build ID to title mappings
indices_map = id_map.set_index('id')


# We will build a hybrid function which will combine the techniques used in content based recommender system and surprise library based Collaborative filtering recommender system.

# In[67]:


def hybrid(userId, title):
    
    #Extract the cosine_sim index of the movie
    index = indices[title]
    
    #Extract the TMDB ID of the movie
    tmdbId = id_map.loc[title]['id']
    
    #Extract the movie ID internally assigned by the dataset
    movie_id = id_map.loc[title]['movieId']
    
    #Extract the similarity scores and their corresponding index for every movie from the cosine_sim matrix
    similiarity_scores = list(enumerate(cosine_similarity[int(index)]))
    
    #Sort the index, score in decreasing order of similarity scores
    similiarity_scores = sorted(similiarity_scores, key=lambda x: x[1], reverse=True)
    
    #Select the top 25 tuples, excluding the first 
    similiarity_scores = similiarity_scores[1:26]
    
    #Store the cosine_sim indices of the top 25 movies in a list
    movie_indices = [i[0] for i in similiarity_scores]
    
    #Extract the metadata of the aforementioned movies
    movies = small_data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    
    #Compute the predicted ratings using the SVD filter
    movies['predicted_ratings'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    
    #Sort the movies in decreasing order of predicted rating
    movies = movies.sort_values('predicted_ratings', ascending=False)
    
    #Return the top 10 movies as recommendations
    return movies.head(10)


# In[68]:


hybrid(1, 'The Dark Knight')


# In[69]:


hybrid(500, 'The Dark Knight')


# We see that we get different suggestions for different users for our hybrid recommender while the film is the same. Our reviews are also more personalised and tailored to individual users.

# ## Conclusion:
# Thus, We have built four different recommendation system based on different ideas and algorithms. They are as follows:
# 
# 1)**Simple Recommender**: This system used TMDB Vote Count and Vote Averages overall to create Top Movies Charts, in general and for a particular genre. The IMDB Weighted Ranking System was used to determine ratings for which final sorting was performed.
# 
# 2)**Content Based Recommender**: We have used contents (storyline, genre, cast, director etc.) of the movie to find out other movies which have similar content.It ranks similar movies according to their similarity scores and recommends the most relevant movies to the user.
# 
# 3)**Collaborative Filtering Recommender:** We have build two collaborative engines;one that uses the powerful Surprise Library to create a collaborative filter based on a decomposition of a single value. The obtained RMSE was less than 1 and the engine received approximate ratings for a given user and film. And other engine that uses pearson's coefficient algorithm to recommend movies that was liked by user with similar taste.
# 
# 4)**Hybrid Recommender:** We put together ideas from content and collaborative filtering to create an algorithm that gave recommendations of movies to a specific user based on the average ratings it had internally measured for that user.
