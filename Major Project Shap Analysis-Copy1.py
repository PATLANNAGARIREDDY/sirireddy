#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install -q numpy pandas matplotlib plotly wordcloud scikit-learn')


# In[2]:


import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import save_npz
import warnings
warnings.filterwarnings('ignore')


# In[3]:


Movie_data = pd.read_csv("Movie_Dataset 123.csv")
Movie_data.head()


# In[4]:


Movie_data.info()


# In[5]:


Movie_data.isnull().sum()


# In[6]:


Movie_data.fillna('', inplace=True)


# In[7]:


Movie_data.describe(include='all').T


# In[8]:


import pandas as pd
print(Movie_data.columns)


# In[9]:


movie_counts = Movie_data['Certificate'].value_counts().sort_index()
fig = go.Figure(data=go.Bar(x=movie_counts.index, y=movie_counts.values))
fig.update_layout(
    plot_bgcolor='rgb(17, 17, 17)',  
    paper_bgcolor='rgb(17, 17, 17)',  
    font_color='white', 
    title='Certificate of the movies ',  
    xaxis=dict(title='Year'),  
    yaxis=dict(title='Number of Movies')
)
fig.update_traces(marker_color='red')
fig.show()


# In[10]:


movie_type_counts = Movie_data['Rating'].value_counts()

fig = go.Figure(data=go.Pie(labels=movie_type_counts.index, values=movie_type_counts.values))

fig.update_layout(
    plot_bgcolor='rgb(17, 17, 17)',  
    paper_bgcolor='rgb(17, 17, 17)', 
    font_color='white',  
    title=' Movies Ratings',
)
fig.update_traces(marker=dict(colors=['blue']))
fig.show()


# In[11]:


top_countries = Movie_data['Gener'].value_counts().head(10)

fig = px.treemap(names=top_countries.index, parents=["" for _ in top_countries.index], values=top_countries.values)

fig.update_layout(
    plot_bgcolor='rgb(17, 17, 17)',  
    paper_bgcolor='rgb(17, 17, 17)', 
    font_color='white',  
    title='Movie Gener',
)
fig.show()


# In[12]:


ratings       = list(Movie_data['No of Ratings'].value_counts().index)
rating_counts = list(Movie_data['No of Ratings'].value_counts().values)

fig = go.Figure(data=[go.Bar(
    x=ratings,
    y=rating_counts,
    marker_color='#E50914'
)])

fig.update_layout(
    title='Movie Ratings Distribution',
    xaxis_title='Rating',
    yaxis_title='Count',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0.7)',
    font=dict(
        color='white'
    )
)

fig.show()


# In[13]:


ratings       = list(Movie_data['Runtime'].value_counts().index)
rating_counts = list(Movie_data['Runtime'].value_counts().values)

fig = go.Figure(data=[go.Bar(
    x=ratings,
    y=rating_counts,
    marker_color='#E50914'
)])

fig.update_layout(
    title='Movie Durations Distribution',
    xaxis_title='Rating',
    yaxis_title='Count',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0.7)',
    font=dict(
        color='white'
    )
)

fig.show()


# In[15]:


Movie_data


# In[16]:


new_data =  Movie_data[[ 'Movie', 'Year ', 'Certificate', 'Gener', 'Overview',
       'Runtime', 'Rating', 'No of Ratings', 'Hero', 'Heroin', 'Director',
       'Music Director']]
new_data.set_index('Movie', inplace=True)
new_data.head()


# In[17]:


class TextCleaner:
    def separate_text(self, texts):
        unique_texts = set()
        for text in texts.split(','):
            unique_texts.add(text.strip().lower())
        return ' '.join(unique_texts)

    def remove_space(self, texts):
        return texts.replace(' ', '').lower()

    def remove_punc(self, texts):
        texts = texts.lower()
        texts = texts.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(texts.split())

    def clean_text(self, texts):
        texts = self.separate_text(texts)
        texts = self.remove_space(texts)
        texts = self.remove_punc(texts)
        return texts


# In[18]:


cleaner = TextCleaner()


# In[19]:


import numpy as np 

new_data['Rating'] = new_data['Rating'].apply(lambda x: cleaner.remove_punc(x) if isinstance(x, str) else np.nan)

new_data['Rating'].fillna(value="", inplace=True)

new_data['Rating'] = new_data['Rating'].apply(lambda x: x.lower())

new_data['No of Ratings'] = new_data['No of Ratings'].apply(lambda x: cleaner.remove_space(str(x)) if isinstance(x, int) else x)

new_data['Rating'] = new_data['Rating'].apply(lambda x: x.lower())


# In[20]:



new_data['Certificate'] = new_data['Certificate'].apply(cleaner.separate_text)
new_data['Overview'] = new_data['Overview'].apply(cleaner.remove_space)


new_data['Runtime'] = new_data['Runtime'].apply(lambda x: cleaner.separate_text(x) if isinstance(x, str) else x)

new_data['Rating'] = new_data['Rating'].apply(cleaner.remove_punc)
new_data['No of Ratings'] = new_data['No of Ratings'].apply(cleaner.remove_space)

new_data.head()


# In[21]:


new_data['BoW'] = new_data.apply(lambda row: ' '.join(row.dropna().astype(str).values), axis=1)
new_data.drop(new_data.columns[:-1], axis=1, inplace=True)
new_data.head()


# In[22]:


tfid = TfidfVectorizer()
tfid_matrix = tfid.fit_transform(new_data['BoW'])


# In[23]:


cosine_sim = cosine_similarity(tfid_matrix, tfid_matrix)
cosine_sim


# In[24]:


cosine_sim


# In[25]:


np.save('tfidf_matrix.npy', tfid_matrix)
np.save('cosine_sim_matrix.npy', cosine_sim)


# In[26]:


with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfid, f)


# In[27]:


final_data = Movie_data[['Movie', 'Gener']]
final_data.head()


# In[28]:


final_data.to_csv('movie_data.csv',index=False)


# In[29]:


import re
class FlixHub:
    def __init__(self, df, cosine_sim):
        self.df = df
        self.cosine_sim = cosine_sim
    
    def recommendation(self, title, total_result=5, threshold=0.5):
        idx = self.find_id(title)
        self.df['similarity'] = self.cosine_sim[idx]
        sort_df = self.df.sort_values(by='similarity', ascending=False)[1:total_result+1]
        
        movies = sort_df['Movie']
       
        
        similar_movies = []
       
        
        for i, movie in enumerate(movies):
            similar_movies.append('{}. {}'.format(i+1, movie))
        
       
        return similar_movies

    def find_id(self, name):
        for index, string in enumerate(self.df['Movie']):
            if re.search(name, string):
                return index
        return -1


# In[30]:


flix_hub = FlixHub(final_data, cosine_sim)
movies = flix_hub.recommendation('Bangarraju', total_result=10, threshold=0.5)

print('Similar Movie(s) list:')
for movie in movies:
    print(movie)


# In[31]:


flix_hub = FlixHub(final_data, cosine_sim)
movies = flix_hub.recommendation('Action,Drama', total_result=2, threshold=0.5)

print('Similar Movie(s) list:')
for movie in movies:
    print(movie)


# In[53]:


import re

class FlixHub:
    def __init__(self, df, cosine_sim):
        self.df = df
        self.cosine_sim = cosine_sim

    def recommendation(self, title, total_result=5, threshold=0.5):
        idx = self.find_id(title)
        self.df['similarity'] = self.cosine_sim[idx]
        sort_df = self.df.sort_values(by='similarity', ascending=False)[1:total_result+1]

        movies = sort_df['Movie']

        similar_movies = []

        for i, movie in enumerate(movies):
            similar_movies.append('{}. {}'.format(i+1, movie))

        return similar_movies

    def find_id(self, name):
        for index, string in enumerate(self.df['Gener']):
            if re.search(name, string):
                return index
        return -1

flix_hub = FlixHub(final_data, cosine_sim)

user_input = input("Enter the Genre: ")
movies = flix_hub.recommendation(user_input, total_result=10, threshold=0.5)
print('Similar Movie(s) list:')
for movie in movies:
    print(movie)


# In[1]:


pip install shap


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
shap.initjs()


# In[3]:


get_ipython().system('pip install xgboost')


# In[ ]:





# In[4]:


data=pd.read_csv("Movie_Dataset 123.csv")
print(len(data))
data.head()


# In[5]:


plt.scatter(data['Runtime'],data['Rating'])
plt.ylabel('Rating',size=20)
plt.xlabel('Runtime',size=20)


# In[6]:


cont=['Movie','Certificate','Gener','Overview','Runtime','Rating','No of Ratings','Hero','Heroin','Director','Music Director']
corr_matrix=pd.DataFrame(data[cont],columns=cont).corr()
sns.heatmap(corr_matrix,
           cmap='coolwarm',
           center = 0,
           annot=True,
           fmt='.5g')


# In[7]:


Y = data['Rating']
X = data[["Movie","Certificate","Gener","Overview","Runtime","No of Ratings","Hero","Heroin","Director","Music Director"]]


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


Y_pred = model.predict(X)
plt.figure(figsize=(5,5))
plt.scatter(Y,Y_pred)
plt.plot([0, 30],
        [0, 30],
        color = 'r',
        linestyle = '-',
        linewidth = 2)

plt.ylabel('Predicted', size = 20)
plt.xlabel('Actual', size = 20)


# In[28]:


import shap


# In[29]:


#STANDARD SHAP VALUES
explainer = shap.Explainer(model)
shap_values = explainer(X)


# In[31]:


import numpy as np


# In[32]:


np.shape(shap_values.values)


# In[33]:


#WATERFALL PLOT
shap.plots.waterfall(shap_values[0])


# In[34]:


shap.plots.waterfall(shap_values[1], max_display=5)


# In[36]:


#SHAP FOR BINARY TARGET VALUES
#y_bin = [1 if y_>10 else 0 for y_ in Y]


# In[38]:


shap.initjs()


# In[39]:


#FORCE PLOT
shap.plots.force(shap_values[0])


# In[40]:


# STACKED FORCE PLOT
shap.plots.force(shap_values[0:100])


# In[41]:


#ABSOLUTE MEAN SHAP PLOT
shap.plots.bar(shap_values)


# In[42]:


#BEESWARM PLOT
shap.plots.beeswarm(shap_values)


# In[45]:


#DEPENDENCY PLOT
shap.plots.scatter(shap_values[:,"Gener"])


# In[47]:


shap.plots.scatter(shap_values[:,"Gener"],
                    color = shap_values[:,"Runtime"])


# In[ ]:




