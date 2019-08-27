import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle



df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data


user_interaction = df['email'].value_counts().values


# Fill in the median and maximum number of user_article interactios below

median_val = 3.0 # 50% of individuals interact with ____ number of articles or fewer.
max_views_by_user = 364 # The maximum number of user-article interactions by any 1 user is ______.


# In[138]:



# `2.` Explore and remove duplicate articles from the **df_content** dataframe.  

# In[139]:





# Remove any rows that have the same article_id - only keep the first
df_content = df_content[~df_content.duplicated(subset=['article_id'],keep='first')]


# `3.` Use the cells below to find:
# 
# **a.** The number of unique articles that have an interaction with a user.  
# **b.** The number of unique articles in the dataset (whether they have any interactions or not).<br>
# **c.** The number of unique users in the dataset. (excluding null values) <br>
# **d.** The number of user-article interactions in the dataset.

# In[141]:




unique_articles = df['article_id'].nunique() # The number of unique articles that have at least one interaction
total_articles = df_content['article_id'].nunique() # The number of unique articles on the IBM platform
unique_users =  5148.0 # The number of unique users
user_article_interactions = len(df) # The number of user-article interactions


# `4.` Use the cells below to find the most viewed **article_id**, as well as how often it was viewed.  After talking to the company leaders, the `email_mapper` function was deemed a reasonable way to map users to ids.  There were a small number of null values, and it was found that all of these null values likely belonged to a single user (which is how they are stored using the function below).

# In[146]:




# In[147]:


most_viewed_article_id = '1429.0' # The most viewed article in the dataset as a string with one value following the decimal 
max_views = 937 # The most viewed article in the dataset was viewed how many times?


# In[148]:


## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header



# In[149]:




# ### <a class="anchor" id="Rank">Part II: Rank-Based Recommendations</a>
# 
# Unlike in the earlier lessons, we don't actually have ratings for whether a user liked an article or not.  We only know that a user has interacted with an article.  In these cases, the popularity of an article can really only be based on how often an article was interacted with.
# 
# `1.` Fill in the function below to return the **n** top articles ordered with most interactions as the top. Test your function using the tests below.

# In[151]:


top_art = df['article_id'].value_counts().head().index.tolist()



# In[152]:





# In[153]:


def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Your code here
    ids = get_top_article_ids(n)
    top_articles = []
    
    for idd in ids:
        top_articles.append(df[df['article_id']==idd]['title'].iloc[0])
        
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Your code here
    top_articles = df['article_id'].value_counts().head(n).index.tolist()
    return top_articles # Return the top article ids


# In[154]:


# ### <a class="anchor" id="User-User">Part III: User-User Based Collaborative Filtering</a>
# 
# 
# `1.` Use the function below to reformat the **df** dataframe to be shaped with users as the rows and articles as the columns.  
# 
# * Each **user** should only appear in each **row** once.
# 
# 
# * Each **article** should only show up in one **column**.  
# 
# 
# * **If a user has interacted with an article, then place a 1 where the user-row meets for that article-column**.  It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.  
# 
# 
# * **If a user has not interacted with an item, then place a zero where the user-row meets for that article-column**. 
# 
# Use the tests to make sure the basic structure of your matrix matches what is expected by the solution.

# In[156]:


# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Fill in the function here
    df = df[~df.duplicated(subset=['user_id','article_id'])]
    df = df.pivot(index='user_id',columns='article_id',values='title')
    df.replace(to_replace=np.nan,value=0,inplace=True)
    user_item = df.replace(to_replace=r'[A-Za-z]',value=1,regex=True)
    
    return user_item # return the user_item matrix 

user_item = create_user_item_matrix(df)


# In[157]:


# `2.` Complete the function below which should take a user_id and provide an ordered list of the most similar users to that user (from most similar to least similar).  The returned result should not contain the provided user_id, as we know that each user is similar to him/herself. Because the results for each user here are binary, it (perhaps) makes sense to compute similarity as the dot product of two users. 
# 
# Use the tests to test your function.

# In[158]:


def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    # compute similarity of each user to the provided user
    similar = user_item[user_item.index==user_id].dot(user_item.T)
    # sort by similarity
    similar = similar.sort_values(by=user_id, axis=1, ascending=False)
    # create list of just the ids
    most_similar_users = similar.columns.tolist()
    # remove the own user's id
    most_similar_users.remove(user_id)
    
    return most_similar_users # return a list of the users in order from most to least similar
        

# In[160]:


def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    
    # Your code here
    article_ids = list(map(float, article_ids))
    #article_names = df.loc[article_ids,'title'].drop_duplicates().values.tolist()
   
    #return article_names # Return the article names associated with list of article ids
    
    # Your code here
    article_names = df[df['article_id'].isin(article_ids)]['title'].drop_duplicates().values.tolist()
    return article_names # Return the article names associated with list of article ids
    

def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # Your code here
    article_ids = df[df['user_id']==user_id]['article_id'].drop_duplicates().values.tolist()
    article_ids = list(map(str, article_ids))
    article_names = get_article_names(article_ids)
    
    return article_ids, article_names # return the ids and names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
     # Your code here
    similar_users = find_similar_users(user_id)
    
    recs = list(set(df[df['user_id'].isin(similar_users)]['article_id']))
    
    return recs[:m]


# In[161]:


# Check Results
get_article_names(user_user_recs(1, 10)) # Return 10 recommendations for user 1


# In[162]:
# `4.` Now we are going to improve the consistency of the **user_user_recs** function from above.  
# 
# * Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.
# 
# 
# * Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be  what would be obtained from the **top_articles** function you wrote earlier.

# In[225]:


def sort_notseen(values,df=df):
    df = df[df['article_id'].isin(values)]
    df = df['article_id'].value_counts()
    df = df.index.tolist()
    df = list(map(str, df))
    return df


# In[288]:


def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # Your code here
    similar_users = find_similar_users(user_id)
    
    #num_interaction = df.groupby(by=['user_id'])['title'].count()
    
    col = ['neighbor_id','similarity','num_interactions']
    
    neighbors_df = pd.DataFrame(columns=col)

    for sim in similar_users:
        neighbor_id = sim
        num_interaction = df.groupby(by=['user_id'])['title'].count()
        num_interaction = num_interaction[num_interaction.index == sim].values[0]
        similarity = user_item[user_item.index == user_id].dot(user_item.loc[sim].T).values[0]
    
    
        neighbors_df = neighbors_df.append({'neighbor_id' : neighbor_id, 'similarity' : similarity, 'num_interactions' : num_interaction}, ignore_index=True)
    
    neighbors_df.sort_values(by=['similarity','num_interactions'], ascending=False, inplace=True)
    
    return neighbors_df # Return the dataframe specified in the doc_string



def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    # Your code here
    # get similar users to the input user
    similar_users = get_top_sorted_users(user_id).neighbor_id.values.tolist()
    # get all articles that read by the input user
    articles_read_by_user,_ = get_user_articles(user_id)
    # get the top articles has most interaction and views
    top_articles = get_top_article_ids(m)
    
    recs = []
    rec_names = []
    
    for usr in similar_users:
        articles_similar_user,_ = get_user_articles(usr)
        # the articles the seen by both users
        seen = np.intersect1d(articles_read_by_user,articles_similar_user)
        #The articles the has been seen by the input users
        unseen = list(set(seen) ^ set(articles_similar_user))
        # filter unseen list by with most viewd articles,
        filtered_unseen = np.intersect1d(unseen,top_articles)
        
        
        if len(np.setdiff1d(filtered_unseen,recs))>0:
            
            diff_arti = np.setdiff1d(filtered_unseen,recs)
            recs.extend(diff_arti)
            rec_names.extend(get_article_names(diff_arti))
        
        
        if len(recs)>=m:
            recs = recs[:m]
            rec_names = rec_names[:m]
                                
            break
        
        
    
    
    
    
    
    
    return recs, rec_names


# In[290]:


# Quick spot check - don't change this code - just use it to test your functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)


