#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ## load review data ##
# 

# In[2]:


review = pd.read_csv('data_reviews_final.csv')


# In[3]:


review.head()


# In[4]:


meta_data = pd.read_csv('meta_data.csv')


# In[5]:


meta_data.head()


# In[6]:


prd_in_review = list(review.asin.unique())
meta_data = meta_data[meta_data['asin'].isin(prd_in_review)]
meta_data.head()


# ## Put the data inside dictionary to faster the execution ## 

# In[7]:


dict_user_item_rating = {}
list_us = list(review.reviewerID)
list_items = list(review.asin)
rat = list(review.overall)

for i in range(len(list_us)):
    us = list_us[i]
    it = list_items[i]
    if us not in dict_user_item_rating:
        dict_user_item_rating[us]={}
    dict_user_item_rating[us][it] = rat[i]


# In[8]:


meta_data = meta_data[meta_data['main_cat'].notna()]


# In[9]:


categories = {'Books':1, 'Arts, Crafts & Sewing':2, 'Amazon Home':3, 'All Electronics':4,
       'Office Products':5, 'Health & Personal Care':6,
       'Industrial & Scientific':7, 'Toys & Games':8,
       'Tools & Home Improvement':9, 'Sports & Outdoors':10, 'Software':11,
       'Camera & Photo':12, 'Home Audio & Theater':13, 'Movies & TV':14, 'Grocery':15,
       'Pet Supplies':16, 'All Beauty':17,  'Automotive':18,
       'Baby':19, 'Cell Phones & Accessories':20, 'Computers':21,
       
       'Car Electronics':22}


# ## build profile of user and vector  of features##

# In[10]:


dic_user_profile = {}
user_min_max_price = {}
for user in dict_user_item_rating.keys():
    
    profile_of_user = np.zeros(23) #22 for categories 1 for mean price 
    
    prcd_prd = list(dict_user_item_rating[user].keys())
    
    
    data_prcs_buy = meta_data[meta_data['asin'].isin(prcd_prd)]
    prc_cat = list(data_prcs_buy[data_prcs_buy['main_cat'].notna()].main_cat)
                   
    prices = list(data_prcs_buy[data_prcs_buy['price'].notna()].price)
    l  = []               
    mean_price = 0.0
    for pr in prices:
        if pr.startswith('$'):
            d = pr.split(' -')
            mean_price += float(d[0].replace(',','')[1:])
            l.append(float(d[0].replace(',','')[1:]))
            
    user_min_max_price[user] = {} 
    if len(l)==0:
        user_min_max_price[user]['min_price']= 0
        user_min_max_price[user]['max_price']= 1
    else:
        user_min_max_price[user]['min_price']= min(l)
        user_min_max_price[user]['max_price']= max(l)
    
        
        
    if mean_price!=0:
         profile_of_user[22] = mean_price/len(prices)
    
    for ct in prc_cat :
        profile_of_user[categories[ct]-1] += 1
        
    dic_user_profile[user] = profile_of_user
                   
    
    
    
    
    


# In[11]:


dic_user_profile


# In[12]:


user_min_max_price


# In[13]:


from statistics import mean
import math
from sklearn.metrics.pairwise import euclidean_distances
def get_k_nearst_neighbors(user,k):
    global review
    global dict_user_item_rating
    
    #data_of_user = data1[data1['reviewerID']==user]
    #prec_items = list(data_of_user.asin.unique())
    #print(prec_items)
    prec_items = dict_user_item_rating[user].keys()
    
    profile_user_c = dic_user_profile[user]
    
    
     
    
    data_of_similair_users = review[review['asin'].isin(prec_items)]
    #print(data_of_similair_users)
    users_similair = list(data_of_similair_users[data_of_similair_users['reviewerID']!=user].reviewerID.unique())
    #print(len(users_similair))
    
    
    users_similarity = {}
    
    for user_s in users_similair:
        
        profile_sim_user = dic_user_profile[user_s]
        sim = euclidean_distances([profile_user_c],[profile_sim_user])[0][0]
        '''
        mn_r_us_n = mean(list(dict_user_item_rating[user_s].values()))
        sim = 0.0
        qt = 0.0
        data_of_suser = dict_user_item_rating[user_s]
        items_of_sus = dict_user_item_rating[user_s].keys()
        sim_items = list(set(prec_items).intersection(set(items_of_sus)))
        for i in range(len(sim_items)):
            item = sim_items[i]
            #print(user,user_s,item)
            
            user_rat = data_of_user[item]
            #print(item,data_of_suser[data_of_suser['asin']==item].overall.unique())
            usern_rat = data_of_suser[item]
            
            sim += (user_rat - mn_r_us_c)*(usern_rat - mn_r_us_n)
            qt += (user_rat - mn_r_us_c)**2*(usern_rat - mn_r_us_n)**2
        if qt!=0:
            sim = sim/math.sqrt(qt)'''
        users_similarity[user_s] = abs(sim)
        
    
    sorted_s_us = {k: v for k, v in sorted(users_similarity.items(), key=lambda item: item[1])}
    
    return list(sorted_s_us.keys())[:k]


# In[14]:


def new_items(user,sim_users):
    global review
    data_of_user = review[review['reviewerID']==user]
    
    prec_items = list(data_of_user.asin.unique())
    
    prd_sim_us = review[review['reviewerID'].isin(sim_users)] 
    prd_with_good_rating = prd_sim_us[prd_sim_us['overall']>=3.5].asin.unique()
    sim = list(set(prd_with_good_rating).difference(set(prec_items)))
    return sim


# ## get recommendation for the first 100 users in the data using KNN##

# In[15]:



users = list(review['reviewerID'].unique())

i = 0

k = 5

for user in users:
    s = get_k_nearst_neighbors(user,k)
    print('recommended items for user ',user,' step 1 \n\n:',new_items(user,s),'\n')
    
    i+=1 
    if i== 100:
        break
    


# In[18]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# ## calculate most repuated words for every category ##

# In[19]:


#here i applied the steps of tokenization , remove stop words , lemmatization ....
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
import collections
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()


stop_words = set(stopwords.words('english'))
import operator
cat_200rep = {}
for cat in categories.keys():
    prd_of_cat = meta_data[meta_data['main_cat']==cat].asin.unique()
    
    d_text_review = review[review['asin'].isin(prd_of_cat) ]
    text_reviews = list(d_text_review[d_text_review['overall']>=4.0].reviewText)
    long_sent = ' '.join(text_reviews)
    word_tokens = word_tokenize(long_sent)  
  
    filtered_sentence = [w for w in word_tokens if not w in stop_words]  
    
    specific_words = [',',':','.','?','!','$','this','I','IF','he']

    filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence if not w in specific_words] 
            
    counter=collections.Counter(filtered_sentence)
    
    sorted_countor = sorted(counter.items(),key=operator.itemgetter(1),reverse=True)
    
    cat_200rep[cat] = []
    
    for j in range(min(200,len(sorted_countor))):
        cat_200rep[cat].append(sorted_countor[j][0])
    
    print('done with',cat)


# In[21]:


cat_200rep.keys()


# In[22]:


prd_in_review = list(review.asin.unique())
len(prd_in_review)


# ## calculate nb top fives repeted words in most 200 top word in category  ##

# In[23]:




dic_prod_nb = {}

for prd in prd_in_review:
    reviews_on_prd = list(review[review['asin']==prd].reviewText)
    
    cat = list(meta_data[meta_data['asin']==prd].main_cat)
    if len(cat)==0:
        
        dic_prod_nb[prd] = 0
        continue
    
    sen_of_rev = ' '.join(reviews_on_prd)
    
    word_tokens = word_tokenize(sen_of_rev)  
  
    filtered_sentence = [w for w in word_tokens if not w in stop_words]  
    
    specific_words = [',',':','.','?','!','$','this','I','IF','he']

    filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence if not w in specific_words] 
            
    counter=collections.Counter(filtered_sentence)
    
    sorted_countor = sorted(counter.items(),key=operator.itemgetter(1),reverse=True)
    
    
    
    tmp = []
    for j in range(min(5,len(sorted_countor))):
        tmp.append(sorted_countor[j][0])
    
    dic_prod_nb[prd] = len(list(set(tmp).intersection(set(cat_200rep[cat[0]]))))


# In[24]:


len(dic_prod_nb.keys())


# ## apply ontology with Knn ##

# In[37]:


def onto_knn(user,sim_users,list_prd,k):
    global review
    global dic_prod_nb
    global meta_data
    global user_min_max_price

    dic_prdVec_dist_to_vecP1 = {}
    
    vecP1 = [1,5,5]
    
    reviews_of_sim_users = review[review['reviewerID'].isin(sim_users)]
    
    for prd in list_prd:
        avg_rat = mean(list(reviews_of_sim_users[reviews_of_sim_users['asin']==prd].overall))
        
        price_prd = meta_data[meta_data['asin']==prd]
        
        price_prd = list( price_prd[price_prd['price'].notna()].price )
        
        
        if len(price_prd)!=0.0 :
            if price_prd[0].startswith('$'):
                d = price_prd[0].split(' -')
                price = float(d[0].replace(',','')[1:])
            else:
                price = 0.0
        else:
            price = 0.0
          
        if (price <= user_min_max_price[user]['max_price'] ) and  (price >= user_min_max_price[user]['min_price'] ) : 
            feat1 = 1
        else:
            feat1 = 0
            
        feat3 = dic_prod_nb[prd]
        
        vect_prd = [feat1 , avg_rat , feat3]
        
        sim = euclidean_distances([vect_prd] , [vecP1])
        
        dic_prdVec_dist_to_vecP1[prd] = sim
        
    sorted_s_prd = {m: v for m, v in sorted(dic_prdVec_dist_to_vecP1.items(), key=lambda item: item[1])}
    
    return list(sorted_s_prd.keys())[:k]
    


# ## get recommendation for the first 100 users in the data  
# ## using KNN basic then using hybrid model 

# In[38]:


users = list(review['reviewerID'].unique())

i = 0

k = 5

for user in users:
    s = get_k_nearst_neighbors(user,k)
    new_items_from_knn = new_items(user,s)
    print('\nrecommended items for user ',user,' using knn \n\n:',new_items_from_knn,'\n')
    print('\nusing hybrid model \n\n',onto_knn(user , s, new_items_from_knn , k))
    
    i+=1
    if i == 100: 
        break


# In[ ]:




