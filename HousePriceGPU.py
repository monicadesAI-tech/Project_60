#!/usr/bin/env python
# coding: utf-8

# ### Kaggle Advance House Price Prediction Using Pytorch- Tabular Dataset
# 
# https://docs.fast.ai/tabular.html
# https://www.fast.ai/2018/04/29/categorical-embeddings/
# https://www.fast.ai/2018/04/29/categorical-embeddings/
# https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
# 
# 
# 1. Category Embedding
# 

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('houseprice.csv',usecols=["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                         "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


for i in df.columns:
    print("Column name {} and unique values are {}".format(i,len(df[i].unique())))


# In[7]:


import datetime
datetime.datetime.now().year


# In[8]:


df['Total Years']=datetime.datetime.now().year-df['YearBuilt']


# In[9]:


df.drop("YearBuilt",axis=1,inplace=True)


# In[10]:


df.columns


# In[11]:


cat_features=["MSSubClass", "MSZoning", "Street", "LotShape"]
out_feature="SalePrice"


# In[12]:


from sklearn.preprocessing import LabelEncoder
lbl_encoders={}
lbl_encoders["MSSubClass"]=LabelEncoder()
lbl_encoders["MSSubClass"].fit_transform(df["MSSubClass"])


# In[13]:


lbl_encoders


# In[14]:


from sklearn.preprocessing import LabelEncoder
lbl_encoders={}
for feature in cat_features:
    lbl_encoders[feature]=LabelEncoder()
    df[feature]=lbl_encoders[feature].fit_transform(df[feature])


# In[15]:


df


# In[31]:


### Stacking and Converting Into Tensors
import numpy as np
cat_features=np.stack([df['MSSubClass'],df['MSZoning'],df['Street'],df['LotShape']],1)
cat_features
                       


# In[32]:


### Convert numpy to Tensors
import torch
cat_features=torch.tensor(cat_features,dtype=torch.int64)
cat_features


# In[42]:


#### create continuous variable
cont_features=[]
for i in df.columns:
    if i in ["MSSubClass", "MSZoning", "Street", "LotShape","SalePrice"]:
        pass
    else:
        cont_features.append(i)


# In[43]:


cont_features


# In[44]:


### Stacking continuous variable to a tensor
cont_values=np.stack([df[i].values for i in cont_features],axis=1)
cont_values=torch.tensor(cont_values,dtype=torch.float)
cont_values


# In[45]:


cont_values.dtype


# In[47]:


### Dependent Feature 
y=torch.tensor(df['SalePrice'].values,dtype=torch.float).reshape(-1,1)
y


# In[52]:


df.info()


# In[50]:


cat_features.shape,cont_values.shape,y.shape


# In[54]:


len(df['MSSubClass'].unique())


# In[63]:


#### Embedding Size For Categorical columns
cat_dims=[len(df[col].unique()) for col in ["MSSubClass", "MSZoning", "Street", "LotShape"]]


# In[64]:


cat_dims


# In[65]:


embedding_dim= [(x, min(50, (x + 1) // 2)) for x in cat_dims]


# In[66]:


embedding_dim


# In[ ]:





# In[71]:


import torch
import torch.nn as nn
import torch.nn.functional as F
embed_representation=nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dim])
embed_representation


# In[134]:


cat_features


# In[138]:


cat_featuresz=cat_features[:4]
cat_featuresz


# In[141]:


pd.set_option('display.max_rows', 500)
embedding_val=[]
for i,e in enumerate(embed_representation):
    embedding_val.append(e(cat_features[:,i]))


# In[142]:


embedding_val


# In[144]:


z = torch.cat(embedding_val, 1)
z


# In[146]:


#### Implement dropupout
droput=nn.Dropout(.4)


# In[147]:


final_embed=droput(z)
final_embed


# In[103]:


##### Create a Feed Forward Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
class FeedForwardNN(nn.Module):

    def __init__(self, embedding_dim, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dim])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((out for inp,out in embedding_dim))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


# In[187]:


len(cont_features)


# In[123]:


torch.manual_seed(100)
model=FeedForwardNN(embedding_dim,len(cont_features),1,[100,50],p=0.4)


# In[ ]:





# In[124]:


model


# #### Define Loss And Optimizer

# In[125]:


loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


# In[126]:


df.shape


# In[192]:


cont_values


# In[127]:


cont_values.shape


# In[128]:


batch_size=1200
test_size=int(batch_size*0.15)
train_categorical=cat_features[:batch_size-test_size]
test_categorical=cat_features[batch_size-test_size:batch_size]
train_cont=cont_values[:batch_size-test_size]
test_cont=cont_values[batch_size-test_size:batch_size]
y_train=y[:batch_size-test_size]
y_test=y[batch_size-test_size:batch_size]


# In[129]:


len(train_categorical),len(test_categorical),len(train_cont),len(test_cont),len(y_train),len(y_test)


# In[130]:



epochs=5000
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model(train_categorical,train_cont)
    loss=torch.sqrt(loss_function(y_pred,y_train)) ### RMSE
    final_losses.append(loss)
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[148]:

# conda install -c anaconda ipython  [in anaconda prompt]
import matplotlib.pyplot as plt
import IPython
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.plot(range(epochs), final_losses)
#plt.ylabel('RMSE Loss')
#plt.xlabel('epoch');
#plt.show(block=True)
# AttributeError: 'NoneType' object has no attribute 'run_line_magic'
# It will work in Jupyter Notebook, above 4lines so you can uncomment it to see plots.


# In[150]:


#### Validate the Test Data
y_pred=""
with torch.no_grad():
    y_pred=model(test_categorical,test_cont)
    loss=torch.sqrt(loss_function(y_pred,y_test))
print('RMSE: {}'.format(loss))


# In[178]:


data_verify=pd.DataFrame(y_test.tolist(),columns=["Test"])


# In[179]:


data_predicted=pd.DataFrame(y_pred.tolist(),columns=["Prediction"])


# In[180]:


data_predicted


# In[184]:


final_output=pd.concat([data_verify,data_predicted],axis=1)
final_output['Difference']=final_output['Test']-final_output['Prediction']
final_output.head()


# In[186]:


#### Saving The Model
#### Save the model
torch.save(model,'HousePrice.pt')


# In[189]:


torch.save(model.state_dict(),'HouseWeights.pt')


# In[188]:


### Loading the saved Model
embs_size=[(15, 8), (5, 3), (2, 1), (4, 2)]
model1=FeedForwardNN(embs_size,5,1,[100,50],p=0.4)


# In[190]:


model1.load_state_dict(torch.load('HouseWeights.pt'))


# In[191]:


model1.eval()


# In[ ]:




