#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Titanic Dataset.csv")


# In[3]:


# View the data
df.head()


# In[4]:


# Basic information about the data
df.info()


# In[5]:


# Describe the data
df.describe()


# # Data Preprocessing

# In[6]:


df.duplicated().sum()


# In[7]:


df.isnull().sum()


# In[8]:


# Replace null values
# df.replace(np.nan,'0',inplace = True)
# Fill missing values using mean of the numeric column

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# In[9]:


#Filter data

df[df['Pclass']==1].head()


# # EDA

# In[10]:


sb.displot(data=df, x='Age', hue='Survived', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Age Distribution by Survived')
plt.show()


# In[11]:


sb.displot(data=df, x='Age', hue='Pclass', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Age Distribution by Pclass')
plt.show()


# In[12]:


sb.displot(data=df, x='Fare', hue='Survived', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Fare Distribution')
plt.show()


# In[13]:


# Fare Transformation (Log)
df['Fare_org']=df['Fare']
df['Fare']=np.log(df['Fare']+1)
sb.displot(data=df, x='Fare', hue='Survived', kind='kde', fill=True, palette=sb.color_palette('bright'))
plt.title('Fare Distribution')
plt.show()


# # Heat Map

# In[14]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[15]:


# Pclass can be a proxy for socio-economic status (SES)
sb.boxplot(x="Pclass",y="Age",data=df,palette=sb.color_palette('bright')[:3])


# In[16]:


# Pclass can be a proxy for socio-economic status (SES)
sb.boxplot(x="Pclass",y="Fare_org",data=df,palette=sb.color_palette('bright')[:3])


# In[18]:


# Viz for each var
sb.pairplot(data=df)
plt.title('PairPlot')


# # Correlation

# In[19]:


df_corr = df.drop(['PassengerId'], axis = 1)


# In[20]:


#Corplot
sb.heatmap(df_corr.corr(),annot = True, cmap = plt.cm.Blues)
plt.title('Correlation Matrix')


# # Classification

# In[22]:


#drop unused var
df =df.drop(columns=['Name', 'Ticket', 'Cabin', 'Fare_org'], axis=1)


# Label Encoding

# In[23]:


from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()


# # Model Training

# In[24]:


#Split Data
train_df = df.sample(frac=0.7, random_state=25)
test_df = df.drop(train_df.index)


# In[25]:


train_df.head()

# input split
X = train_df.drop(columns=['PassengerId', 'Survived'], axis=1)
y = train_df['Survived']


# In[27]:


X.head()


# In[29]:


from sklearn.model_selection import train_test_split, cross_val_score
# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))

    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:',np.mean(score))


# # Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)


# In[31]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model)


# In[34]:


pip install xgboost


# In[35]:


from xgboost import XGBClassifier
model = XGBClassifier()
classify(model)


# # LGBM Classifier

# In[37]:


pip install lightgbm


# In[42]:


from lightgbm import LGBMClassifier
model = LGBMClassifier()
model.fit(X, y)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[43]:


from lightgbm import LGBMClassifier
model = LGBMClassifier()
classify(model)


# # Cat Boost Classifier

# In[44]:


pip install catboost


# In[45]:


from catboost import CatBoostClassifier
model = CatBoostClassifier(verbose=0)
classify(model)


# # Confusion Matrix Display

# In[46]:


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


# # Survived People by Class 

# In[47]:


survived_class=df.groupby('Pclass')['Survived'].count()
survived_class.to_frame('Survivers')


# In[48]:


df['Family_members']=df['Parch']+df['SibSp']
df['Family_members']


# In[49]:


sns.barplot(x =df['Family_members'], y =df['Survived'])


# In[50]:


survived_people=df.groupby('Survived').median()
survived_people


# In[51]:


sns.pairplot(df, hue="Survived", height=2.5)
plt.show()


# In[52]:


'''Thank you'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




