#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("fake_bills.csv",sep = ";")
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.dropna(inplace=True)


# In[8]:


data['diagonal'] = data['diagonal'].astype('int')
data['height_left'] = data['height_left'].astype('int')
data['height_right'] = data['height_right'].astype('int')
data['margin_low'] = data['margin_low'].astype('int')
data['margin_up'] = data['margin_up'].astype('int')
data['length'] = data['length'].astype('int')
data['is_genuine'] =data['is_genuine'].astype('int')


# In[9]:


(data.isnull().sum()/data.shape[0])*100


# Dropping the null values

# In[10]:


data.dropna(inplace=True)


# In[11]:


data.isnull().sum()


# Train Test Split Model

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X = data.drop(['is_genuine'],axis=1)
X.head()


# In[14]:


y = data['is_genuine']
y.head()


# In[15]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[16]:


Churn = (sum(data['is_genuine'])/len(data['is_genuine'].index))*100
Churn


# Looking at the correlations between the variables

# In[17]:


plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(data.corr(),annot = True)
plt.show()


# Running the First Training Model

# In[18]:


import statsmodels.api as sm


# In[19]:


X_train_Gm = sm.add_constant(X_train)
ls = sm.GLM(y_train, X_train_Gm, family = sm.families.Binomial())
ls.fit().summary()


# Feature Selection Using RFE

# In[20]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[21]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)


# In[22]:


rfe.support_


# In[23]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[24]:


col = X_train.columns[rfe.support_]


# In[25]:


X_train.columns[~rfe.support_]


# Assessing the model with StatsModels

# In[26]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[27]:


y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[28]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# Creating a dataframe with the actual churn flag and the predicted probabilities

# In[29]:


y_train_pred_final = pd.DataFrame({'Fake':y_train.values, 'Fake_Prob':y_train_pred})
y_train_pred_final['CustID'] = y_train.index
y_train_pred_final.head()


# In[30]:


y_train_pred_final['predicted'] = y_train_pred_final.Fake_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[31]:


from sklearn import metrics


# In[32]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Fake, y_train_pred_final.predicted )
print(confusion)


# In[33]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Fake, y_train_pred_final.predicted))


# In[34]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# Metrics beyond simply accuracy

# In[35]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[36]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[37]:


# Let us calculate specificity
TN / float(TN+FP)


# In[38]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[39]:


# positive predictive value 
print (TP / float(TP+FP))


# In[40]:


# Negative predictive value
print (TN / float(TN+ FN))


# Plotting ROC

# In[41]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[42]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Fake, y_train_pred_final.Fake_Prob, drop_intermediate = False)


# In[43]:


draw_roc(y_train_pred_final.Fake, y_train_pred_final.Fake_Prob)


# Finding Optimal Cutoff Point

# In[44]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Fake_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[45]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Fake, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[46]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[47]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Fake_Prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[48]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Fake, y_train_pred_final.final_predicted)


# In[49]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Fake, y_train_pred_final.final_predicted )
confusion2


# In[50]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[51]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[52]:


# Let us calculate specificity
TN / float(TN+FP)


# In[53]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[54]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[55]:


# Negative predictive value
print (TN / float(TN+ FN))


# Precision and Recall

# In[56]:


confusion = metrics.confusion_matrix(y_train_pred_final.Fake, y_train_pred_final.predicted )
confusion


# Precision

# In[57]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# Recall

# In[58]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[59]:


from sklearn.metrics import precision_score, recall_score


# In[60]:


precision_score(y_train_pred_final.Fake, y_train_pred_final.predicted)


# In[61]:


recall_score(y_train_pred_final.Fake, y_train_pred_final.predicted)


# Precision and recall tradeoff

# In[62]:


from sklearn.metrics import precision_recall_curve


# In[63]:


y_train_pred_final.Fake, y_train_pred_final.predicted


# In[64]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Fake, y_train_pred_final.Fake_Prob)


# In[65]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# Making predictions on the test set

# In[66]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg)             # running RFE with 13 variables as output
rfe = rfe.fit(X_test, y_test)


# In[67]:


col = X_test.columns[rfe.support_]


# In[68]:


X_test_sm = sm.add_constant(X_test[col])
logm3 = sm.GLM(y_test,X_test_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[69]:


y_test_pred = res.predict(X_test_sm)


# In[70]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[71]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[85]:


y_test_df['Notes'] = y_test_df.index


# In[86]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[87]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[88]:


y_pred_final.head()


# In[89]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Fake_Prob'})


# In[90]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[92]:


y_pred_final['final_predicted'] = y_pred_final.Fake_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[94]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.is_genuine, y_pred_final.final_predicted)


# In[96]:


confusion2 = metrics.confusion_matrix(y_pred_final.is_genuine, y_pred_final.final_predicted )
confusion2


# In[97]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[98]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[99]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:




