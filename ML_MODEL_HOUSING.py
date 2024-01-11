#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np


# In[2]:


os.getcwd()


# In[3]:


housing = pd.read_csv('housing.csv')
housing.head()


# In[4]:


housing.info() #The info() method is useful to get a quick description of the data


# In[5]:


housing["ocean_proximity"].value_counts() # You can find out what categories exist and how many districts belong to each category by using thevalue_counts() method:'''


# In[6]:


housing.describe() # The describe() method shows a summary of the numerical attributes.


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[8]:


housing.head()


# In[9]:


# checking the data numerical efficiency with HISTOGRAM
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[10]:


# Creating train_test_split
from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing , test_size=0.2 , random_state=42)
# our data is split into training set=80% and testing set=20%
print(len(train_set),"/",len(test_set))


# In[11]:


# The following code uses the pd.cut() function to create an income category attribute with 5 categories 
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3,4.5,6.,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()


# In[12]:


# ready to do stratified sampling based on the income category. # this step is not compulsory , here data was less so we did it
from sklearn.model_selection import  StratifiedShuffleSplit 
split =StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[13]:


strat_test_set["income_cat"].value_counts()/len(strat_test_set)


# In[14]:


#remove the income_cat attribute so the data is back to its original state:
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#We spent quite a bit of time on test set generation for a good reason: this is an often neglected but critical part of a Machine Learning project


# ### Discover and Visualize the Data to Gain Insights

# In[15]:


'''So far you have only taken a quick glance at the data to get a general understanding of
the kind of data you are manipulating. Now the goal is to go a little bit more in depth.'''


# In[16]:


#Let’s create a copy so you can play with it without harming the training set:
housing = strat_train_set.copy()
housing.head()


# In[17]:


#Since there is geographical information (latitude and longitude), it is a good idea to -
# -create a scatterplot of all districts to visualize the data 
housing.plot(kind='scatter' , x='longitude' , y='latitude')


# In[18]:


# Setting the alpha option to 0.1 makes it much easier to visualize the places -
# -where there is a high density of data points 
housing.plot(kind='scatter' , x='longitude' , y='latitude' , alpha=0.1)


# In[19]:


# now lets look at the housing price . 
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
#This image tells you that the housing prices are very much related to the location


# ### Looking for Correlations

# In[20]:


#Since the dataset is not too large, you can easily compute the standard correlation -
# -coecient (also called Pearson’s r) between every pair of attributes using the corr() method:


# In[21]:


corr_matrix = housing.corr()


# In[22]:


# Now let’s look at how much each attribute correlates with the median house value:
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[23]:


'''The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that
there is a strong positive correlation; for example, the median house value tends to go
up when the median income goes up. When the coefficient is close to –1, it means
that there is a strong negative correlation; you can see a small negative correlation
between the latitude and the median house value (i.e., prices have a slight tendency to
go down when you go north). Finally, coefficients close to zero mean that there is no
linear correlation'''


# In[24]:


#Another way to check for correlation between attributes is to use Pandas’ -
# -scatter_matrix function, which plots every numerical attribute against every other numerical attribute.
# -there are now 11 numerical attributes, you would get 112 =121 plots, 
# - which would not fit on a page, so let’s just focus on a few promising
#- attributes that seem most correlated with the median housing value


# In[25]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))


# ### The most promising attribute to predict the median house value is the median income, so let’s zoom in on their correlation scatterplot

# In[26]:


housing.plot(kind='scatter' , x ="median_income" , y="median_house_value", alpha=0.1 )


# #### This plot reveals a few things. First, the correlation is indeed very strong; you can
# clearly see the upward trend and the points are not too dispersed. Second, the price
# cap that we noticed earlier is clearly visible as a horizontal line at $500,000. But this
# plot reveals other less obvious straight lines: a horizontal line around $450,000,
# another around $350,000, perhaps one around $280,000, and a few more below that.
# You may want to try removing the corresponding districts to prevent your algorithms
# from learning to reproduce these data quirks.
# 

# #### DATA QUIRKS - an unusual habit or part of someone's personality, or something that is strange and unexpected

# #### One last thing you may want to do before actually preparing the data for Machine Learning algorithms is
# to try out various attribute combinations. For example, the
# total number of rooms in a district is not very useful if you don’t know how many
# households there are. What you really want is the number of rooms per household.
# Similarly, the total number of bedrooms by itself is not very useful: you probably
# want to compare it to the number of rooms. And the population per household also
# seems like an interesting attribute combination to look at. Let’s create these new
# attributes:
# 

# In[27]:


housing.head()


# In[28]:


housing['rooms_per_households'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_rooms'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_households'] = housing['population']/housing['households']


# In[29]:


# lets again check how much attributes are corelatted with median house value


# In[30]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# #### This round of exploration does not have to be absolutely thorough; the point is to
# start off on the right foot and quickly gain insights that will help you get a first rea‐
# sonably good prototype. But this is an iterative process: once you get a prototype up
# and running, you can analyze its output to gain more insights and come back to this
# exploration step.

# # Prepare the Data for Machine Learning Algorithms

# #### But first let’s revert to a clean training set (by copying strat_train_set once again),
# and let’s separate the predictors and the labels since we don’t necessarily want to apply
# the same transformations to the predictors and the target values (note that drop()
# creates a copy of the data and does not affect strat_train_set):
# 

# In[33]:


housing = strat_train_set.drop('median_house_value' , axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.head()


# ## Data Cleaning

# ### You noticed earlier that the total_bedrooms
# attribute has some missing values, so let’s fix this
# You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna()
# methods
# housing.dropna(subset=["total_bedrooms"]) # option 1
# housing.drop("total_bedrooms", axis=1) # option 2
# median = housing["total_bedrooms"].median() # option 3
# housing["total_bedrooms"].fillna(median, inplace=True)

# ### Scikit-Learn provides a handy class to take care of missing values: SimpleImputer.
# Here is how to use it. First, you need to create a SimpleImputer instance, specifying
# that you want to replace each attribute’s missing values with the median of that
# attribute:
# 

# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')


# ### Since the median can only be computed on numerical attributes, we need to create a
# copy of the data without the text attribute ocean_proximity:

# #### Since the median can only be computed on numerical attributes, we need to create a
# copy of the data without the text attribute ocean_proximity:

# In[36]:


housing.head()


# In[39]:


housing_num = housing.drop("ocean_proximity" ,axis=1)
housing_num.head() # ocean_proximity column is gone


# #### Now you can fit the imputer instance to the training data using the fit() method:

# In[40]:


imputer.fit(housing_num)


# #### The imputer has simply computed the median of each attribute and stored the result
# in its statistics_ instance variable. Only the total_bedrooms attribute had missing
# values, but we cannot be sure that there won’t be any missing values in new data after
# the system goes live, so it is safer to apply the imputer to all the numerical attributes:
# 

# In[41]:


imputer.statistics_


# In[42]:


housing_num.median().values


# #### Now you can use this “trained” imputer to transform the training set by replacing
# missing values by the learned medians:

# In[43]:


X = imputer.transform(housing_num)


# #### The result is a plain NumPy array containing the transformed features. If you want to
# put it back into a Pandas DataFrame, it’s simple:

# In[44]:


housing_tr = pd.DataFrame(X , columns=housing_num.columns)


# In[46]:


housing_tr.head()


# In[47]:


housing_tr.info()


# ## Handling Text and Categorical Attributes
# Earlier we left out the categorical attribute ocean_proximity because it is a text
# attribute so we cannot compute its median:

# In[44]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# #### Most Machine Learning algorithms prefer to work with numbers anyway, so let’s con‐
# vert these categories from text to numbers. For this, we can use Scikit-Learn’s Ordina
# lEncoder class19:
# 

# In[45]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()


# In[46]:


housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# #### You can get the list of categories using the categories_ instance variable. It is a list
# containing a 1D array of categories for each categorical attribute (in this case, a list
# containing a single array since there is just one categorical attribute):

# In[47]:


ordinal_encoder.categories_


# ##### One issue with this representation is that ML algorithms will assume that two nearby
# values are more similar than two distant values. This may be fine in some cases (e.g.,
# for ordered categories such as “bad”, “average”, “good”, “excellent”), but it is obviously
# not the case for the ocean_proximity column (for example, categories 0 and 4 are
# clearly more similar than categories 0 and 1). To fix this issue, a common solution is
# to create one binary attribute per category: one attribute equal to 1 when the category
# is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is
# “INLAND” (and 0 otherwise), and so on. This is called one-hot encoding, because
# only one attribute will be equal to 1 (hot), while the others will be 0 (cold). The new
# attributes are sometimes called dummy attributes. Scikit-Learn provides a OneHotEn
# coder class to convert categorical values into one-hot vectors20:

# In[48]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# ###### Notice that the output is a SciPy sparse matrix, instead of a NumPy array. This is very
# useful when you have categorical attributes with thousands of categories. After one￾hot encoding we get a matrix with thousands of columns, and the matrix is full of
# zeros except for a single 1 per row. Using up tons of memory mostly to store zeros
# would be very wasteful, so instead a sparse matrix only stores the location of the non‐Notice that the output is a SciPy sparse matrix, instead of a NumPy array. This is very
# useful when you have categorical attributes with thousands of categories. After one￾hot encoding we get a matrix with thousands of columns, and the matrix is full of
# zeros except for a single 1 per row. Using up tons of memory mostly to store zeros
# would be very wasteful, so instead a sparse matrix only stores the location of the non‐

# In[49]:


housing_cat_1hot.toarray()


# ### Once again, you can get the list of categories using the encoder’s categories_
# instance variable:
# 

# In[50]:


cat_encoder.categories_


# ###### If a categorical attribute has a large number of possible categories
# (e.g., country code, profession, species, etc.), then one-hot encod‐
# ing will result in a large number of input features. This may slow
# down training and degrade performance. If this happens, you may
# want to replace the categorical input with useful numerical features
# related to the categories: for example, you could replace the
# ocean_proximity feature with the distance to the ocean (similarly,
# a country code could be replaced with the country’s population and
# GDP per capita). Alternatively, you could replace each category
# with a learnable low dimensional vector called an embedding. Each
# category’s representation would be learned during training
# 

# # Custom Transformation Feature Scaling

# In[ ]:





# ## Transformation Pipelines
# 

# ####  Fortunately, Scikit-Learn provides the Pipeline class to help with
# such sequences of transformations. Here is a small pipeline for the numerical
# attributes:
# Now let's build a pipeline for preprocessing the numerical attributes (note that we could use CombinedAttributesAdder() instead of FunctionTransformer(...) if we preferred):

# In[73]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[80]:


num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])



housing_num_tr = num_pipeline.fit_transform(housing_num)


# ### So far, we have handled the categorical columns and the numerical columns sepa‐
# rately. It would be more convenient to have a single transformer able to handle all col‐
# umns, applying the appropriate transformations to each column. In version 0.20,
# Scikit-Learn introduced the ColumnTransformer for this purpose, and the good news
# is that it works great with Pandas DataFrames. Let’s use it to apply all the transforma‐
# tions to the housing data:

# In[81]:


from sklearn.compose import ColumnTransformer


# In[ ]:





# In[82]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num" , num_pipeline , num_attribs),("cat" , OneHotEncoder() , cat_attribs),])
housing_prepared = full_pipeline.fit_transform(housing)


# # Select and Train a Model

# #### At last! You framed the problem, you got the data and explored it, you sampled a
# training set and a test set, and you wrote transformation pipelines to clean up and
# prepare your data for Machine Learning algorithms automatically. You are now ready
# to select and train a Machine Learning model.
# 

# ## Training and Evaluating on the Training Set
# Let’s first train a Linear Regression model

# In[85]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared , housing_labels)


# ##### Done! You now have a working Linear Regression model. Let’s try it out on a few
# instances from the training set:

# In[86]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("predictions:" , lin_reg.predict(some_data_prepared))
print("labels:" , list(some_labels))


# ### It works, although the predictions are not exactly accurate (e.g., the first prediction is
# off by close to 40%!). Let’s measure this regression model’s RMSE on the whole train‐
# ing set using Scikit-Learn’s mean_squared_error function:

# In[88]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels , housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# ##### Okay, this is better than nothing but clearly not a great score This is an example of a model underfitting
# the training data. When this happens it can mean that the features do not provide
# enough information to make good predictions, or that the model is not powerful
# enough.
# the main ways to fix underfitting are to
# select a more powerful model, to feed the training algorithm with better features, or
# to reduce the constraints on the model. This model is not regularized, so this rules
# out the last option. You could try to add more features (e.g., the log of the popula‐
# tion), but first let’s try a more complex model to see how it does.

# ### Let’s train a DecisionTreeRegressor. This is a powerful model, capable of finding
# complex nonlinear relationships in the data

# In[90]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared , housing_labels)


# #### Now that the model is trained, let’s evaluate it on the training set

# In[106]:


housing_prediction = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels , housing_prediction)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# #### No error at all? Could this model really be absolutely perfect? Of course,
# it is much more likely that the model has badly overfit the data. How can you be sure?
# As we saw earlier, you don’t want to touch the test set until you are ready to launch a
# model you are confident about, so you need to use part of the training set for train‐
# ing, and part for model validation.

# # Better Evaluation Using Cross-Validation

# ###### One way to evaluate the Decision Tree model would be to use the train_test_split
# function to split the training set into a smaller training set and a validation set, then train your models against the smaller training set and evaluate them against the vali‐
# dation set. It’s a bit of work, but nothing too difficult and it would work fairly well.
# A great alternative is to use Scikit-Learn’s K-fold cross-validation feature. The follow‐
# ing code randomly splits the training set into 10 distinct subsets called folds, then it
# trains and evaluates the Decision Tree model 10 times, picking a different fold for
# evaluation every time and training on the other 9 folds. The result is an array con‐
# taining the 10 evaluation scores:
# 

# In[99]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg , housing_prepared , housing_labels, scoring="neg_mean_squared_error" , cv=10)
tree_rmse_scores = np.sqrt(-scores)
print(tree_rmse_scores)
print("scores :" ,scores)
print("mean" , scores.mean())
print("Std Dev:" ,scores.std())


# In[98]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
 scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
display_scores(tree_rmse_scores)


# #### Now the Decision Tree doesn’t look as good as it did earlier. In fact, it seems to per‐
# form worse than the Linear Regression model! Notice that cross-validation allows
# you to get not only an estimate of the performance of your model, but also a measure
# of how precise this estimate is (i.e., its standard deviation).

# ##### Let’s compute the same scores for the Linear Regression model just to be sure:

# In[101]:


lin_scores = cross_val_score(lin_reg , housing_prepared , housing_labels, scoring="neg_mean_squared_error" , cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(lin_rmse_scores)
print("scores :" ,lin_scores)
print("mean" , lin_scores.mean())
print("Std Dev:" , lin_scores.std())


# In[102]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[104]:


#### That’s right: the Decision Tree model is overfitting so badly that it performs worse than the Linear Regression model.


# # Let’s try one last model now: the RandomForestRegressor

# In[111]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_prediction = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels , housing_prediction)
forest_rmse = np.sqrt(forest_mse)
print("forest_mse: ",forest_rmse)



# In[113]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)
def display_scores(forest_scores):
    print("Scores:", forest_scores)
    print("Mean:", forest_scores.mean())
    print("Standard deviation:", forest_scores.std())
    
display_scores(forest_rmse_scores)


# ##### Random Forests look very promising. However, note that
# the score on the training set is still much lower than on the validation sets, meaning
# that the model is still overfitting the training set. 

# # Fine-Tune Your Model

# ### Let’s assume that you now have a shortlist of promising models. You now need to
# fine-tune them. Let’s look at a few ways you can do that.
# 

# ## Grid Search
# 

# ### you should get Scikit-Learn’s GridSearchCV to search for you. All you need to
# do is tell it which hyperparameters you want it to experiment with, and what values to
# try out, and it will evaluate all the possible combinations of hyperparameter values,
# using cross-validation. For example, the following code searches for the best combi‐
# nation of hyperparameter values for the RandomForestRegressor:

# In[117]:


from sklearn.model_selection import GridSearchCV
param_grid = [
    ## try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3,10,30] , 'max_features': [2,4,6,8],
    # then try 6 (2×3) combinations with bootstrap set as False
    'bootstrap': [False], 'n_estimators': [3,10],'max_features': [2,3,4] },
]

forest_reg = RandomForestRegressor(random_state = 42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV (forest_reg ,param_grid,cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score = True)
grid_search.fit(housing_prepared, housing_labels)


# ### The best hyperparameter combination found:

# In[118]:


grid_search.best_params_


# In[119]:


grid_search.best_estimator_


# #### Let's look at the score of each hyperparameter combination tested during the grid search:

# In[120]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ### Congratulations, you have successfully fine-tuned your best model!

# In[121]:


pd.DataFrame(grid_search.cv_results_)


# ## Randomised Search

# In[122]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# In[123]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ### Analyze the Best Models and Their Errors

# In[124]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[125]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# ### Evaluate Your System on the Test Set

# In[126]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[127]:


final_rmse


# ### We can compute a 95% confidence interval for the test RMSE:

# In[128]:


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# #### We could compute the interval manually like this:

# In[129]:


m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)


# ### Alternatively, we could use a z-scores rather than t-scores:

# In[130]:


zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# In[ ]:




