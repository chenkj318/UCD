# ECS171FinalProject

## Table of Contents:
  - Introduction
  - Figures
  - Methods
  - Results
  - Discussion
  - Conclusion
  - Collaboration



## Introduction <br/>
&emsp;The purpose of this project is to build a regression model for predicting flights delay in the U.S.  For a better prediction model, passengers can have better schedule planning instead of missing any possible important events. Related to daily experience, people tend to believe that airport congestion or specific low-cost flight is the main cause of flight delays. The model should be able to reveal how each factor impacts the flight's delay and how long the arriving delay might be. To be clear, this project is focused on prediciting flights' arriving delay and hopefully benefit for picking up passengers and good time planning afterwards, not apply to predict departure delay.  <br/>
&emsp;The dataset is found from Kaggle's “2015 Flight Delays and Cancellations”, originally abstracted from the U.S. Department of Transportation (DOT). For this project, based on different goals compared to the Kaggle Event, not full of the data was used to build the prediction model for delays. The model mainly relies on the  “flights.csv”, which includes detail of each flight (5819079 samples) in the USA from 2015. <br/>
 
 
 ## Figures <br/>
 
We use pair plot and heat map to show the distribution of each feature and the correlation between features. From the pair plot, we can see that none of the features follow the normal distribution and that is why we choose MixMax scalar to standardize our data instead of normalizing. 

![heatmap](https://user-images.githubusercontent.com/84880988/205522309-44018b5f-8758-4fc7-9c9a-462ec7977201.png)

![output](https://user-images.githubusercontent.com/84880988/205522215-9ba765de-1629-441f-a3fd-df5b6a491d1f.png)

After testing around several regression models, we finally decide to use ridge regression. We plot the full scatter plot for the training data and each color indicates one feature. 
```python
print("Full scatter plot for the training data")
plt.figure(figsize=(7, 7))
for i in range (len(X_train.columns)):
  sns.scatterplot(X_train[X_train.columns[i]], y_train)
plt.show()
```
![scatterplot](https://user-images.githubusercontent.com/84880988/205522502-479d87d7-1151-4a67-907e-9dbfc676bc15.png)



We also scatter plot the three most significant features that influence the target variable arrival delay by first printing out all model weights. This is from the SGDRegressor. 
```python
sns.scatterplot(X_train["AIR_TIME"], y_train)
sns.scatterplot(X_train["DISTANCE"], y_train)
sns.scatterplot(X_train["DEPARTURE_DELAY"], y_train)
```
![download](https://user-images.githubusercontent.com/78530943/205525250-51ac6d11-f8f1-42ef-b64f-26286aebf998.png)

 
 
 ## Methods <br/>
 #### Data Downloads
```python
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import warnings
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
warnings.filterwarnings('ignore')
```

```python
link = 'https://drive.google.com/file/d/1_H1j57EahEEXpZtQ413BpF2JiAAr1Anv/view?usp=share_link'
id = "1_H1j57EahEEXpZtQ413BpF2JiAAr1Anv"
print (id) # Verify that you have everything after '='
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('flights.csv')  
```
Pandas and Numpy is required for this project for basic data processing.
```python
import  pandas as pd
import  numpy as np

flights = pd.read_csv("flights.csv")
```
 #### Data Preprocessing
  Note: Full code with comments for Data Preprocessing can be found in "Data_Preprocessing.ipynb"
 
 1. Drop features unrelated features.
 ```python
columns = flights.columns
print("Features of current dataset",columns)
flights=flights.drop(['WEATHER_DELAY','LATE_AIRCRAFT_DELAY','AIRLINE_DELAY','SECURITY_DELAY','AIR_SYSTEM_DELAY','CANCELLATION_REASON',
 "CANCELLED", "DIVERTED","TAXI_IN", "TAXI_OUT"],axis=1)
 ```
 2. Drop overlapped features. 
```python
flights=flights.drop(['SCHEDULED_ARRIVAL','ARRIVAL_TIME','DEPARTURE_TIME','SCHEDULED_DEPARTURE','TAIL_NUMBER','FLIGHT_NUMBER', 'WHEELS_OFF', 'WHEELS_ON', "SCHEDULED_TIME", "ELAPSED_TIME"],axis=1)
 ```
 4. Drop feature of the year.
```python
print(flights["YEAR"].unique())
flights=flights.drop(['YEAR'],axis=1)
columns = flights.columns

print(flights.dtypes)
 ```
 6. Extract categrical features, Encode categrical features with Ordinal Encoder.
Library preprocessing is used for Encoding.
```python
from sklearn import preprocessing
num_flights = flights.loc[:,['DEPARTURE_DELAY',"AIR_TIME","DISTANCE"]]
one_hot_original = flights.loc[:,['AIRLINE',"DAY_OF_WEEK","MONTH"]]
ordinal_orginal = flights.loc[:,['DAY',"ORIGIN_AIRPORT","DESTINATION_AIRPORT"]]

display(one_hot_original)
one_hot_encode=one_hot_original.astype("str")
encoder = preprocessing.OrdinalEncoder()
encoder.fit(one_hot_encode)
ordinal_cat = encoder.transform(one_hot_encode)
print(ordinal_cat, ordinal_cat.shape)

display(ordinal_orginal)
ordinal_orginal["DAY"] = ordinal_orginal["DAY"].astype("str")
ordinal_orginal.dtypes
ordinal_orginal=ordinal_orginal.astype("str")
encoder = preprocessing.OrdinalEncoder()
encoder.fit(ordinal_orginal)
ordinal = encoder.transform(ordinal_orginal)
print(ordinal, ordinal.shape)
```

 7. Build the Dataframe after partically encoding dataset.

```python
df_ordinal = pd.DataFrame(ordinal, columns = ['DAY','ORIGIN_AIRPORT','DESTINATION_AIRPORT'])
df_cat = pd.DataFrame(ordinal_cat, columns = ['AIRLINE','DAY_OF_WEEK','MONTH'])

df_flights = num_flights.join(df_ordinal)
df_flights = df_flights.join(df_cat)
```

 8. Scale the dataset and keep the dataset as Dataframe.
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_flights)
df_flight_noscale =df_flights
df_flights = scaler.transform (df_flights)

# Make df_flights into Dataframe
df_flights = pd.DataFrame(df_flights)
df_flights = df_flights.rename(columns={0:'DEPARTURE_DELAY',1:'AIR_TIME',2:'DISTANCE',3:'DAY',4:'ORINGIN_AIRPORT',5:'DESTINATION_AIRPORT',6:'AIRLINES',7:'DAY_OF_WEEK',8:'MONTH' })
target = flights["ARRIVAL_DELAY"]
df_flights = df_flights.join(target)
df_flights
```
 
 9. Clean up NAN Values
```python
# check NAN values
display(df_flights)
display(df_flights.isna().any())

df_flights['DEPARTURE_DELAY'] = df_flights['DEPARTURE_DELAY'].fillna(0)
df_flights['ARRIVAL_DELAY'] = df_flights['ARRIVAL_DELAY'].fillna(0)

print("After filling DEPARTURE_DELAY, and ARRIVAL_DELAY:")
display(df_flights.isna().any())

df_flights = df_flights.dropna()
display(df_flights)
print("after deleting all rows that contain NaN values,")
display(df_flights.isna().any())
```
 #### Models Training
 
 Note: Only the code of building models is displaying below. Plotting/graphs and Full code with comments for Models Training can be found in "Model_Training.ipynb"
 
 1. Linear Regression (by SGD Regressor)

```python
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
reg = SGDRegressor()
reg.fit(X=X_train, y=y_train)

y_pred = reg.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred)

print("Model weights: ")
print(reg.coef_)
print('Testing MSE:',mse1)
print("Model score:",reg.score(X_test, y_test) )
```

 2. Poylnomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform((X_train))
plr = LinearRegression()
# Note that I didn't do reshape on X_poly as it's already a matrix.
plr.fit(X_poly, (y_train))
    
predicted = plr.predict(poly.transform((X_test)))
    
display(plr.intercept_)
display(plr.coef_[0:3])

print(f'Polynomial regression with degree = {3}')
print(f'Training MSE error is:',mean_squared_error(plr.predict(X_poly), y_train))
print(f'Testing MSE error is:', mean_squared_error(predicted, y_test))
print("Model score:",plr.score(X_test, y_test))
```

 3. Lasso Regression

```python
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train, y_train)

y_train_pred = reg.predict(X_train)

train_mse = mean_squared_error(y_train,y_train_pred)
print('Training MSE:',train_mse)
y_hat = reg.predict(X_test)
print(f'Testing MSE error is: {round(mean_squared_error(y_hat, y_test),4)}')
print("Model score:",reg.score(X_test, y_test))
```

 4. Ridge Regression

```python
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
# plt.figure(figsize=(7, 7))
# sns.scatterplot(X_train, y_train)
# plt.show()

train_mse = mean_squared_error(y_train,y_train_pred)
print('Training MSE:',train_mse)
y_hat = reg.predict(X_test)
print(f'Testing MSE error is: {round(mean_squared_error(y_hat, y_test),4)}')

print("Model weights: ")
print(reg.coef_)

z = reg.score(X_test, y_test)
print("Accuracy score: ",z)
```

## Discussion <br/>

*Notes: To see the results and graphs, please view the Juypter Notebook "Final_Project_Full_Path.ipynb"
 
#### Data Preprocessing: <br/>

&emsp;The very first part of this project is to extract the features that relates to the main purpose of this project (predict the ARRIVAL_DELAY). So the data set of “airlines” and “airports” are not selected to use. All the features in “flights” data that are unrelated with predicting is dropped in the first place. For example, “weather_delay”, “late_airport_dealy”, “cancelled” and “diverted”. These features indicates either not delaying or reasons of delaying. The “YEAR” is dropped since all data are collected from 2015. <br/>

&emsp;Then the overlapping features are dropped. For example, there are two features “scheduled_departure” and “departure_time),  which is overlapped with “departure_delay”. “Tail_number” is funcitoning same as index. These overlapped features are dropped.
The dataframe has 5819079 rows × 10 columns at this point. Six of features “MONTH”, “DAY”, “DAY_OF_WEEK”, “AIRLINE”, “ORIGIN_AIRPORT”,”DESTINATION_AIRPORT” , are categorical features. (For the date,  they are not necessarily categorical feature, but I think it is more proper to apply them rather than numercial, since the number of a month does not really meaning a value).  <br/>

&emsp;Because there are 970 different airports  and 14 airlines, it is more reansonable to encode these categorical features by ordinal encoding. The one-hot encoding is increasing too much “dimensions” which has a bad trade off with any possibile higher accuracy. After encoding, dataframe is applied to MinMaxScaler (distributions are not normal so cannot use standard scaler) in order to keep scale and easier  for data training. <br/>

&emsp;Only three features include NAN Values, “ARRIVAL_DELAY”, “DEPARTURE_DELAY”, “AIR_TIME”. For the first two, NAN Values indicates there is no delay happening, and the cell is filled by 0.  For “AIR_TIME”, NAN values indicates either “diverted” or “canceld”, these samples are dropped. Compare to the overall amount of samples, 0.18%  of samples are dropped, and have a small impact. <br/>


#### Model Selection and Training: <br/>

&emsp;Model training starts by splitting test and training data, we started spliting the test_size of 10%. While the testing error  is much higher than the training error, which indicates overfitting condition. The final splitting rate applied is 20:80 which has the most stable errors. Note we have a very big data-set that training all dataset takes a very long time and could lead to RAM fulfilled on the system. Hence, some part of samples are extrated from random select function (built in pandas), 10% of data is extrated and the training size 457120.Since the goal is to predict the value of “ARRIVAL_DELAY”, it is set use the target.

&emsp;The first two models, we simply started it with linear regression with SGD regressor and a polynomial regression model. Selecting these two model is the initial stage to see how does the dataset overall performance of predicting the target by regressions. SGD is selected due to its efficiency, since there is a large dataset, and first model is to overview the performace. Two parameters are applied here, shuffle and alpha has been reset. The idea is to have a more random training in each eopch and faster model built. Alpha has been tested and 0.0001 has the optimized output. Scatter plotting is applied on three most weighted attributes.

&emsp;Then for the polynomial regression, based on the amount of dataset, our machine is not supporting degrees over 5. For the testing purpose, degree 3 is applied here with stable error on both test and training set. Parameter n_jobs is set to -1 in LinearRegression to use all processors, but very higher speeds.  

&emsp;With the first two models given the intuition, the dataset is able to provide a prediction below 200. The next two models have a similar process to add regularization to a linear model. For both of these two models (Lasso and Ridge Regression) are applied here to improve performance by “adjusting”/regularizing the weights of features. Also, it is optimizing relate to the main purpose of this project. It needs to indicate which aspects affect the arrival delay most.

&emsp;For Lasso Regression, parameters “selection”  and “alpha” (controlling the strength of regularization). Alpha around 0.01 has the best performance based on approaches, and selection is selected as “random” to have a more randomized training model. For Ridge Regression,  it has similar accuracy compared to Lasso Regression. Meanwhile, the parameters do not influence the accuracy as much as Lasso Regression. The “solver” is chosen as “sag”, which is based on Stochastic Average Gradient descent since the dataset has been scaled in preprocessing part.

 #### Result Analysis:

&emsp;Overall the best model we achieved is Lasso Regression and Ridge Regression. The testing error in around 143 to 148. Back to the context for prediction, the error indicates the model predict arrival delay in about 11-13 minutes. 

&emsp;We create four models for this dataset. The first one is linear regression with SGD. The testing error is 224.7 and the model score is 0.85. The features air time, distance, and departure delay hold the largest three weight magnitude. The second model we choose is the polynominal regression. We set the degree equals to 3. Now, the testing MSE error reduces to 131.64. 

&emsp;We then choose Lasso regession with and without random selection to train and test our dataset. The testing MSE errors are 145.9 and 194.7 respectively. The model scores are 0.90 and 0.87 respectively. We think the Lasso regression with random selection model had a good performance on predicting. But we run Ridge regression to see if it performs better than the previous ones. 

&emsp;The testing MSE error for Ridge regression is 144.6 and the accuracy score is 0.904. We think this model has the best performance and choose its reuslts as the final results. We print out the model weights for all features and find the biggest three weights by comparing their magnitudes. The corresponding features are departure delay (2.09e+03), air time(2.40e+02), and distance(-2.21+e02). This indicates the features departure delay, air time, and distance affect the target varialble arrival delay most significantly. 

Graph of the highest three weight corresponds to attribute Departure delay, airtime and distance. (Ridge Regression)
Blue scatterplot indicates departure_delay. Orange indicates the airtime and green indicates distance, these two are high correlated.

![scatterplot_three_features](https://user-images.githubusercontent.com/84880988/205522548-4a6454fb-d7fe-4437-984d-97f24215d487.png)

#### Possibly frauts / short-comings: <br/>

&emsp;The original selected data contains a huge amount of samples, so the model can be only built on part of it based on the machine’s limit. It would be better to randomly extract a larger portion from the original dataset. 

&emsp;Also, the purpose set at the initial stage is to fit with regression models and predict the numerical value of delays. It could be better to make a classification problem to indicate ranges of delaying time. Then there could be more options of models to build with.

&emsp;We are lucky to find the "sweet point" for balancing the accuracy without overfitting, in the simple models. With higher efficiency from gradient descent, it has overall good performance. 

## Conclusion <br/>

&emsp;Overall, the models based on Lasso Regression and Ridge Regression have a good performance with an error of about 12 minutes in predicting the arrival delay. With details of weights “air time”, “distance” and “departure delay” has the most significant impact on “arrival delay” compared to other attributes. One of the main editions that could be applied in future is to apply more sub-attributes that are provided from the same source. For example, in this dataset, we trained with contribute “origin airport” and “destination airport”. There are actually more attributes related to the airport,  for example, “longitude”, and “latitude”. These could bring data exploration on weather aspects, that we did not apply in this project.

&emsp;Another of the main editions could be applied to a slightly different classification question “Will the flight delay” instead of “how long will the flight delay”. It can also bring more dimensions of attributes with attributes “canceled” and “diverted”. Some thresholds could be adjusted to determine delay or not or slight delay. And more models could be approached.
 
## Collaboration:
Zhaolin Zhong: Lead the teamwork and went through the entire procedure of this project (from topic selection, data exploration, preprocessing, and model training.) Works more on the model(parameters) selections. Write most parts of the README file with Shang Wu together. 

Shang Wu: Lead the teamwork and went through the entire procedure of this project (from topic selection, data exploration, preprocessing, and model training.) Focuses more on the figures display and explanation. Write most parts of the README file with Zhaolin together. 

Xueqi Zhang: Work on data selection and preprocessing of dropping unrelated features. Discuss with data selection and the purpose of the project.

Kejing Chen: Work on data selection and collect errors from each regression model. 

Huaiyu Jiang:  Create ad organize the GitHub repository. Work on data selection and preprocessing. Discussed data exploration and model selection. Proofread README file and work on Conclusions.


