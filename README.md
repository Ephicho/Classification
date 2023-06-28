# Classification
Classification 
Classification
Introduction
In machine learning, classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, based on a training set of data containing observations (or instances) whose category membership is known. Couple examples of classification problems are: (a) deciding whether a received email are a spam or an organic e-mail; (b) assigning a diagnosis of a patient based on observed characteristics of the patient (age, blood pressure, presence or absence of certain symptoms, etc.)
In this article I use the data from dap-projects-database.database.windows.net, to build a model to predict whether customer is going to churn or not depending on some attributes. We will try to build a classification model. After building each model we will evaluate them and compare which model are the best for our case. We will then try to optimise our model by tuning the hyper parameters of the model by using GridSearch. Lastly, we will save the prediction result from our dataset and then save our model for re-usability.
To start we will load some basic libraries such as Pandas and NumPy and then make some configuration to some of those libraries.
Data Pre-Processing
Before we can begin to create our first model we first need to load and pre-process. This step ensure that our model will receive a good data to learn from, as they said "a model is only as good as it's data". The data pre-processing will be divided into few steps as explained below.
Loading Data
In this first step we will load our dataset that has been uploaded from the database. From the dataset documentation found in medium we can see below are the list of column we have in our data:
# create the servers instance variable such as the server you are connecting to, database , username anf password
server = 'dap-projects-database.database.windows.net'
database = 'dapDB'
username = 'dataAnalyst_LP2'
password = ''

# This will connection string is an f string that includes all the variable above to extablish a connection 
# to the server
connection_string = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
Workflow
1) EDA - What data are we dealing with?
2) Preprocessing - Let's clean our data
3) Training time - Model building and comparison
4) Predictions and Model Evaluation
5) Conclusions and Final Thoughts
Exploratory Data Analysis - What data are we dealing with?
Understanding what data we're dealing with, will allow us to make better decisions about what preprocessing steps we need to apply, which classifier we're going to choose, and how to interpret the results.
Looking at data types and correct those wrongly encoded
data_raw.dtypes## We have one variable that's wrongly encoded as string
data_raw['TotalCharges'] = pd.to_numeric(data_raw['TotalCharges'], errors='coerce')
Missing values and Cardinality
Cardinality refers to the number of unique values of categorical variables. The following function allows us to check for missing values, as well as cardinality.
Some EDA
Tenure by Months
Customers at the beginning are most likely to churn
Customers also leave once they reach 70 months with the company
Crosstab of churn and security:

 OnlineSecurity  False  True 
Churn                       
False             861    746
True              617    124

p =  2.6365457216168814e-43

Rounded p value: 0.0
null hypothesis: having tech support does not affect churn
alternative hypothesis: having tech support affects a customer churning
Crosstab of churn and security:

 TechSupport  False  True 
Churn                    
False          845    762
True           630    111

p =  2.579438071668035e-51

Rounded p_value: 0.0
Null hypothesis: having online security does not affect churn
Alternative hypothesis: having online security affects a customer churning
Crosstab of churn and security:

 OnlineSecurity  False  True 
Churn                       
False             861    746
True              617    124

p =  2.6365457216168814e-43

Rounded p_value: 0.0
Null hypothesis rejected
Let's look at the relationship between categorical and numerical variables:
In our case, this would allow us to answer questions like
* "Are churned customers likely to get charged more?",
* "When do customers churn?", or "Are senior citizens more likely to churn?".
# Categorical-numerical variables
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

## Are churned customers likely to get charged more?
plt.subplot(1,3,1)
sns.boxplot(x= data['Churn'],y= data['MonthlyCharges'])
plt.title('MonthlyCharges vs Churn')

## When do customers churn?
plt.subplot(1,3,2)
sns.boxplot(x=data['Churn'],y= data['tenure'])
plt.title('Tenure vs Churn')

## Are senior citizen more likely to churn?
plt.subplot(1,3,3)
counts = (data2.groupby(['Churn'])['SeniorCitizen']
  .value_counts(normalize=True)
  .rename('percentage')
  .mul(100)
  .reset_index())
plot = sns.barplot(x="SeniorCitizen", y="percentage", hue="Churn",
                   data=counts).set_title('SeniorCitizen vs Churn')
* We can clearly see from this that monthly charges for churning customers are higher, while tenure is much lower.
* For senior citizens, there are actually more customers churning than staying with the company.
* This might be indicative that we're not focusing on the right customer segment.
Feature Engineering and Feature Selection
# Looking at multicollinearity with the feature matrix and selecting best features

plt.figure(figsize=(6,4))
correlations = data.corr()
sns.heatmap(correlations, annot=True, cmap='Blues')
plt.show()
Preprocessing
# Creating a scikit learn pipeline for preprocessing

## Selecting categorical and numeric features
numerical_ix = x_train.select_dtypes(include=np.number).columns
categorical_ix = x_train.select_dtypes(exclude=np.number).columns

## Create preprocessing pipelines for each datatype 
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())])

## Putting the preprocessing steps together
preprocessor = ColumnTransformer([
        ('numerical', numerical_transformer, numerical_ix),
        ('categorical', categorical_transformer, categorical_ix)],
         remainder='passthrough')
Model Building
model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
knn = KNeighborsClassifier()
rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model_dt.fit(x_train,y_train)
knn.fit(x_train,y_train)
rf.fit(x_train,y_train)
