#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:41:41 2020

@author: saransharora
"""

"""
Data and Description

Loan_ID: Unique Loan ID
Gender: Male/ Female
Married: Applicant married (Y/N)
Dependents: Number of dependents
Education: Applicant Education (Graduate/ Under Graduate)
Self_Employed: Self employed (Y/N)
ApplicantIncome: Applicant income
CoapplicantIncome: Coapplicant income
LoanAmount: Loan amount in thousands
Loan_Amount_Term: Term of loan in months
Credit_History: credit history meets guidelines
Property_Area: Urban/ Semi Urban/ Rural
Loan_Status: Loan approved (Y/N)
"""


"""
Before starting to code, I am trying to come up with a few Null Hypothesis:
    1. Applicant Income - Higher the income of the applicant, higher will be the chance of loan approval
    2. Loan Amount - Higher the loan amount, lesser will be the chance of getting the loan approved
    3. Loan_Amount_Term - Higher the loan amount term, lesser will be the chance of getting the loan approved
    4. Credit_History: People who have repaid their previous loans, have a higher chance of getting their loan approved
    5. Property_Area: If property in urban area, the chance of getting the loan getting approved should be higher
"""
#Importing dependencies
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('LoanPrediction/train.csv')
print(data.head())

test = pd.read_csv('LoanPrediction/test.csv')
test_original = test.copy()
#checking size
print(data.size)

#checking shape
print(data.shape)

#checking columns
print(data.columns)

#Viewing all columns for the first row
print(data.iloc[0])

#Checking datatypes
print(data.dtypes)  #Object means categorical variables. 

#checking the description and summary
print(data.describe())
print(data.info)

#check for missing values in dataframe
print(data.isnull().sum())

#Making a copy of the train dataset
loan = data.copy()

#Uni-variate analysis
print(loan.Loan_Status.value_counts())
print(loan.Loan_Status.value_counts(normalize=True))

#Visualising Nominal(Categorical) variables
print(loan.Loan_Status.value_counts().plot.bar())
print(loan.Gender.value_counts().plot.bar())
print(loan.Married.value_counts().plot.bar())
print(loan.Self_Employed.value_counts().plot.bar())
print(loan.Credit_History.value_counts().plot.bar())


#Visualising Ordinal(Categorical) variables
print(loan.Dependents.value_counts().plot.bar())
print(loan.Education.value_counts().plot.bar())
print(loan.Property_Area.value_counts().plot.bar())

#Visualise contiuous variables - numerical
AppInc = loan.ApplicantIncome
AppInc.plot.hist(bins=100)
LoanAm = loan.LoanAmount
LoanAm.plot.hist(bins=100)


# Bi-variate analysis
# Checking loan status for Gender
gender_ct = gender_ct.div(gender_ct.sum(1).astype(float), axis=0)
gender_ct.plot.bar(stacked=True) #The proportion of applications approved for male and female is approximately the same

# Checking loan status for all nominal categorical variables
# Married vs Loan Status
pd.crosstab(loan.Married, loan.Loan_Status).div(pd.crosstab(loan.Married, loan.Loan_Status).sum(1).astype(float), axis=0).plot.bar()

# Self Employment vs Loan Status
loan_ct = pd.crosstab(loan.Self_Employed, loan.Loan_Status)
loan_div = loan_ct.div(loan_ct.sum(1).astype(float),axis=0)
loan_div.plot.bar(rot=0)

# Credit History vs Loan Status
credit_ct = pd.crosstab(loan.Credit_History, loan.Loan_Status)
credit_div = credit_ct.div(credit_ct.sum(1).astype(float), axis=0)
credit_div.plot.bar(rot=0)
#The plot shows that Loan Status approval depends a lot on Credit history

# Dependents vs Loan Status
dep_ct = pd.crosstab(loan.Dependents, loan.Loan_Status)
dep_div = dep_ct.div(dep_ct.sum(1).astype(float), axis=0)
dep_div.plot.bar()

# Education vs Loan_Status
edu_ct = pd.crosstab(loan.Education, loan.Loan_Status)
edu_div = edu_ct.div(edu_ct.sum(1).astype(float), axis=0)
edu_div.plot.bar()

# Property_Area vs Loan_Status
prop_ct = pd.crosstab(loan.Property_Area, loan.Loan_Status)
prop_div = prop_ct.div(prop_ct.sum(1).astype(float), axis=0)
prop_div.plot.bar(stacked=True)

#Applicant income vs Loan Status 
appinc_cut = pd.cut(loan.ApplicantIncome, bins = 5, labels = ['Very Low','Low','Medium','High','Very High'])
appinc_ct = pd.crosstab(appinc_cut, loan.Loan_Status)
appinc_div = appinc_ct.div(appinc_ct.sum(1).astype(float), axis=0)
appinc_div.plot.bar(rot=0, stacked=True)

#Loan amount vs Loan Status
Loan_Amount_cut = pd.cut(loan.LoanAmount, bins=5, labels=['Very Low','Low','Medium','High','Very High'])
loan_amount_ct = pd.crosstab(Loan_Amount_cut, loan.Loan_Status)
loan_amount_div = loan_amount_ct.div(loan_amount_ct.sum(1).astype(float), axis=0)
loan_amount_div.plot.bar(rot=0, stacked=True)

loan.Dependents.replace('3+',3,inplace=True)
loan.Dependents = loan.Dependents.astype(str)

test.Dependents.replace('3+',3,inplace=True)
test.Dependents = loan.Dependents.astype(str)

loan.Loan_Status.replace('N',0,inplace=True)
loan.Loan_Status.replace('Y',1,inplace=True)

# Correlation between variables
cor = loan.corr()
sns.heatmap(cor, cmap = "binary")
# Loan Amount - Applicant Income & Credit History - Loan Status correlate with high values

# Treating missing values
loan.isnull().sum()
# Filling up categorical values with mode and numerical values with mean.
loan.Gender.fillna(loan.Gender.mode()[0], inplace=True)
loan.Married.fillna(loan.Married.mode()[0], inplace=True)
loan.Dependents.fillna(loan.Dependents.mode()[0], inplace=True)
loan.Self_Employed.fillna(loan.Self_Employed.mode()[0], inplace=True)
loan.Credit_History.fillna(loan.Credit_History.mode()[0], inplace=True)
loan.Loan_Amount_Term.fillna(loan.Loan_Amount_Term.mode()[0], inplace=True)
loan.LoanAmount.fillna(loan.LoanAmount.median(), inplace=True)


#Filling na values in test data
test.Gender.fillna(loan.Gender.mode()[0], inplace=True)
test.Married.fillna(loan.Married.mode()[0], inplace=True)
test.Dependents.fillna(loan.Dependents.mode()[0], inplace=True)
test.Self_Employed.fillna(loan.Self_Employed.mode()[0], inplace=True)
test.Credit_History.fillna(loan.Credit_History.mode()[0], inplace=True)
test.Loan_Amount_Term.fillna(loan.Loan_Amount_Term.mode()[0], inplace=True)
test.LoanAmount.fillna(loan.LoanAmount.median(), inplace=True)

#test.isnull().sum()

# Feature Engineering
LoanAm = loan.LoanAmount
LoanAm.plot.hist(bins=100)
# The data is right skewed

loan['LoanAmount_logged'] = ""
loan['LoanAmount_logged'] = np.log(loan.LoanAmount)

test['LoanAmount_logged'] = ""
test['LoanAmount_logged'] = np.log(test.LoanAmount)


LoanAm_log = loan.LoanAmount_logged
LoanAm_log.plot.hist(bins=100)
# The data is not skewed now, and will help in better predictions. 

# Since Loan ID will not be used to predict, we can drop it off from both train and test dataset. 

loan.drop('Loan_ID', axis=1, inplace= True)
test.drop('Loan_ID', axis=1, inplace= True)

# Making separate dataframes for X and Y variables
X = loan.drop('Loan_Status',axis=1)
Y = loan.Loan_Status

train_X = pd.get_dummies(X)
train_X.drop(labels=['Gender_Male','Married_No','Dependents_0','Dependents_nan','Education_Not Graduate','Self_Employed_No','Property_Area_Rural'],axis=1, inplace=True)
Test_X =  pd.get_dummies(test)
Test_X.drop(labels=['Gender_Male','Married_No','Dependents_0','Dependents_nan','Education_Not Graduate','Self_Employed_No','Property_Area_Rural'],axis=1, inplace=True)


train_X.iloc[1]
Test_X.iloc[1]


from sklearn.model_selection import train_test_split
# Cross validation
x_train, x_cv, y_train, y_cv = train_test_split(train_X, Y, test_size = 0.3)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)

pred_values = model.predict(x_cv)
accuracy_score(y_cv, pred_values)

# Predicting for test data

submission=pd.read_csv("LoanPrediction/sample_submission.csv")
pred_test = model.predict(Test_X)
ids = test_original.Loan_ID
f_df = pd.DataFrame(data=list(zip(ids,pred_test)), columns=['Loan_ID','Loan_Status'])

f_df.Loan_Status.replace(0,'N',inplace=True)
f_df.Loan_Status.replace(1,'Y',inplace=True)

pd.DataFrame(f_df).to_csv('Submission1.csv', index=False)

# Uploaded the solution and found accuracy: 77.78% 
# https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/
# Working on Decision tree, Random Forests and XGBoost algorithms
