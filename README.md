# Binary-Classification
Binary Classification using different Machine Learning Models, Feature Extraction and Preprocessing, Business Overview and Analysis.

# **Table of Contents**

1. [Introduction and Instructions](#my-first-title)
2. [Overview of Learning from the Specialization](#my-second-title)
3. [Labs and Projects](#my-third-title)
4. [Results and Conclusion](#my-fourth-title)

## **Introduction and Instructions**
This report is aimed at explaining the methodology that was chosen to analyze and do the binary classification based on clients’ data set for prediction of Default status of the clients. 
Available information is related to the clients’ features including Occupation, Marital Status, Observation Date, Income, Loan_amount_requsted, Term Length, Installment/Income, Schufa credit Score, Number of Applicants.
Generally, we have 10000 number of observation and 10 features for years (2008-2018) time intervals which in each date there are set of observation falling and the class target(target_var) is binary values of 0 and 1 which shows the status of the clients’ application whether is rejected or approved.



## **Method**

### **1. Data Visualizing and Preprocessing**
#### **1.1. Data Quality Assurance**

As the first step in the data processing and preparation, the time “00:00:00” was dopped from (‘OBS_DATE’), because there is no meaning and time "00:00:00" is same for all data, and format of the date in OBS_DATE variable was changed to standard date format.
Then we check if the target values are distributed equally or no, which is shown is figure.

![Capture](https://user-images.githubusercontent.com/75788150/178348852-920841d4-ba20-47f8-a205-8aecf890fa49.PNG)
