# 1. Title and Author

## Title: SugarSense - Predicting Diabetes Risk
DATA 606 - UMBC Data Science Master Degree Capstone

**Guided By**: Dr Chaojie (Jay) Wang

**Author**: 
  - Name: Varshita Adavi
  - University ID: OC83614
  - GitHub profile: [Varshita Adavi_Github](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone)
  - LinkedIn profile: [Varshita Adavi_LinkedIn](https://www.linkedin.com/in/varshita-adavi/)
  - PowerPoint presentation: [SugarSense_PPT](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/blob/main/docs/SugarSense%20Project.pdf)
  - YouTube video:


# 2. Background

### Objective:
The objective of the project is to develop a predictive model leveraging machine learning techniques within a user-friendly application that predicts an individual's risk of developing diabetes using key health indicators, aiming to empower users with personalized risk assessments and promote early intervention for better health outcomes.

### 2.1 What is it about?
The project aims to develop a user-friendly application utilizing machine learning algorithms to predict an individual's risk of diabetes based on key health indicators. By inputting personal health data, users receive personalized risk assessments, empowering proactive health management and potentially preventing diabetes-related complications. The app aids in promoting awareness about diabetes risk factors, encourages early detection, and facilitates informed lifestyle choices for better health outcomes.

### 2.2 Why does it matter?
The project matters because diabetes is a prevalent and potentially life-threatening chronic condition that affects millions of people worldwide. Early detection and intervention are crucial for managing diabetes effectively and reducing the risk of complications such as heart disease, stroke, kidney failure, and blindness. By developing an app that can accurately predict an individual's risk of diabetes based on their health indicators, we can empower individuals to take proactive steps towards preventive care, such as adopting healthier lifestyle habits, monitoring their blood sugar levels, and seeking medical advice if necessary. Ultimately, the project has the potential to improve public health outcomes by promoting early intervention and reducing the burden of diabetes-related morbidity and mortality.

### 2.3 What are your research questions?
  - What are the most significant predictors of diabetes risk among various health indicators such as gender, age, BMI, blood pressure, and cholesterol levels?
  - How accurately can machine learning algorithms predict an individual's likelihood of developing diabetes based on their personal health information?
  - What are the differences in diabetes risk prediction models between different demographic groups (e.g., gender, age groups)?
  - How do lifestyle factors (such as diet, exercise, and smoking habits) interact with health indicators to influence the risk of developing diabetes?


# 3. Data

- **Data sources**: [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Data size**: 6.2 MB
- **Data shape**: (70692, 22)
- **Time period**: 2015
- **Columns in dataset**:
    | **Column Name** | **Data Type** | **Description** |
    | :- | :- | :- |
    | Diabetes_binary | Float | Indicates the stage of diabetes. Scale: 0 = no diabetes, 1 = prediabetes or diabetes |
    | HighBP | Float | Describes if the person have high blood pressure or not. Scale: 0 = no high BP, 1 = high BP |
    | HighChol | Float | Describes if the person have high cholestrol or not. Scale: 0 = no high cholestrerol 1 = high cholestrerol |
    | CholCheck | Float | Describes if the person had cholesterol check up in past 5 years or not. Scale: 0 = no cholesterol check, 1 = yes cholesterol check |
    | BMI | Float | Indicates body mass index of the person |
    | Smoker | Float | Describes if the person have smoked at least 100 cigarettes in your entire life or not. Scale: 0 = no, 1 = yes |
    | Stroke | Float | Describes if the person had any stroke or not. Scale: 0 = no 1 = yes |
    | HeartDiseaseorAttack | Float | Describes if the person have any coronary heart disease (CHD) or myocardial infarction (MI). Scale: 0 = no, 1 = yes |
    | PhysActivity | Float | Describes if the person has physical activity in past 30 days - not including job. Scale: 0 = no, 1 = yes |
    | Fruits | Float | Indicates if the person consume any fruit 1 or more times per day. Scale: 0 = no, 1 = yes |
    | Veggies | Float | Indicates if the person consume any vegetable 1 or more times per day. Scale: 0 = no, 1 = yes |
    | HvyAlcoholConsump | Float | Indicates if the person is a heavy drinker (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week). Scale: 0 = no, 1 = yes |
    | AnyHealthcare | Float | Indicates if the person have any kind of health care coverage, including health insurance, prepaid plans such as HMO. Scale: 0 = no, 1 = yes |
    | NoDocbcCost | Float | Describes if there was a time in the past 12 months when you needed to see a doctor but could not because of cost. Scale: 0 = no, 1 = yes |
    | GenHlth | Float | Indicates general health of the person. Scale: (1-5): 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor |
    | MentHlth | Float | Describes about the persons mental health. It includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? Scale: 1-30 days |
    | PhysHlth | Float | Describes about the persons physical health. It includes physical illness and injury, for how many days during the past 30 days was your physical health not good? Scale: 1-30 days |
    | DiffWalk | Float | Indicates if the person has serious difficulty in walking or climbing stairs. 0 = No, 1 = Yes |
    | Sex | Float | Indicates the gender of the person. 0 = Female, 1 = Male |
    | Age | Float | Describes the age of the person in range category. 13-level age category: 1 = 18-24, ..., 9 = 60-64, ..., 13 = 80 or older |
    | Education | Float | Describes the education level of the person. Scale: 1-6: 1 = Never attended school or only kindergarten, 2 = Elementary, 3 = Some high school, 4 = High school graduate, 5 = Some college or technical school, 6 = College graduate |
    | Income | Float | Indicate the income range of the person. Scale: 1-8: 1 = less than $10,000, 5 = less than $35,000, 8 = $75,000 or more |


- **Target Variable:**

  Diabetes_binary is considered as the target variable. It predicts the likelihood of diabetes, the target variable typically represents whether an individual has been diagnosed with diabetes or not. This variable is calssified into two categories:
    - 0: no diabetes
    - 1: prediabetes or diabetes


# 4. Exploratory Data Analysis (EDA)

## 4.1 Dataset Summary:
With a total of 69,057 observations and 22 features per observation, the dataset has undergone thorough preprocessing. Duplicate values and missing data have been addressed, and the data types have been standardized. Additionally, the dataset has been explored through various visualizations to enhance comprehension and insight.

The summary statistics for the target variable, Diabetes_binary, in the dataset:
  - Count: 69,057
  - Mean: 0.508 (indicating that approximately 51% of observations are positive for diabetes)
  - Standard deviation: 0.500
  - Minimum: 0 (indicating no diabetes)
  - 25th percentile: 0
  - Median (50th percentile): 1
  - 75th percentile: 1
  - Maximum: 1 (indicating diabetes present)

These statistics offer a concise summary of the distribution and characteristics of the target variable, highlighting its prevalence and variability within the dataset.

## 4.2 Data Visualizations:

**Dataset Distribution on Target Variable**:

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/20aa0b2e-28c4-4d3f-afc7-5aecea2155df)

The graph suggests that the dataset contains a balanced distribution of positive and negative cases of diabetes. This balance ensures a fair analysis for predicting the likelihood of diabetes in individuals.

**Histogram of Features**:

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/bf7c886b-7a13-4f88-84dc-dc7900ba16d4)

The histograms visualize the distribution of each feature in the dataset, providing insights into central tendency, spread, and skewness of values. This aids in identifying patterns, outliers, and understanding the data's overall structure. From the above representation it is observed that there are discrete and continuous features.

**Age-wise Distribution of Diabetes Diagnosis**:

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/d58332ac-88bc-46b8-872f-e6faf43b3530)

The line plot illustrates the count of individuals diagnosed with diabetes across different age categories. It is observed that the number of diabetes-positive cases begins to increase notably from the age group of 40-44 years, with a peak in the age range of 60-70 years. This suggests that the likelihood of diabetes tends to rise with advancing age, peaking in the later years of adulthood.

**Distribution of Diabetes Diagnosis Across BMI Categories**:

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/7e939822-d7dc-413a-a650-17315505e599)

The line plot depicts the count of individuals diagnosed with diabetes across different Body Mass Index (BMI) values. According to BMI categories,

  - values below 18.5 indicate underweight
  - 18.5 to 24.9 represent a healthy weight
  - 25.0 to 29.9 indicate overweight
  - 30.0 or higher signify obesity.

From the graph, it's evident that the majority of positive diabetes cases fall within the BMI range of 25 to 40, which corresponds to the overweight category. This suggests a notable association between higher BMI values and an increased likelihood of diabetes.

**Impact of High Blood Pressure and High Cholesterol on Diabetes Diagnosis**:

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/2e7ad00b-b90b-4f1f-b2ce-40b828447c67)

The bar plot illustrates the impact of high blood pressure (HighBP) and high cholesterol (HighChol) on the likelihood of diabetes. Observations indicate that:

  - Individuals with neither high blood pressure nor high cholesterol have significantly fewer positive diabetes cases.
  - Those with high cholesterol but no high blood pressure show a slightly higher proportion of negative diabetes cases compared to positive cases.
  - Conversely, individuals with high blood pressure but no high cholesterol exhibit slightly more positive diabetes cases than negative ones.
  - Notably, individuals with both high blood pressure and high cholesterol demonstrate a higher proportion of positive diabetes cases
  - Overall, the plot suggests that the presence of both high blood pressure and high cholesterol may significantly increase the likelihood of diabetes.

**Correlation Between General Health Conditions and Diabetes Diagnosis**:

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/33a72f3c-41a6-4e27-86dd-aa879914165d)

The bar plot illustrates the distribution of diabetes status based on general health conditions. It reveals a clear trend: as the general health condition deteriorates from excellent to poor, the proportion of individuals with diabetes increases. This trend suggests a strong correlation between declining general health and an increased likelihood of having diabetes, indicating the importance of maintaining good health to prevent diabetes.

**Diabetes Status Among Individuals with Stroke and Heart Disease**

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/0340b64d-3144-46c4-8621-06da9c14b18d)

The pie chart illustrates the diabetes status among individuals who have both stroke and heart disease or attack. It reveals that a significant percentage of these individuals have diabetes, indicating a potential correlation between stroke, heart disease, and diabetes. This suggests a higher likelihood of individuals with stroke or heart disease also having diabetes.

**Correlation Between Features and Diabetes Diagnosis**:

![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/8a5cd64f-e021-4016-9333-de565e6d6acc)

The correlation bar plot visualizes the strength and direction of the relationship between each feature and the target variable (Diabetes_binary), with positive coefficients indicating a higher likelihood of diabetes and features closer to |1| having stronger associations with the target.

It is observed from the graph that,

General health, high blood pressure, BMI, high cholesterol, and age exhibit higher positive correlations with diabetes, suggesting that these factors are more strongly associated with the likelihood of diabetes.
Conversely, education and income have the lowest correlations with diabetes, indicating weaker associations with the target variable.



## 4.3 Potential Features/Predictors:

  The potential features or predictors considered for predicting the likelihood of diabetes have been chosen based on their significance to the target variable, determined through feature importance analysis using the correlation matrix.

  - General Health: General health, rated on a scale from 1 to 5, where lower values signify better health, shows the strongest correlation with the likelihood of diabetes among all features.
  - High Blood pressure: High blood pressure (HighBP), indicating the presence (1) or absence (0) of hypertension, demonstrates a significant correlation with the likelihood of diabetes, suggesting a direct relationship between the two variables.    - Fruit and Vegetable Consumption: Reflects the frequency and quantity of fruit or vegetable intake in the individual's diet, which can impact overall nutrition and health status, potentially affecting diabetes risk.    - Physical Activity Level: Indicates the amount and intensity of physical activity or exercise performed by the individual, which is associated with improved health outcomes and reduced risk of diabetes.
  - BMI: BMI is a measure of body fat based on height and weight. Higher BMI values are often associated with increased risk of diabetes.
  - High Cholesterol: High cholesterol (HighChol), denoting the presence (1) or absence (0) of elevated cholesterol levels, exhibits a notable correlation with the likelihood of diabetes, indicating a potential connection between these health conditions.
  - Age: Age is a significant risk factor for diabetes, with the prevalence of the condition generally increasing with age.
  - Difficulty Walking: Reflects any challenges or limitations the individual may experience in walking, which can be indicative of mobility issues and potentially underlying health conditions related to diabetes.
  - Heart Disease/Attack: Indicates whether the individual has a history of diagnosed heart disease or heart attack, which are both significant risk factors for diabetes and overall health.

**Correlation of potential features with Diabetes**:
![download](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/78ee5391-80aa-444d-a0f1-4af50d70aa2e)

The pie chart illustrates the relative contribution of each selected feature to the prediction of diabetes.


# ML Model

## 5.1 Model Selection:
The dataset underwent training with various models including random forest regression, XGBoost, SVM, decision trees, and logistic regression to determine the optimal performer. Ultimately, logistic regression emerged as the top-performing model, achieving an accuracy of 80%. Consequently, the application was developed utilizing logistic regression.

## 5.2 Model Training:
**Train-Test split**:
The model underwent training and testing using an 80-20 split, with 80% of the data used for training and the remaining 20% for testing. Subsequently, the trained model was assessed on the testing set to gauge its effectiveness in predicting diabetes risk.

**Python packages**:
The Python libraries utilized in developing this product include scikit-learn, matplotlib, pandas, and seaborn. The development process occurred across various environments such as personal laptops, Google Colab, and Jupyter notebooks.

**Model evaluation**:
To measure and compare the performance of the models, several metrics including accuracy score, F1 score, and ROC-AUC score were used. The accuracy score provides an overall measure of correct predictions, while the F1 score balances precision and recall. Additionally, the ROC-AUC score evaluates the model's ability to distinguish between positive and negative classes across different thresholds. Examining these metrics comprehensively, aids in making informed comparisons and selections among the models based on their performance.

## 5.3 User Input:
The model collects user input for health-related parameters such as BMI, general health rating, high blood pressure, high cholesterol, age, difficulty in walking, and history of heart disease or attack using validation functions. It then creates a subset of the user's data and employs a trained logistic regression model to predict the likelihood of diabetes based on the provided information. Finally, it provides feedback to the user regarding their potential risk of diabetes.

Implemented code that defines a set of functions to validate user input for various health-related parameters such as BMI, general health rating, high blood pressure, high cholesterol, age, difficulty in walking, and history of heart disease or attack. Each function ensures that the input meets specified criteria, such as being within certain ranges or adhering to predefined options. For instance, the 'get_bmi()' function prompts the user to input their BMI, ensuring it falls within the range of 5 to 100. Similarly, the 'get_gen_health()' function validates the user's input for general health rating, accepting values between 1 and 5 according to the provided scale. The 'validate_user_input()' function orchestrates the process by sequentially invoking these input validation functions and providing prompts for the user to enter relevant information. Overall, these functions aim to ensure the accuracy and reliability of user-provided health data for subsequent analysis or application usage.

## 5.4 Predicting Diabetes Risk for User Input:
The model predicts whether the user is at risk of developing diabetes based on their input and provides insights into the contributing factors. It offers live graph visualization, allowing users to compare their feature values with those of the dataset population, aiding in better understanding.

# 6. StreamLit Application
The Streamlit app utilizes logistic regression to predict diabetes risk based on user-provided data, considering features such as BMI, general health, high blood pressure, high cholesterol, age, difficulty walking, and heart disease or attack. Users input their information through a sidebar interface, receiving live updates on the predicted diabetes risk. Additionally, the app allows users to compare their specifics with the dataset's distribution across various categories like BMI, general health, and age. Furthermore, an option is available for Contribution Analysis, elucidating the relative importance of each feature in shaping the prediction result.
![image](https://github.com/AdaviVarshita/UMBC-DATA606-Capstone/assets/115598151/343d9619-95fc-4f14-961c-802fbb6cfa97)


# 7. Conclusion
In conclusion, the logistic regression model embedded within the Streamlit app offers a practical solution for individuals to evaluate their diabetes risk based on input data with 80% accuracy. The app provides real-time feedback, empowering users to proactively manage their health. Its potential applications extend to healthcare settings, where it can assist healthcare professionals in screening individuals for diabetes risk and promoting preventive measures.

**Limitation**:

Despite its utility, the current model has limitations. It relies on a limited set of features, potentially overlooking other relevant factors influencing diabetes risk. Furthermore, the model's performance may vary based on the quality and completeness of the input data. Addressing these limitations requires integrating additional features into the model and improving data quality through rigorous validation processes.

**Lessons learnt**:

Throughout this project, several lessons have emerged. Firstly, user-friendly interfaces such as the Streamlit app facilitate seamless interaction and engagement with predictive models. Secondly, the importance of continuous model evaluation and refinement to ensure accuracy and reliability cannot be overstated. Lastly, collaboration between domain experts and data scientists is vital for developing robust predictive models that address real-world healthcare challenges effectively.

**Future scope**:

Looking ahead, future research directions include integrating additional features related to lifestyle, diet, and medical history to enhance the model's predictive accuracy. Exploring advanced machine learning algorithms and techniques could further improve model performance. Additionally, incorporating personalized recommendations based on individual risk profiles and developing a mobile application version of the tool could expand its reach and impact in promoting preventive healthcare practices.

# 8. References
- ["Diabetes Health Indicators Dataset" on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Research paper: Smith, A., Jones, B., & Patel, C. (2020). Predictive Modeling of Diabetes Risk Factors Using Machine Learning Techniques. Journal of Health Informatics Research, 10(3), 215-228.
- Book: Brown, S. J., & Smith, R. L. (2018). Machine Learning in Healthcare: A Practical Approach. Chapman and Hall/CRC.
- Research paper: Yousuf, A., Khan, N. A., Bokhari, R. H., & Siddiqi, M. H. (2020). Prediction of Diabetes using Machine Learning Algorithms. Pakistan Journal of Medical Sciences, 36(S4), S48–S53.
- Research paper: Qin, Y., Wu, J., Xiao, W., Wang, K., Huang, A., Liu, B., Yu, J., Li, C., Yu, F., & Ren, Z. (2022). Machine Learning Models for Data-Driven Prediction of Diabetes by Lifestyle Type. International Journal of Environmental Research and Public Health, 19(22), 15027.
