# SugarSense: Predicting Diabetes Risk

UMBC Data Science Master Degree Capstone - DATA606

Guided by:

Dr. Chaojie (Jay) Wang

By:
  - Name - Varshita Adavi
  - University ID - OC83614
  - Github - https://github.com/AdaviVarshita/UMBC-DATA606-Capstone
  - LinkedIn - https://www.linkedin.com/in/varshita-adavi


### Objective
The objective of the project is to develop a predictive model leveraging machine learning techniques within a user-friendly application that predicts an individual's risk of developing diabetes using key health indicators, aiming to empower users with personalized risk assessments and promote early intervention for better health outcomes.

## Background

- **What is it about?**

  The project involves developing a user-friendly application that utilizes machine learning algorithms to predict an individual's risk of developing diabetes based on key health indicators such as gender, age, BMI, blood pressure, and cholesterol levels. By inputting their personal health data, users receive personalized risk assessments, empowering them to take proactive steps towards managing their health and potentially preventing diabetes-related complications. The app aims to raise awareness about diabetes risk factors and promote early detection, ultimately encouraging users to make informed lifestyle choices for better health outcomes.

- **Why does it matter?**

  The project matters because diabetes is a prevalent and potentially life-threatening chronic condition that affects millions of people worldwide. Early detection and intervention are crucial for managing diabetes effectively and reducing the risk of complications such as heart disease, stroke, kidney failure, and blindness. By developing an app that can accurately predict an individual's risk of diabetes based on their health indicators, we can empower individuals to take proactive steps towards preventive care, such as adopting healthier lifestyle habits, monitoring their blood sugar levels, and seeking medical advice if necessary. Ultimately, the project has the potential to improve public health outcomes by promoting early intervention and reducing the burden of diabetes-related morbidity and mortality.

- **What are your research questions?**
  
  - What are the most significant predictors of diabetes risk among various health indicators such as gender, age, BMI, blood pressure, and cholesterol levels?

  - How accurately can machine learning algorithms predict an individual's likelihood of developing diabetes based on their personal health information?

  - What are the differences in diabetes risk prediction models between different demographic groups (e.g., gender, age groups, ethnicities)?

  - How do lifestyle factors (such as diet, exercise, and smoking habits) interact with health indicators to influence the risk of developing diabetes?


## Data 

- Data sources: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- Data size: 6.2 MB
- Data shape: (70692, 22)
- Time period: 2015
- Columns in dataset:
    | Column Name | Data Type | Description |
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


- **Potential Features/Predictors:**

  The potential features or predictors considered for predicting the likelihood of diabetes have been chosen based on their significance to the target variable, determined through feature importance analysis using the Random Forest Classifier in machine learning.
  
    - BMI: BMI is a measure of body fat based on height and weight. Higher BMI values are often associated with increased risk of diabetes.
    - Age: Age is a significant risk factor for diabetes, with the prevalence of the condition generally increasing with age.
    - Health: Balanced maintenance of physical, general, and mental health is crucial in predicting diabetes risk, as their harmonious equilibrium is associated with reduced susceptibility to the condition.
    - Blood pressure: High blood pressure, also known as hypertension, is a common comorbidity of diabetes and is often considered a predictor of diabetes risk.
    - Smoking Status: Indicates whether the individual is a smoker or non-smoker, which can influence the risk of diabetes and overall health outcomes.
    - Fruit and Vegetable Consumption: Reflects the frequency and quantity of fruit or vegetable intake in the individual's diet, which can impact overall nutrition and health status, potentially affecting diabetes risk.
    - Sex: Refers to the biological classification of male or female, which can influence physiological factors related to diabetes risk, such as hormonal differences.
    - Physical Activity Level: Indicates the amount and intensity of physical activity or exercise performed by the individual, which is associated with improved health outcomes and reduced risk of diabetes.
    - High Cholesterol: Indicates elevated levels of cholesterol in the blood, which is a risk factor for cardiovascular disease and may also contribute to insulin resistance and diabetes risk.
    - Difficulty Walking: Reflects any challenges or limitations the individual may experience in walking, which can be indicative of mobility issues and potentially underlying health conditions related to diabetes.
    - Heart Disease/Attack: Indicates whether the individual has a history of diagnosed heart disease or heart attack, which are both significant risk factors for diabetes and overall health.
    - Alcohol Consumption: Reflects the frequency and quantity of alcohol intake by the individual, which can impact overall health and may contribute to diabetes risk, particularly when consumed in excess.
 

## Methodology

1. **Data Collection**: Obtain a dataset containing relevant health indicators and diabetes status for a sample population.
2. **Data Preprocessing**: Clean the data by handling missing values, removing duplicates, and addressing any inconsistencies. Perform feature engineering if necessary to create new features or transform existing ones.
3. **Exploratory Data Analysis (EDA)**: Conduct exploratory data analysis to gain insights into the dataset's characteristics, identify patterns, and understand the relationships between variables.
4. **Feature Selection**: Utilize techniques such as feature importance analysis, correlation analysis, or domain knowledge to select the most relevant features for predicting diabetes risk.
5. **Model Selection**: Choose appropriate machine learning algorithms for binary classification tasks, such as Random Forest Classifier, Logistic Regression, Support Vector Machines, or Gradient Boosting Machines.
6. **Model Training**: Split the dataset into training and testing sets, and train the selected machine learning models on the training data.
7. **Model Evaluation**: Evaluate the performance of the trained models using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Fine-tune hyperparameters if necessary to improve model performance.
8. **Application Development**: Develop a user-friendly application where users can input their personal health information. Integrate the trained machine learning model into the application to predict the likelihood of diabetes based on the user's input.
9. **Testing and Validation**: Test the application thoroughly to ensure its functionality and accuracy in predicting diabetes risk. Validate the model's performance on unseen data to assess its generalization ability.
10. **Deployment**: Deploy the application on a suitable platform, such as a web server or mobile app store, to make it accessible to users.
