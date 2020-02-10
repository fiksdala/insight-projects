# Senti-Mentor
*Helping hospices improve patient and family satisfaction*

*Insight Data Science Project by Alexander Fiksdal*

This project predicts family recommendation rates and estimated mean ratings at the hospice facility level and provides insights into improving those satisfaction measures.

# Project Overview

Hospice care eases end of life for patients and families while avoiding unnecessary hospital admissions and saving money. Hospice providers are required to submit a variety of data to the Centers for Medicare & Medicaid Services (CMS), which are then made available to the public. Moreover, these data are used in Medicare's *Hospice Compare* tool, which allows patients and families to compare providers across a variety of performance and service dimensions. However, provider-submitted data is decentralized and no tools exist to help providers analyze what informs measures of patient/family satisfaction. *Senti-Mentor* addresses these needs by combining multiple sources of hospice provider data and analyzing them using regression and gradient-boosted regression tree models. Specifically, it helps hospice providers address the following questions:

- Are our patients and families satisfied?
 - Visualizations of observed satisfaction measures in the context of state and national distributions
 - For facilities that lack satisfaction data: XGBoost model predictions based on available features 
- What factors predict satisfaction?
 - Summaries of regression models predicting family recommendation rates and average ratings from actionable measures of service quality, services delivered, and facility characteristics.
- What factors should take priority in future interventions?
 - Targeted recommendations constructed by taking into account both the magnitude of regression coefficients and specific provider feature values.

# Step 1: Merge CMS Data Sources

**Data Sources**

[Public Use File](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/PAC2017)

[Consumer Assessment of Healthcare Providers & Systems (CAHPS)](https://data.medicare.gov/Hospice-Compare/Hospice-Compare-Provider-CAHPS-Hospice-Survey-Data/gxki-hrr8)

[General Hospice Provider Information](https://data.medicare.gov/Hospice-Compare/Hospice-General-Information/yc9t-dgbk)

[Hospice Quality of Patient Care Data](https://data.medicare.gov/Hospice-Compare/Hospice-Provider-Data/252m-zfp9)

Raw data files were saved in a folder hidden from github with the following path: data/raw

Python scripts used to clean/combine data are located in the following folder: hospice_project/data/

# Step 2: Feature Engineering

The following features were created using the hospice\_project/features/build_features.py script:

- Estimated Ratings
- Service Quality (Usually + Always %)
- Probably + Definitely Would Recommend %
- Facilities within 30/60/90 miles
- Distance to nearest facility
- Per-Beneficiary Measures of Services Delivered and Charges

# Step 3: Train Models

Two models were trained: Insight Model (for interpretation and recommendations) and Prediction Model (for facilities with missing performance measures). The scripts used for comparing and training final models is located in hospice\_project/models/

Scaling and imputation of missing values used transformer classes located in hospice\_project/data/transformers.py

Other fixed values (variable lists, custom dictionaries, etc.) defined in definitions.py file.

# Step 4: Implementing Everything

Executing the following scripts in this order will produce the files required for the streamlit app to function:

hospice\_project/data/clean\_data.py
hospice\_project/features/build\_features.py
hospice\_project/models/insight\_model\_compare.py
hospice\_project/models/prediction\_model\_compare.py
hospice\_project/models/train\_insight.py
hospice\_project/models/train\_prediction.py


# Step 5: streamlit app

Final tool built as streamlit application. To run locally, type this in the terminal:

streamlit run sentimentor.py


# Directory Structure

Directory structure initialized using cookiecutter:

[https://github.com/MisterVladimir/cookiecutter-data-science/tree/v0.1.0](https://github.com/MisterVladimir/cookiecutter-data-science/tree/v0.1.0)

# Presentation

Google slides summarizing the project can be found [here](https://docs.google.com/presentation/d/1cF2lvRfTB_jk-UUtW0xzQMosyj9oB4Ubf4lO7CFE9sg/edit?usp=sharing).



