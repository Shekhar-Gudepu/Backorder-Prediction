# Backorder-Prediction

## What does a backorder mean and ?

An item on backorder is an out of stock product that is expected to be delivered by a certain date once it is back in stock. Businesses will often still sell products on backorder with the guarantee to ship them to the buyer once their inventory has been replenished.
Backordering an item means the shopper can buy the item now and receive it at a future date when the item is in stock and available.

Material backorder is a common supply chain problem, impacting an inventory system service level and effectiveness.Identifying parts with the highest chances of shortage prior its occurrence can present a high opportunity to improve an overall company’s performance.

# Problem statement: 
Predict products that have higher chances of getting backorders to help manage inventory of a bussiness.

# Data Source:
https://www.kaggle.com/c/untadta

# Research paper:
https://www.researchgate.net/publication/319553365_Predicting_Material_Backorders_in_Inventory_Management_using_Machine_Learning


# Solving the problem with Machine Learning:

### Performance metrics used:
     - area under the Receiver Operator Characteristic and precision-recall curves,
     - Confusion Matrices.

### Models used: 
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - XGBoost
     - EasyEnsemble
     
- This repository consists of a jupyter notebook where above mentioned machine learning classifiers are investigated in order to propose a predictive model for the problem.

- This is case of high class imbalance, where the relative frequency of items that goes into backorder is extemely rare when compared to items that do not.

- Data contains 23 columns which include general risk flags,current inventory level,Sales,Forecast etc.,EDA is performed on all features to draw useful insights.

- Generation of new features have been useful for few models.

- Sampling techniques like Random Oversampling and SMOTE are employed in this particular task.

- Among all the models XGBoost peformed well with better resuls in all performance metrics.

