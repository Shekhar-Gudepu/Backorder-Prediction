<h1> Product Backorder Prediction </h1>
<h2> 1. Business Problem </h2>

### Description

#### What does a backorder mean ?

An item on backorder is an out of stock product that is expected to be delivered by a certain date once it is back in stock. Businesses will often still sell products on backorder with the guarantee to ship them to the buyer once their inventory has been replenished.
Backordering an item means the shopper can buy the item now and receive it at a future date when the item is in stock and available.

### Problem statement: 

The objective of the problem is to predict products that have higher chances of getting into backorder based on historical data.
So using the existing data,a binary machine learnig model should be built to classify the products that might run into backorders.

#### Data Source:
https://www.kaggle.com/c/untadta

### Requirements:
- Python 3.6.8
- Scikit-learn 0.22.1
- imblearn 0.6.1

### Scripts
1.  Back order Prediction EDA : This file contains Exploratory Data Analysis, feature engineering, feature encoding , feature binning performed on the data.
2.  Back order Prediction : contains various machinelearning models trained on the data and the final evaluation result of various models.

<h2> 2.Solving the problem with Machine Learning</h2>
 <h3> Data Overview </h3>
 
- Train data consists of 1687861 entries whereas test data consists of 242076 entries.
- Both Train and test files have 23 columns.    

- Data fields:
    - sku - unique code for identifing each item
    - national_inv- Current inventory level of component
    - lead_time -Transit time
    - in_transit_qty - Quantity in transit
    - forecast_x_month - Forecast sales for the net 3, 6, 9 months
    - sales_x_month - Sales quantity for the prior 1, 3, 6, 9 months
    - min_bank - Minimum recommended amount in stock
    - potential_issue - Indictor variable noting potential issue with item
    - pieces_past_due - Parts overdue from source
    - perf_x_months_avg - Source performance in the last 6 and 12 months
    - local_bo_qty - Amount of stock orders overdue
    - X17-X22 - General Risk Flags
    - went_on_back_order - Product went on backorder
    
### Performance metrics:
     - area under the Receiver Operator Characteristic and precision-recall curves,
     - f1-score
     - Confusion Matrices.
     
<h3> Distribution of class variable </h3>

Dataset contains 23 columns, among them 15 features are numerical,7 are categorical and one target variable('went_on_backorder) with values 'YES' for a backorder and 'NO' for a non-backorder.
Data is extremely imbalanced with only 0.7% of the data belonging to backorder products.The ratio of class labels is 1:148 (1 backorder per 148 non-backorder products).Class distribution is as follows,

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/data_skew.png?raw=true)

<h2>Resampling Techniques for handling class imbalance</h2>

1.Random Oversampling: Data in the minority class has been randomly repeated multiple times to match the no.of datapoints in the majority class.

2.Class weight parameter of sklearn module: sklearn provides a parameter 'class_weight' through which minority datapoints can be given more weight when the model fits the data.

3.Imblearn module's ensemble models: Imblearn provides various ensemble models where each base learner is fed with data that is oversampled or undersampled.Imblearn's 'BalancedRandomForestClassifier' has been used where each bootstrap sample is undersampled.


<h2>Feature Engineering</h2>

All the categorical features(general risk flags) are encoded as 'YES':1 and 'NO':0

Following are the general causes of a backorder:

1.Unusual demand of a product

2.Error in forecast sales

3.Manufacturing issues

4.Low safety stock

Considering the above causes of backorders new features like
Unusual demand,error in forecast,low safety stock have been generated
as follows,

- Unusual demand: Compare the sales of 1, 3, 6, 9 months of the
product,if there is a significant increase in the sales of one of the 3
month period, then flag the product as 1 else 0.

- Error in forecast: Sales - Forecast

- Low safety stock: if current inventory is less than minimum
recommended stock then flag the product 1 else 0.

<h2>Machine Learning models and their results</h2>

Below are the results obtained from applying various ML models,


|      MODEL       | ROC-AUC | PR-AUC | PRECISION | RECALL | +VE F1-SCORE |
|------------------|---------|--------|-----------|--------|--------------|
|      Logit       |   0.9   |  0.09  |    0.05   |  0.87  |     0.09     |
|   Logit_binned   |   0.89  |  0.09  |    0.05   |  0.8   |     0.09     |
|        DT        |   0.93  |  0.21  |    0.07   |  0.86  |     0.12     |
|    DT_binned     |   0.92  |  0.18  |    0.06   |  0.84  |     0.12     |
|        RF        |   0.96  |  0.39  |    0.17   |  0.77  |     0.27     |
|    RF_binned     |   0.95  |  0.32  |    0.13   |  0.78  |     0.22     |
|       XGB        |   0.96  |  0.44  |    0.17   |  0.79  |     0.28     |
|    XGB_binned    |   0.96  |  0.41  |    0.29   |  0.61  |     0.39     |
|  RF_oversampled  |   0.95  |  0.24  |    0.13   |  0.7   |     0.23     |
| RF_undersampled  |   0.95  |  0.21  |    0.07   |  0.89  |     0.13     |
| XGB_oversampled  |   0.97  |  0.5   |    0.63   |  0.39  |     0.48     |
| XGB_undersampled |   0.95  |  0.19  |    0.07   |  0.89  |     0.13     |
|   Balanced_RF    |   0.95  |  0.27  |    0.08   |  0.87  |     0.14     |
|   EasyEnsemble   |   0.93  |  0.17  |    0.06   |  0.86  |     0.09     |


Based on the above performance table, Random Forest trained on undersampled data performed well on the data and can be used for predicting backorders in realtime.
