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
1.  Back order Prediction EDA : This file contains Exploratory Data Analysis, feature engineering, feature encoding , feature binning performed on data available.
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


<h3> Exploratory Data Analysis</h3>
<h3> Distribution of class variable </h3>

Dataset contains 23 columns, among them 15 features are numerical,7 are categorical and one target variable('went_on_backorder) with values 'YES' for a backorder and 'NO' for a non-backorder.
Data is extremely imbalanced with only 0.7% of the data belonging to backorder products.The ratio of class labels is 1:148 (1 backorder per 148 non-backorder products).Class distribution is as follows,

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/data_skew.png?raw=true)

<h3> Missing values </h3>
- There are no null values in any feature except for lead_time, lead time has nearly 6% of data missing.

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/MissingValues.png?raw=true)

<h3> Analysis on numerical features </h3>
<h4> Distribution plots of Numerical features </h4>

- Below are distribution plots of few numerical features (only few distribution plots are considered below because most of the distribtuions are similar),

##### sales forecast:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/forecast_9_monthnumerical_dist.png?raw=true)

##### transport time:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/lead_timenumerical_dist.png?raw=true)

##### current inventory level:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/national_invnumerical_dist.png?raw=true&s=50)

##### product performance in 12 months:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/perf_12_month_avgnumerical_dist.png?raw=true&s=50)

##### sales in 9 months:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/sales_9_monthnumerical_dist.png?raw=true&s=50)

#### Observations from the plots:

- All the features are extemely skewed to the right with most of the values being present at the initial values.
  
    
- Maximum values for all the features differ between backorder and non-backorder products significantly,maximum values of backorder products tend to be lot lesser than non-backorder products.
    
    
- There are some negative values in the distribution of product performance features which could be due to some error,further analysis on the feature determined that the values of performance should be in the range 0.1-1.0 not negative values.


- Current inventory level feature also has negative values(~1%) which could be due to error while collecting the data,all the feature values are converted to positive values.
    
    
- For all the features, distributions of Backorder products and Non-Backorder products are similar,making it difficult to draw insights about backorder products.Difference in class distributions could have helped classifying the data better.

<h3> Percentile plots: </h3>
Following are the percentile plots of features that gave some insights,

##### forecast sales for 3 months:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/percentiles_forecast3months.png?raw=true)

##### No.of units in transport:

 Percentile values:
 
 - 0 %le	 :0.0 
 - 10 %le	:0.0
 - 20 %le	:0.0
 - 30 %le	:0.0
 - 40 %le	:0.0 
 - 50 %le	:0.0
 - 60 %le	:0.0
 - 70 %le	:0.0
 - 80 %le	:1.0
 - 90 %le	:16.0
 - 100%le	:489408.0
 
![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/percentiles_intransitQty.png?raw=true)

##### Minimum recommended stock:

percentile values:

- 0 %le  	: 0.0
- 10 %le  : 0.0
- 20 %le  : 0.0
- 30 %le 	: 0.0
- 40 %le  : 0.0
- 50 %le 	: 0.0
- 60 %le  : 1.0
- 70 %le 	: 1.0
- 80 %le 	: 3.0
- 90 %le 	: 21.0
- 100 %le : 7669.0

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/percentiles_minStock.png?raw=true&s=50)

##### no.of units due in past from source:

percentile values:

-  0 %le  	 : 0.0
-  10 %le  	: 0.0
-  20 %le  	: 0.0
-  30 %le  	: 0.0
-  40 %le  	: 0.0
-  50 %le  	: 0.0
-  60 %le  	: 0.0
-  70 %le  	: 0.0
-  80 %le  	: 0.0
-  90 %le  	: 0.0
-  100 %le  	: 720.0

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/percentiles_piecesPastdue.png?raw=true)




<h5>Observations:</h5>
All the numerical features values spiked up after the 90th percentile value except for lead time.Upon further analysis on descriptive statstic and percentile plots gave the following insights,
 
 - Based on the analysis on current inventory level, current no.of units held in stock for the items backordered tend to be less than non-backordered items.
 - Distributions of both backorder and non-backorder product's lead time are similar with most of the products lead time being 2,8,12 days.
- Probability of a product being delivered to the retailer in 8 days is slighlty higher for non-backorder products.
- For the feature intransit quantity, there is a difference in backorder and non-backorder products for values greater than 1,backorder products have higher intransit quantity.
- 60% of the products forecast sales are 0 units,only 40% of the products have forecast sales of atleast one unit.
- Typically backorder products sales tend to be high but there isn't any difference between the distributions of backorder product sales and non-backorder product sales.
- There are nearly 40% of products that didn't have any sales in the 9 months and most frequent no.of units sold is 0.
- By looking at the individual sales of each 3 month periods there are products whose sales have decreased over the time and there are products that have sales increased over time.
- One of the reason for backorders is unusual demand or unusual sales of product due to seasonality.Upon checking for sudden spike in sales over time, data points in both backorder and non-backorder have normal increase in sales without any sudden increase.So unusual sales doesn't help in the classification of the backorder products. 
- There are 50% of products whose minimum no.of stock units recommended in the inventory equal to 0.
- 99% of the products have less than 100 recommended stock units.
- 99% of the values are just 0's for units overdue from source in the past.
- Most products performance is greater than 0.6 with most products having the performance ratings 0.99,0.78,0.98.
- Feature local backorder quantity which indicates the amount of stock overdue will not be usefull in predicting the backorder products since nearly all the values are 0's.

Above are some of the insights drawn from descriptive statistics and percentile plots of numerical features.Since all features values spike after 90%le, values upto 90%le are considered for training Machine Learning models.

Negative values in Performance features are removed from the data.

Missing values in lead time feature have been imputed with most frequent value.

<h3> Analysis on categorical features: General risk flags</h3>

##### Deck risk:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/deck_risk_riskFlag.png?raw=true)

##### Oe constraint:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/oe_constraint_riskFlag.png?raw=true)

##### potential issue:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/potential_issue_riskFlag.png?raw=true)

##### ppap risk:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/ppap_risk_riskFlag.png?raw=true)

##### rev stop:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/rev_stop_riskFlag.png?raw=true)

##### stop auto buy:

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/stop_auto_buy_riskFlag.png?raw=true)



<h5>Observations:</h5>

- Generally products that have some risk factors tend get into backorders but there is no significant difference that can be observed between backorder products and non-backorder products due to general risk flags.
- rev_stop and potential issue are not useful for the classification because no product have any of these risks.

<h3>Multivariate Analysis</h3>
<h5>Correlation Matrix</h5>

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/correlation.png?raw=true)

<h5>Pairplot</h5>

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/pairplot.png?raw=true)

<h5>category plots</h5>

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/sales_9_monthcatplot.png?raw=true)

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/forecast_9_monthcatplot.png?raw=true)

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/in_transit_qtycatplot.png?raw=true)

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/lead_timecatplot.png?raw=true)

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/min_bankcatplot.png?raw=true)

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/national_invcatplot.png?raw=true)

![alt text](https://github.com/Shekhar-Gudepu/Backorder-Prediction/blob/main/EDA/perf_12_month_avgcatplot.png?raw=true)

<h5>Observations:</h5>

- Sales till 1st month,3rd month,6th month and 9th month are highly correlted to each other.

- Minimum no of units required in the inventory and no.of units in transport are correlated with both forecast features and sales features.

- Sales of products are high for the products with lesser lead time.There is a decline in the sales when lead time increases.

- There is an increase in performance scores as the forecast values increse.

- products that went on backorder have lesser amount of stock in the inventory,lesser no.of units in transport and lesser minimum no.of units that are required in stock.

- Products that went on backorder have lower sales and forecast values.

- for the products that have higher current inventory level in_transit quantity is less.

- there are less instock units for the products with lead time between 20 and 40.

- sales tend to be slightly higher for products with lesser current inventory level.

- intransit quantity starts decreasing as the forecast sales increases.

- Backorder products are dense at high performance values.

- All the sales features and forecast features are highly correlated to each other,so including any one of the sales feature and any one of the forecast feature will be sufficient for builiding the model.

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

<h2>Resampling Techniques for handling class imbalance</h2>

1.Random Oversampling: Data in the minority class has been randomly repeated multiple times to match the no.of datapoints in the majority class.

2.Class weight parameter of sklearn module: sklearn provides a parameter 'class_weight' through which minority datapoints can be given more weight when the model fits the data.

3.Imblearn module's ensemble models: Imblearn provides various ensemble models where each base learner is fed with data that is oversampled or undersampled.Imblearn's 'BalancedRandomForestClassifier' has been used where each bootstrap sample is undersampled.

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


Based on the above performance table, Random Forest trained on undersampled data seems to perform well on the data and can be used for predicting backorders in realtime.
