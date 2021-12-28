<h1> Product Backorder Prediction </h1>
<h2> 1. Business Problem </h2>

### Description

#### What does a backorder mean ?

An item on backorder is an out of stock product that is expected to be delivered by a certain date once it is back in stock. Businesses will often still sell products on backorder with the guarantee to ship them to the buyer once their inventory has been replenished.
Backordering an item means the shopper can buy the item now and receive it at a future date when the item is in stock and available.

Material backorder is a common supply chain problem, impacting an inventory system service level and effectiveness.Identifying parts with the highest chances of shortage prior its occurrence can present a high opportunity to improve an overall company’s performance.
### Problem statement: 

The objective of the problem is to predict products that have higher chances of getting into backorder based on historical data.
So using the existing data,a binary machine learnig model should be built to classify the products that might run into backorders.

<h3> Business Objectives/Constraints: </h3>
Following are the Objectives/Constraints to be considered while building the model,


- Misclassification could be a problem based on the space available in the warehouse.
- Interpretability is an important aspect,knowing why the model predicts backorder could help the seller to determine the reorder point.
- Latency is not an important aspect.

#### Data Source:
https://www.kaggle.com/c/untadta

#### Research paper:
https://www.researchgate.net/publication/319553365_Predicting_Material_Backorders_in_Inventory_Management_using_Machine_Learning


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
     - Confusion Matrices.


<h2> 3.Exploratory Data Analysis</h2>
