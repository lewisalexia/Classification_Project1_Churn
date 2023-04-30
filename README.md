# To Churn or not to Churn: Telco
# Project Description
 
Telco has been experiencing high amounts of customer churn in the past year. 
 
# Project Goal
 
* Discover drivers of churn in Telco's customer base.
* Use drivers to develop a machine learning model to classify customers as ending in churn or not ending in churn.
* Churn is defined as a customer leaving the company. 
* This information could be used to further our understanding of which elements contribute to or detract from a customer's decision to churn.
 
# Initial Thoughts
 
My initial hypothesis is that internet customers are churning because of monthly contract cost.
 
# The Plan
 
* Acquire data from MySQL Server (using Codeup credentials in env.py file).
 
* Prepare data
   * Look at the data frame's info and note:
		* nulls
		* corresponding value counts
		* object types
		* numerical types
		* names of columns
 
* Explore data in search of drivers of churn
   * Answer the following initial questions:
       1. Is phone service and fiber internet related in any way?
       2. From a revenue standpoint, where should we focus our initial efforts?
       3. Are month to month contracts causing churn? Why?
       4. Were average monthly charges higher for churned members?
       
* Develop a Model to predict if a customer will churn:
   * Use drivers identified in the explore phase to build predictive models of different types.
   * Evaluate models on the train and validate datasets.
   * Select the best model based on the highest accuracy.
   * Evaluate the best model on the test data.
 
* Draw conclusions
	* Merge findings back onto the customer base to identify customers with a high chance of churning.
	* Make recommendations to minimize churn.
	* Identify takeaways from company data.	 
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Churn| **target** Whether or not a customer has churned|
|Phone Service| Whether or not the customer has a phone line|
|Internet Service Type| Customer's type of internet (Fiber optic, DSL, None)|
|Total Charges| The total charges a customer has paid|
|Contract Type| Customer contract type (Month-to-month **focus**, One year, Two year)|
|Multiple Lines| Whether a customer has more than one line|

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from MySQL servers using your own Codeup credentials stored in an env.py file.
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* Everyone who has fiber internet, has phone service. These are related features.
* Cost of monthly one-line phone service negatviely affects fiber internet metrics.
* 1,297 churned phone customers cost Telco 2.6 million dollars.
* The churn culprit is the one-line, month-to-month, phone contracts.
* There is wiggle room to reduce the price to stem the bleeding.
 
# Recommendations
* There is wiggle room to reduce the price to stem the bleeding.
* Reduce cost by $10 to even the slope a bit.
