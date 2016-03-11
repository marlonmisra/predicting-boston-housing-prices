import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import grid_search

numbers = [1,2,3,4,5]
mean = np.mean(numbers)
median = np.median(numbers)
std = np.std(numbers)

numbers_np = np.array([1,2,3,4,5], float)
element_0 = numbers_np[0]
first_2 = numbers_np[0:2]
numbers_np[0] = 10 #change an element

two_d = np.array([[1,2,3], [4,5,6], [7,8,9]])
element_0_0 = two_d[0, 0]

#you can add subtract, and multiply (element by element) np arrays
#you can also use the np.dot function


#making a series
series_1 = pd.Series(["marlon", 22, "san francisco"])

#you can also manually assign indices
series_2 = pd.Series(["marlon", 22, "san francisco"], index = ["name", "age", "city"])

#you can index series
series_2_name = series_2["name"]

#can also use boolean operators to select items
age_true_boolean = series_2 == 22
age_true_only_age = series_2[series_2 == 22]


#creating data frames

data = {'year': [1990, 2000, 2010, 2020], 
		'winner': ['s04', 'arsenal', 'barca', 'rmd'],
		'points': [30, 28, 14, 42]}

data_df = pd.DataFrame(data)

desc = data_df.describe()
types = data_df.dtypes
head = data_df.head()
tail = data_df.tail()


#medals exercise

'''
Create a pandas dataframe called 'olympic_medal_counts_df' containing
the data from the table of 2014 Sochi winter olympics medal counts.  

The columns for this dataframe should be called 
'country_name', 'gold', 'silver', and 'bronze'.  

There is no need to  specify row indexes for this dataframe 
(in this case, the rows will automatically be assigned numbered indexes).

You do not need to call the function in your code when running it in the
browser - the grader will do that automatically when you submit or test it.
'''

countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
             'Netherlands', 'Germany', 'Switzerland', 'Belarus',
             'Austria', 'France', 'Poland', 'China', 'Korea', 
             'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
             'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
             'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

olympic_medal_counts = {'countries': pd.Series(countries),
                         'gold': pd.Series(gold),
                         'silver': pd.Series(silver),
                         'bronze': pd.Series(bronze)}

olympic_medal_counts_df = pd.DataFrame(olympic_medal_counts)

#avg bronze medals for countries with at least one gold
avg_bronze_at_least_one_gold = np.mean(olympic_medal_counts_df[olympic_medal_counts_df.gold > 0]['bronze'])

#average medal count
avg_medal_count_one_medal = olympic_medal_counts_df[['gold', 'silver', 'bronze']].apply(np.mean) 

#points exercise

points = np.dot(olympic_medal_counts_df[['gold', 'silver', 'bronze']], [4,2,1])

olympic_points = {
	'country_name': olympic_medal_counts_df.countries,
	'points': points
}
olympic_points_df = pd.DataFrame(olympic_points)


d = {'age': [10, 13, 23, 30],
	 'weight': [40, 45, 89, 120]}

d_df = pd.DataFrame(d)

means = d_df.apply(np.mean) #using apply 

criteria_met = d_df['weight'].map(lambda x: x > 44) #using map

criteria_met_df = d_df.applymap(lambda x: x > 50) #using applymap for every df

#dot product
a = [1,2,3]
b = [4,5,6]
dotp = np.dot(a,b)


#-----------------
#gaussian naive bayes

x = np.array([[-1,-1], [-2,-1], [-3,-2]])
y = np.array([1,2,2])
clf = GaussianNB()
clf.fit(x, y)
pred = clf.predict([[-0.8, -1]])


#---- accuracy implementation
y_pred = [1,2,1,5]
y_true = [1,2,1,4]
acc_score = accuracy_score(y_pred, y_true) #75%

#splitting data into train/test
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=0)0)


#------ grid search cv exampleelp

parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
#clf.fit(iris.data, iris.target)










