#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                'bonus',
                'deferral_payments',
                'deferred_income',
                'director_fees',
                'exercised_stock_options',
                'expenses',
                'from_messages',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'long_term_incentive',
                'other',
                'restricted_stock',
                'restricted_stock_deferred',
                'salary',
                'shared_receipt_with_poi',
                'to_messages',
                'total_payments',
                'total_stock_value',
                'from_poi_pct',
                'to_poi_pct'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

data_dict.pop("LOCKHART EUGENE E",0)

data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for name, item in my_dataset.items():
    my_dataset[name]['poi_email_pct'] = 0
    my_dataset[name]['from_poi_pct'] = 0
    my_dataset[name]['to_poi_pct'] = 0
    total_messages = item['from_messages'] + item['to_messages']
    if total_messages != 0 and item['from_poi_to_this_person'] != 'NaN' and item['from_this_person_to_poi'] != 'NaN':
        my_dataset[name]['poi_email_pct'] = ((float(item['from_poi_to_this_person'])+
        float(item['from_this_person_to_poi']))/(float(total_messages)))
        if item['from_messages'] != 0:
            my_dataset[name]['from_poi_pct'] = (float(item['from_poi_to_this_person'])/ float(item['to_messages']))
        if item['to_messages'] != 0:
            my_dataset[name]['to_poi_pct'] = (float(item['from_this_person_to_poi'])/ float(item['from_messages']))

### Extract features and labels from dataset for local testing
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#from sklearn.cross_validation import StratifiedShuffleSplit
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)
#folds = 1000
#cv = StratifiedShuffleSplit(
#     labels, folds, random_state=random)    

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'entropy', max_features = 'sqrt')
#decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', max_features = 'sqrt')

test_classifier(clf, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)