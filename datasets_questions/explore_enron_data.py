#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)

count = 0
for thing in enron_data:
    if enron_data[thing]["poi"] == True:
        count += 1
print count


print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]
print enron_data["LAY KENNETH L"]["total_payments"]

salary_count = 0
for thing in enron_data:
    if enron_data[thing]["salary"] != "NaN":
        salary_count += 1
print "salary count", salary_count

email_count = 0
for thing in enron_data:
    if enron_data[thing]["email_address"] != "NaN":
        email_count += 1
print "email count", email_count

pay_count = 0
for thing in enron_data:
    if enron_data[thing]["poi"] == True:
        if enron_data[thing]["total_payments"] == "NaN":
            pay_count += 1
print "pay count", pay_count

print float(pay_count)/float(count)

nan_pay_count = 10
for thing in enron_data:
    if enron_data[thing]["total_payments"] == "NaN":
        nan_pay_count += 1
print "nan pay count", nan_pay_count

print float(nan_pay_count)/float(len(enron_data)+10)