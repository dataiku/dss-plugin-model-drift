# Evaluating the Data Drift through recipes

## Goal

We want to come up with a plugin recipe for analysing the 
drift of two or many datasets. The drift implies the 
distinction of distributions between the datasets but it can
also means a shift in term of distribution. The concept
of drift is not direct to grasp. Are we speaking of an
outlier detection or distribution disimilarity ?

## Content

In this example, we would like to provide a recipe that take
two datasets and compute a disimilarity measure.

## Secondary aim

Define the abstract concepts of surrogate models and other
stuff. 

## Ideas

The concepts of dissimilarity depends on the boundary,
if we use a linear classifier or deep learning, this
will not be the same definition in our model.

Also, this work is tightly close to the Scikit model
object and the preprocessor objects.