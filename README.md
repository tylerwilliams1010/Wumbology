# Wumbology
WUMBO Algorithm (Weighted UID-filtered anoMaly-Based Outlier detection)

WUMBO is designed to identify outliers (given an alpha) for a multi-dimensional dataset.

# The approach


## Removing UID's from the equation
This algorithm was created with use cases for Cybersecurity, Insider Threat, and Fraud in mind. In these areas, there are times where measurements need to be identified where identity is removed as a factor. For example, say a dataset measures different examples of user behavior alongside potential data exfiltration, one single user could show up in multiple rows. If using an algorithm like DBSCAN, a user could cluster with itself - reducing the ability to identify behavior as an outlier.

## Parameterless (Almost)
Unsupervised Clustering algorithms such as DBSCAN and OPTICS require parameters such as epsilon and minimum points, which can be difficult to identify accurately. Additionally, the metrics can have some unintended consequences - the goal for WUMBO is to require minimal parameters. The only parameter that needs to be chosen is an alpha, which is used to identify fairly close to the chosen alpha-top outliers in the dataset.

## Local Density and Average Distance

## Chebyshev's Inequality for Outlier Detection

## Scaling with Robust Scaler
