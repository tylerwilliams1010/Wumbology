# Wumbology
WUMBO Algorithm (Weighted UID-filtered anoMaly-Based Outlier detection)

Unsupervised Learning algorithm designed to identify outliers in datasets, created for use in Cybersecurity, Insider Threat, and Fraud - use cases that benefit from filtering UID's so that users don't automatically make clusters through excessive amounts of their own behavior. The algorithm iteratively evaluates a dataset by identifying local inverse log probability (from a kernel density function) and average distance to k-Nearest Neighbors - and then combining those metrics into a feature that generally represents its "Outlierness." By using these metrics from a local k-Nearest Neighbors sampling, computation time is generally low.

![Image of Wumbo Algo being used IRL.](https://assets.change.org/photos/0/ui/gi/ssUIGiKyMdDGReV-800x450-noPad.jpg?1530521121)

# The approach

## Removing UID's from the equation
This algorithm was created with use cases for Cybersecurity, Insider Threat, and Fraud in mind. In these areas, there are times where measurements need to be identified and identity is removed as a factor. For example, say a dataset measures different examples of odd user behavior alongside potential data exfiltration, one single user could show up in multiple rows of a larger dataset. If using an algorithm like DBSCAN, a user could cluster with itself - reducing the ability to identify behavior as an outlier.

## Parameterless (Almost)
Unsupervised Clustering algorithms such as DBSCAN and OPTICS require parameters such as epsilon and minimum points, which can be difficult to identify accurately. Additionally, the metrics can have some unintended consequences (read about UID filtering above) - the goal for WUMBO is to require minimal parameters. The only parameter that needs to be chosen is an alpha, which is used to identify fairly close to the chosen alpha-top outliers in the dataset. In this implementation, we don't care to identify individual clusters - only outliers from the given dataset.

## Local Density and Average Distance
First, a k-Nearest Neighbors value is selected that scales with the size of the dataset - this is calculated by taking the square root of the distinct count of UID's in the dataset, bounded with a minimum of 5 and maximum of 50. Then the dataset is filtered to remove the UID - a new array is created by identifying the k-Nearest Neighbors of each individual point, and then calculating both log likelihood kernel density as well as the average Euclidean distance to it's k-Nearest Neighbors.

## Chebyshev's Inequality for Outlier Detection
In Data Science for Security Information and Event Management, oftentimes datasets have no expectation of being 'normal.' In many cases, using Chebyshev's Inequality for evaluation of the meaning of standard deviation is incredibly useful. Wikipedia has a great page to read on the topic: (https://en.wikipedia.org/wiki/Chebyshev%27s_inequality#:~:text=In%20probability%20theory%2C%20Chebyshev's%20inequality,certain%20distance%20from%20the%20mean)
