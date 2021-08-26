# Wumbology
WUMBO Algorithm (Weighted UID-filtered Multimetric-Based Outlier detection)

![Image of Wumbo Algo being used IRL.](https://assets.change.org/photos/0/ui/gi/ssUIGiKyMdDGReV-800x450-noPad.jpg?1530521121)

**Purpose**:
Unsupervised Learning algorithm designed to identify outliers in datasets, created originally for use in Cybersecurity, Insider Threat, and Fraud. The purpose of the algorithm was to improve upon DBSCAN/OPTICS-style algorithms by implementing the following:
- **_UID-Filtering_**
When working with data in User Entity Behavior Analytics, some larger datasets may have multiple rows corresponding to the same User/Entity. In these situations, if specific behavioral measures are the same across user behavior then it becomes possible for user behavior metrics to provide increased density to an arbitrary region. 
For example: In searching for Data Exfiltration Activity, a user may have multiple heuristic measures that can point to risk of Data Exfiltration. If the expected detection for Data Exfiltration detection combines some behavioral characteristics to include data egress into a density-based model, a user could exfiltrate multiple examples of benign data to outbound locations along with one malicious example. In this example, the user has poisoned the density-based model to some degree because multiple examples of user action are able to contribute to the outlier factor of the activity. By filtering out UID's a user is unable to impact a dataset in this way since their data is compared to the environment without possibility of referencing their own behavior as potential neighbors.
- **_Use an Alpha and Provide a Weight_**
With models such as DBSCAN, parameters such as epsilon and minimum points need to be specified, which can be somewhat difficult to properly assign. WUMBO uses an alpha metric to approximately return some top (alpha * 100)% of results based on a combination of local and environment-wide factors. Additionally, like Isolation Forest or Local Outlier Factor, this unsupervised learning algorithm outputs a Weighting of Outlierness in the form of a "Risk Score."
- **_Identify and expect local clusters of arbitrary size and density_**
Large networks are going to have many users with small clusters of different arbitrary behaviors. Imagine a network with 25,000 users that delivers financial services to the general public. Multiple clusters of users are going to exist when measuring heuristics of behavior: an Application Development team will use certain softwares and servers differently than a Sales Development team. Even within groups of an organization, such as sales, there are going to exist an arbitrary number of even smaller subgroups. Within Sales, there are Managers, Graphics Designers, people who travel to customer sites, people who work on-site only, etc. Many SIEM products offer enrichment by bucketing users into organizational or business unit groups, but this classification can tend to be mis-leading or noisy because of the subgroups. With WUMBO, a user is compared to it's k-Nearest Neighbors across multiple metrics to determine general local measures of density (local/similar working group) and then measured in the next later as an ensemble against the entire dataset (environment=wide) to identify behaviors without rigid classification requirements. This enables general anomaly detection across large datasets with arbitrarily-sized subgroups of behavior and a representation of those data points within those groups measured across the environment as a whole.


**Process**:
WUMBO first filters out UID (to prevent one user with multiple points from clustering on itself or potentially poisoning some region of the data), then establishes each row's k-Nearest Neighbors (where k is defined by 5>=sqrt(N)<=50). From each row and kNN, multiple optional metrics are used to identify "Outlierness" to include max distance, average distance, and inverse kernel density. These measurements are passed on as features into the next layer where now all data points are measured using kernel density to define each data point's "Risk Score" and "Outlier" is defined as a 1 or 0 dependent on the declared alpha value and the final "Risk Score" for the data point.

