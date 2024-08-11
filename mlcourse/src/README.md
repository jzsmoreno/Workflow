# What are Gradient Boosting algorithms?

## Preliminary concepts

### What are decision trees?

A decision tree uses a tree structure to represent a number of possible decision paths and an outcome for each path.

<center>
    <img src="https://miro.medium.com/v2/resize:fit:1104/1*HXQJ8Eb8vNuxzprfKObodQ.gif"width=55% height=55%>
</center>

Consider the example above where a simple decision tree has been constructed for the purpose of estimating a credit range. The idea behind gradient boosting algorithms is to have a set of weak learners (decision trees) in order to have a better prediction making it a more robust algorithm.

## Definition and features

<center>
    <img src="https://talperetz.github.io/Tal-Peretz/mastering_the_new_generation_of_gradient_boosting/photos/gbdts.png"width=55% height=55%>
</center>

In practical terms, Gradient Boosting is a robust ensemble machine learning algorithm known for its versatility and effectiveness. It stands out because it:

- **Excels with structured data**, making it a popular choice for a wide range of applications.
- **Frequently used in winning solutions** in machine learning competitions, showing its competitive advantage.
- **Supports both regression and classification tasks**, offering flexibility in problem-solving.
- **Handles continuous and categorical data**, ensuring broad applicability across different data types.
- **Models both linear and non-linear relationships**, making it adaptable to various underlying patterns in the data.
- **Performs well with datasets of varying sizes**, whether small or large, demonstrating its scalability and robustness.

## Example

Suppose you want to train an algorithm in order to adjust a set of points, as shown in the image below.

- The first step would be to have a baseline estimate using an decision tree, which is the case in the image, but for explanatory purposes, we will use the mean. So our first approach would be, $\hat{y_{1}}^{(0)} = \text{mean}(y_{1})$.

for $f(\mathbf{x}) = y_{1}$

<center>
    <img src="https://almablog-media.s3.ap-south-1.amazonaws.com/image_43_deeb0633cc.png"width=55% height=55%>
</center>

- The second step is to improve this prediction using for example a decision tree. Taking into account the error we have made in the previous step.

$$
\begin{equation}
r_{(0)}(\mathbf{x}) = y_{1} - \hat{y}_{1}^{(0)}
\end{equation}
$$

Therefore, the improvement $i$, due to the decision tree $D^{(i)}(\mathbf{x})$ would be as follows

$$
\begin{equation}
\hat{y_{1}}^{(i+1)} = \hat{y_{1}}^{(i)} + \alpha \cdot D^{(i)}(\mathbf{x})
\end{equation}
$$

where the prediction of the decision tree $D^{(i)}(\mathbf{x})$ models the residue $r_{(i)}(\mathbf{x})$, for $i$ equal to 0 up to $N$. Where the residues are:

$$
\begin{equation}
r_{(i)}(\mathbf{x}) = y_{1} - \hat{y}_{1}^{(i)}
\end{equation}
$$

Thus the prediction of each decision tree $i$ is

$$
\begin{equation}
D^{(i)}(\mathbf{x}) = \hat{r}_{(i)}(\mathbf{x})
\end{equation}
$$

## Differences between Gradient Boosting and Extreme Gradient Boosting algorithm

### 1. Core Concept
- **Gradient Boosting**: Focuses on minimizing the loss function by iteratively adding weak models (usually decision trees)
- **Extreme Gradient Boosting (XGBoost)**: Builds upon Gradient Boosting by incorporating additional optimizations and techniques to improve performance and efficiency.

### 2. Key Differences and Enhancements in XGBoost
- **Regularization**: XGBoost introduces a regularization term to prevent overfitting (includes L1 (Lasso) and L2 (Ridge) regularization).
- **Handling missing values**: XGBoost can handle missing values more effectively than traditional Gradient Boosting.
- **Parallelization**: XGBoost is designed to take advantage of multi-core processors, which significantly speeds up the training process.
- **Hyperparameter tuning**: XGBoost offers a wide range of hyperparameters and options, including learning rate, number of trees, maximum depth, and more.
- **Handling categorical features**: XGBoost has improved handling of categorical features.

For a more interactive explanation, please see [Gradient Boosting explained](https://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html).
