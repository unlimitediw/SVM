# Use SVM to solve Adult Census problem

1. [Project Description](#project)
2. [Dataset Preprocessing and Interpretation](#preprocessing)
3. [Linear Soft Margin SVM](#svm)
4. [Performance](#performance)


<a name="project"></a>
## Project Description

* In this exercise, I implement my own linear soft SVM with rbf, linear and polynomial kernel to solve the Adult Census Income problem. I first do the data preprocessing to abandon the rows containing '?' and select the features with highest priority as model input features. After that, I design and train my linear soft SVM model with different kernels and validate it by 10-fold-cross validation while visualize it like drawing boundary at the same time. In the model training process I also compare the performance of my model with different kernels, different hyperparameters such as 'C', 'sigma', 'tolerance' and so on. At the end. I also add the bagging methods to improve my svm model and evaluate a boosting methods to compare with svm to compare with the svm model. I finally using learning curves to show the performance for different methods.

<a name="preprocessing"></a>
## Dataset preprocessing and interpretation
There are two parts of preprocessing: data purify, feature selection
* Data purify: removing the rows with data containing '?'
    coding:
    
        data = data[data.occupation != '?']
        keys = data.keys()
        for key in keys:
            if data[key].dtype != 'int64':
        data = data[data[key] != '?']
        


<a name="svm"></a>
## Linear Soft Margin SVM



<a name="performance"></a>
## Performance
