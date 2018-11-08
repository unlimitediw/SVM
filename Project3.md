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
There are three parts of preprocessing: data purify, dealing with discrete features and feature selection
* Data purify: removing the rows with data containing '?'

   * coding:
    
         data = data[data.occupation != '?']
         keys = data.keys()
         for key in keys:
             if data[key].dtype != 'int64':
         data = data[data[key] != '?']
   * explanation:
      * The datatype of 'data' is 'pandas.core.frame.DataFrame' which is a csv file readed pandas. Furthermore, it provide a 'vector operation: data = data[data.occupation == target]' which allow you to keep the target data without loop it manually. In this place, the target is the data not equal to '?'. 

* Dealing with discrete features:
   
   * Strategy: 
      1. divide the discrete features into four parts (@unlimitediw):
         1. correlated feature: the feature which has some correlation in it's values. For instance, to the feature 'education', it is obviously that Preschool < Doctorate and so on.              
            * treatment: to the well-distributed integer clssificated feature, we can diretly used it as a input feature with normalization while to the high correlated data such as the education, we can convert it to well-distributed integer classified feature 'education num' in a proper weighted way.
            * inside classification:
               * age
               * education
               * education num
               * hours.per.week
         2. independent feature: the feature which has no relationship in it's values such as the native country
            * treatment: Although we can not find the correlation inside of it directly, we can first do the clustering depends on the class relation with label. Forexample: to the occupation, we calculate the percentage of '>50k' of each class of occupation and rank it. After that, create a new boolean feature that takes the left half as strong occupation and the right half as the weak occupation.
            * inside classification:
               * native country
               * race
               * occupation
         3. boolean feature: these kind of data is the most basic one, we don't need to do any treatment with it.
            * treatment: It is the most basic feature and we can just set the true as 1 and false as -1 and take it as input feature.
            * inside classification:
               * sex
         4. compound feature: some feature is very hard to find the relation inside of it. May be some parts of it are correlated but other are not and the classification of this feature is not well-distributed.
            * treatment: Someway same as the independent features. However, there may some different since it still has some correlation inside some of it's feature classes and we may deploy the data unequally with more heuristic instruction. For instance, we can rank the feature classes based on percentage of '>=50k' and average the feature classes with transistion stats such as married divorce because it is close.
            * inside classification:
               * marital status
               * workclass


<a name="svm"></a>
## Linear Soft Margin SVM



<a name="performance"></a>
## Performance
