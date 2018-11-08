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
divide the discrete features into four parts (@unlimitediw):
  1. Correlated Feature: the feature which has some correlation in it's values. For instance, to the feature 'education', it is obviously that Preschool < Doctorate and so on.              
      * treatment: to the well-distributed integer clssificated feature, we can diretly used it as a input feature with normalization while to the high correlated data such as the education, we can convert it to well-distributed integer classified feature 'education num' in a proper weighted way.
      * inside classification:
         * age
         * education
         * education num
         * hours.per.week
  2. Independent Feature: the feature which has no relationship in it's values such as the native country
      * treatment: Although we can not find the correlation inside of it directly, we can first do the clustering depends on the class relation with label. Forexample: to the occupation, we calculate the percentage of '>50k' of each class of occupation and rank it. After that, create a new boolean feature that takes the left half as strong occupation and the right half as the weak occupation.
      * inside classification:
         * native country
         * race
         * occupation
  3. Boolean Feature: these kind of data is the most basic one, we don't need to do any treatment with it.
      * treatment: It is the most basic feature and we can just set the true as 1 and false as -1 and take it as input feature.
      * inside classification:
         * sex
  4. Compound Feature: some feature is very hard to find the relation inside of it. May be some parts of it are correlated but other are not and the classification of this feature is not well-distributed.
      * treatment: Someway same as the independent features. However, there may some different since it still has some correlation inside some of it's feature classes and we may deploy the data unequally with more heuristic instruction. For instance, we can rank the feature classes based on percentage of '>=50k' and average the feature classes with transistion statUs such as 'Married-spouse-absent' and 'Married-civ-spouse' because it is too close.
      * inside classification:
         * marital status
         * workclass

* Spilt the dataset for stratified 10-fold-cross validation.
    * coding:
    
            class KFold(object):
                def __init__(self, X, Y, foldTotal=10):
                    self.X = X
                    self.Y = Y
                    self.foldTotal = foldTotal
                    self.spiltLength = len(self.Y) // foldTotal

                def spilt(self, foldTime):
                    '''
                    It will be a little not well distributed because there is a remain for len(self.Y) // foldTotal.
                    But the remain will smaller than foldTotal and does not matter comparing with the large training set.
                    :param foldTime: the counter of spilt operation
                    :return: training data of input and label, validating
                    '''

                    validateStart = foldTime * self.spiltLength
                    validateEnd = (foldTime + 1) * self.spiltLength
                    trainX = self.X[0:validateStart] + self.X[validateEnd:]
                    trainY = self.Y[0:validateStart] + self.Y[validateEnd:]
                    validateX = self.X[validateStart:validateEnd]
                    validateY = self.Y[validateStart:validateEnd]
                    return trainX, trainY, validateX, validateY

    * Explanation:
    
        It is basically the same as 'sklearn.model_selection.KFold' but I put the data and label address inside of the class rather than just return the index outside which I believe that it will be more clear to usage. Furthermore, I don't let the KFold class to do preprocessing of spilt because it is memory cost for large size of data.

* Feature Analysis and Selection:
    * Feature unify: As previous section explained, I converted all discrete feature to continuous feature or basic boolean feature with 1 and -1 as input. Furhtermore, I will do the information gain calculation to select data first rather than normalization.
    * Information Gain ranking: 
        * coding:
            * Information Gain Calculator:
          
                    class EntropyGainHelper(object):
                        def __init__(self,Y):
                            self.Entropy = self.calEntropy(Y)

                        def calEntropy(self, Y):
                            m = len(Y)
                            typeDic = {}
                            for elem in Y:
                                if elem not in typeDic:
                                    typeDic[elem] = 1
                                else:
                                    typeDic[elem] += 1
                            res = 0
                            for key in typeDic.keys():
                                res -= typeDic[key] / m * math.log2(typeDic[key] / m)
                            return res

                        # attention: input X should be transformed to X.T previously
                        # then C = X[i]
                        def calEG(self, C, Y):
                            charTypeDic = {}
                            m = len(Y)
                            res = self.Entropy
                            for i in range(m):
                                if C[i] not in charTypeDic:
                                    charTypeDic[C[i]] = [Y[i]]
                                else:
                                    charTypeDic[C[i]].append(Y[i])
                            for key in charTypeDic.keys():
                                res -= len(charTypeDic[key])/m * self.calEntropy(charTypeDic[key])
                            return res
             * Feature Information Gain Visulization and Selection:
                    
                    data = df.values
                    # character selection
                    Ecal = EntropyGainGenerator.EntropyGainHelper(Y)
                    charaNameList = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'marital status', 'occupation',
                                     'relationship', 'race', 'sex', 'capital gain', 'capital loss', 'hours.per.week', 'native country']
                    charaEGDic = {}
                    for i in range(len(X)):
                        charaEGDic[charaNameList[i]] = Ecal.calEG(X[i], Y)
                    sort_key = sorted(charaEGDic, key=lambda x: charaEGDic[x])[::-1]
                    rankingEG = []
                    for key in sort_key:
                        rankingEG.append((key, charaEGDic[key]))
                    for val in rankingEG:
                        # print(val)
                        pass
         * Explanation: 
            * For the first part: It return the information gain after slecting one feature with functions:
                ![](https://github.com/unlimitediw/SVM/blob/master/Image/EntropyCal.PNG)
                ![](https://github.com/unlimitediw/SVM/blob/master/Image/InformationGain.PNG)

                
<a name="svm"></a>
## Linear Soft Margin SVM



<a name="performance"></a>
## Performance
