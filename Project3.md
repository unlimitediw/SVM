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
                    trainX = np.concatenate((self.X[:validateStart], self.X[validateEnd:]))
                    trainY = np.concatenate((self.Y[:validateStart], self.Y[validateEnd:]))
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
             * Continuous feature preprocessing:
                    
                    # feature 2, 10, 11 are fnlwgt, capital gain and capital loss respectively
                    X[2] //= 400
                    X[10] //= 2000
                    X[11] //= 2000
                
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
                * E = -sum(p * log(p))
                    * ![](https://github.com/unlimitediw/SVM/blob/master/Image/EntropyCal.PNG)
                * Gain = E - sum(featureP * featureE)
                    * ![](https://github.com/unlimitediw/SVM/blob/master/Image/InformationGain.PNG)
            * For the second part: Apart from discrete data, continuous data also need preprocessing to get the information gain. Instead of use the knowledge of integral, I simply convert it to discrete data for the convenient of information gain calculation.
            * For the third part: I use the class in first part to calculate the information gain for each feature, rank it and visualize it in next section.
        * Visulization:
            * Information Gain Ranking （After converting continuous data to discrete data):
                * ![](https://github.com/unlimitediw/SVM/blob/master/Image/IGRank.png)
                
            * Highest Information gain feature:
                * ![](https://github.com/unlimitediw/SVM/blob/master/Image/dataplotRelationMarried.png)
            * Education and Age data plot:
                * ![](https://github.com/unlimitediw/SVM/blob/master/Image/eduandAge.png)

        * Analysis:
            * If we simply applied information gain calculator to these features, the continuous datatype features of 'fnlwgt' and the discrete datatype 'relationship' will be the two feature with highest information gain which are 0.5806 and 0.1662. 
            * The highest discrete two information gain features are "relationship" and "married status". Nevertheless, there is no too much correlation inside the classes of it and we need to do PCA and specify scalar with these classification features. Thus, I use the more conrete features with high information gain: "age" and "education num"
            * However, we should not use continuous datatype to our information gain model because in feature selection, the entropy is not calculated on the actual attributes, but on the class label. If you wanted to find the entropy of a continuous variable, you could use Differential entropy metrics such as KL divergence, but that's not the point about feature selection. [reference](https://datascience.stackexchange.com/questions/24876/how-can-i-compute-information-gain-for-continuous-valued-attributes)
            * Secondly, I do not selection the discrete feature 'relationship' because as I metion previously, there is not too much correlation inside of the classes of the feature and is not appropriate to SVM model.
            * Finally, the two features age and educationNum with entropy gain of 0.0975 and 0.0934 repspectively are the highest information gain features after removing the continuous features of 'fnlwgt', 'captial gain' and no correlated feature 'relationship' and 'marital status'
                
            
<a name="svm"></a>
## Linear Soft Margin SVM and Kernel SVM
* Before showing my validation result and decision boundary I want to display my handwork coding for my svm training model, linear svm trainer, kernel svm trainer with some kernel functions and smo algorithm inside.
    * Train SVM Model code including linear soft svm, kernel svm with 'rbf' and 'polynomial:

            class SVM_HAND(object):
                def __init__(self, C, XSample, YSample, tolerance = .1, sigma = 3, kernel = 'rbf'):
                    self.XSample = XSample
                    self.YSample = YSample
                    self.C = C
                    self.alpha = np.zeros(YSample.shape)
                    self.b = 0
                    self.sigma = sigma
                    self.kernel = kernel
                    self.m = len(YSample)
                    self.SMO(XSample, YSample, C,tolerance = tolerance)

                def Kernel(self, xi, xj):
                    '''

                    :param xi: np.ndarray
                    :param xj: np.ndarray
                    :param sigma: the lower the sigma, the sharper the model
                    :param kernel: type of kernel
                    :return: gaussian kernel of <xi,xj>
                    '''
                    if self.kernel == 'linear':
                        return xi.dot(xj)
                    if self.kernel == 'rbf':
                        l2_square = np.sum(np.square(xi - xj), axis=-1)
                        k = -np.float64(l2_square/self.sigma ** 2)
                        return np.exp(k)
                    if self.kernel == 'polynomial':
                        return (1 + xi.dot(xj)) ** 2

                def predict(self,x):
                    kernel = self.Kernel(self.XSample, x)
                    result = np.sum(self.alpha * self.YSample * kernel) + self.b
                    return 1 if result >= 0 else -1

                def Hypo(self, x):
                    '''
                    :param alpha: the alpha i weight for sample point
                    :param yi: yi for sample point
                    :param b: threshold for solution
                    :param xi: xi for sample point
                    :param xj: xj for input data
                    :return: yj for predict result
                    '''

                    kernel = self.Kernel(self.XSample, x)
                    result = np.sum(self.alpha * self.YSample * kernel) + self.b
                    return result

                def LHBound(self, yi, yj, alphai, alphaj, C):
                    '''

                    :param yi: label for sample data
                    :param yj: label for input data
                    :param alphai: training alphai
                    :param alphaj: training alphaj
                    :param C:
                    :return:
                    '''
                    if yi != yj:
                        L = max(0, alphaj - alphai)
                        H = min(C, C + alphaj - alphai)
                    else:
                        L = max(0, alphai + alphaj - C)
                        H = min(C, alphai + alphaj)
                    return L, H

                def Eta(self, xi, xj):
                    return 2 * self.Kernel(xi, xj) - self.Kernel(xi, xi) - self.Kernel(xj, xj)

                def AlphaJUpdate(self, alphaJOld, yj, Ei, Ej, eta, H, L):
                    alphaJNew = alphaJOld - yj * (Ei - Ej) / eta
                    if alphaJNew > H:
                        return H
                    elif alphaJNew < L:
                        return L
                    else:
                        return alphaJNew

                def AlphaIUpdate(self, alphaIOld, alphaJOld, alphaJNew, yi, yj):
                    return alphaIOld + yi * yj * (alphaJOld - alphaJNew)

                def BUpdate(self, bOld, Ei, Ej, xi, xj, yi, yj, alphaINew, alphaJNew, alphaIOld, alphaJOld):
                    b1 = bOld - Ei - yi * (alphaINew - alphaIOld) * self.Kernel(xi, xi) - yj * (
                            alphaJNew - alphaJOld) * self.Kernel(xi, xj)
                    if 0 < alphaINew < self.C:
                        return b1
                    b2 = bOld - Ej - yi * (alphaINew - alphaIOld) * self.Kernel(xi, xj) - yj * (
                            alphaJNew - alphaJOld) * self.Kernel(xj, xj)
                    if 0 < alphaJNew < self.C:
                        return b2
                    else:
                        return (b1 + b2) / 2

                def SMO(self, XSample, YSample, C, tolerance=.1, maxPasses=5):
                    '''
                    :param C:
                    :param tolerance:
                    :param maxPasses:
                    :param XSample:
                    :param YSample:
                    :param X:
                    :param sigma:
                    :param kernelT:
                    :return: alpha
                    '''
                    passes = 0
                    self.m = len(YSample)
                    while passes < maxPasses:
                        num_changed_alphas = 0
                        for i in range(self.m):
                            # Calculate Ei using f(xi) - y(i)
                            hypoI = self.Hypo(self.XSample[i])
                            Ei = hypoI - YSample[i]
                            if (YSample[i] * Ei < -tolerance and self.alpha[i] < C) or (
                                    YSample[i] * Ei > tolerance and self.alpha[i] > 0):
                                # Randomly select a j != i
                                j = i
                                while i == j:
                                    j = np.random.randint(1, self.m)
                                # Calculate Ej using f(xj) - y(j)
                                hypoJ = self.Hypo(self.XSample[j])
                                Ej = hypoJ - YSample[j]
                                # Memo old alpha
                                alphaIOld = self.alpha[i]
                                alphaJOld = self.alpha[j]
                                # Compute L and H
                                L, H = self.LHBound(YSample[i], YSample[j], alphaIOld, alphaJOld, C)
                                if L == H:
                                    continue
                                # Compute eta
                                eta = self.Eta(XSample[i], XSample[j])
                                if eta >= 0:
                                    continue
                                # Compute and clip new value for alphaj using
                                self.alpha[j] = self.AlphaJUpdate(alphaJOld,YSample[j],Ei,Ej,eta,H,L)
                                if self.alpha[j] > H:
                                    self.alpha[j] = H
                                elif self.alpha[j] < L:
                                    self.alpha[j] = L
                                if abs(self.alpha[j] - alphaJOld) < 10 ^ -5:
                                    continue
                                # Determine value for alphai
                                self.alpha[i] = self.AlphaIUpdate(alphaIOld,alphaJOld,self.alpha[j],YSample[i],YSample[j])
                                # Compute b
                                self.b = self.BUpdate(self.b, Ei, Ej, XSample[i], XSample[j], YSample[i], YSample[j], self.alpha[i],
                                                 self.alpha[j],
                                                 alphaIOld, alphaJOld)
                                num_changed_alphas += 1
                        print(num_changed_alphas,passes)
                        if num_changed_alphas == 0:
                            passes += 1
                        else:
                            passes = 0

    * validate the svm model with 10-fold-cross-validation 
        * part of model and prediction accuracy generation:
        
                dataset = KF.KFold(X, Y)
                totalAccuracy = 0
                for i in range(10):
                    trainX, trainY, validateX, validateY = dataset.spilt(i)
                    C = 0.3
                    SVM_SAMPLE = SH.SVM_HAND(C, trainX, trainY, tolerance=0.1, kernel='rbf')
                    localAccuracy = validate(validateX, validateY, SVM_SAMPLE)
                    print("local Accuracy for", i, "time:", format(localAccuracy, '.3f'))
                    totalAccuracy += localAccuracy
                totalAccuracy /= 10
                print(format(totalAccuracy, '.3f'))

                plotData()
                plotBoundary(SVM_SAMPLE, 0, 100, 0, 20)
                
        * part of 10-fold-cross-validation:
        
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
                        trainX = np.concatenate((self.X[:validateStart], self.X[validateEnd:]))
                        trainY = np.concatenate((self.Y[:validateStart], self.Y[validateEnd:]))
                        validateX = self.X[validateStart:validateEnd]
                        validateY = self.Y[validateStart:validateEnd]
                        return trainX, trainY, validateX, validateY

* Result and Analysis:
    * Accuracy:
        * ![](https://github.com/unlimitediw/SVM/blob/master/Image/LinearValidateAcc.png)
    * Support Vectors:
        * ![](https://github.com/unlimitediw/SVM/blob/master/Image/LinearSupportVectors.png)
    * Decision Boundary:
        * ![](https://github.com/unlimitediw/SVM/blob/master/Image/LinearDecisionBoundary.png)

    * In this problem I select age and education num to make the visualization more intuitionistic and use 10-fold validation and data ramdomlize to relieve the overfitting problem.
    
* Change C and evaluate model:
    * Relationship between C and SVM performance:
        * SVM perfomance relationship with C value in linear svm model is really not obvious. Generally speaking, though, Larger C means the SVM model will be more strict and has less error point. However, the C value will only affect the performance in this model slightly.
        * ![](https://github.com/unlimitediw/SVM/blob/master/Image/Crelationship.png)
        
    * decision boudary with smaller C 0.01:
        * ![](https://github.com/unlimitediw/SVM/blob/master/Image/LinearCp01.png)
    * decision boudary with medium C 1:
        * ![](https://github.com/unlimitediw/SVM/blob/master/Image/LinearC1.png)
    * decision boundary with larger C 3:
        * ![](https://github.com/unlimitediw/SVM/blob/master/Image/LinearCp03.png)
    

* Train SVM using all features:
    * part 1: More about feature preprocessing: For the calssification type feature, I want to use one hard coding to define the discrete features. However, in this project, more features is still a disaster so I can only a little bunch of data and some important features to train and test. Furthermore, for some features that have similarity I will aggregate it and simply flatten the classification data such as marital status into a series of boolean features such as Speparted[0,1], Never-married[0,1].
        * e.g
            
              Xnames = ['age','education num','hours','capital gain','capital loss','fnlwgt','sex','married']
              X = [33,5,40,0,0,132870,1,0]
    * part 2: I will also do the feature normalization with Standardization since the range of values of raw data varies widely such as capital gain. The range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.




<a name="performance"></a>
## Performance
