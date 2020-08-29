Q1. Describe what is central limit theorem?

    The central limit theorem states that if we draw a sample from a population
    with large enough sample size, the mean of the sample will be normally 
    distributed. As the size of the sample grows, the mean tends toward the 
    population mean and the variance of the mean will tends towards the 
    population variance divide by the sample size. This is true regardless of
    the population distribution, in other words, the population distrubtion 
    does not need to be normal distributed.


Q2. What is stratified sampling?

    In stratified sampling, the dataset is divided into subgroups based on a 
    common factor. Samples are then drawn proportionally from each group. An 
    example where stratified sampling is used is when trying to split a 
    dataset into train part and test part, ideally we want both sets to have
    the same distrubtions for the target to be predicted. In this case, the 
    target variable will be used to divide the data into groups.


Q3. What is type I and type II error?

    type I error is FP. type I error occurs when the null hypothesis is true but rejected.
    type II error is FN. type II error occurs when the null hypothesis is false but erronerously rejected.


Q4. What is the assumption of linear regression?

    1) There is a linear dependence between target and features.
    1) The residue (y_hat - y) is independent from each other and normally distributed.
    2) There is minimum multicollinearity between features.
    3) The variance around the regression line is the same for all features.


Q5. What is selection bias?

    Selection/sampling bias occurs when the data sample gathered or prepared for modeling doesn't 
    represent the true distribution that the model will see in future.


Q6. What is an example of non-Gaussian distribution?

    Binomial (p), multiple tosses of a coin with probablity p of being head and 1 - p being tail.
    Bernoulli (p) single coin tosses
    Poisson (lambda) a rare event will occur in n steps.

Q7. What is the bias-variance trade off?

    Bias and variance are two sources of errors typically occurs in a machine learning model. Bias is an
    error introduced due to over simplification of the model, which can lead to under-fitting.
    Variance is introduced due to too high complexity in the model. It usually performs very well in
    training data but fails to generalize to test data.

    A third type of error, which is called the Baysian Error, is the error of a perfect model that follows
    the true underline distribution / data generation process. This happens when there are noise in the ground
    truth or there are hidden factors which is unknown to the model.

    The Bias and Variance of a model depends on the model's capacity. If we make a plot of Bias and Variance
    vs model capacity, then as capacity increases the bias decreases while variance increases. To achieve optimal
    genralization error on unseen dataset, we have to balance the two type of errors by choosing an model that
    has an appropriate model capacity.

    1) KNN has low bias but high variance. Increase k to reduce variance, which allows more neighbors to contribute to the prediction.
    2) SVM has low bias but high variance. Add weight penality or kernel parameters to reduce variance. For example, 
       for gaussian kernel, we can increase the sigma parameter to reduce variance.
    3) Decision tree has low bias and high variance. Limit the depth of the tree or # of features to be used or stop splitting
       when a node size is below limit size.
    4) Linear regression is high bias and low variance. To reduce bias, we can add more features to the model or feature 
       interactions such polynomial terms or product of multiple features to the model.


Q8. What is confusion matrix?

    Confusion matrix is a metric for measuring the performance of a classification model. In the binary case, it has four elements including
    TP = positive example predicted positive
    FP = negative example predicted positive
    TN = negative example predicted negative
    FN = positive example predicted negative

    recall (sensitivity/true positive rate) = TP / P, percentages of positive examples predicted positive
    precision = TP / TP + FP, of the predicted positives, the percentages that are truely positive.
    specificity (true negative rate) = TN / N
    f1 = 2 * precision * recall / (precision + recall)


Q9. What is the goal of A/B testing?

    It is a hypothesis testing for a randomized experiment with two versions (A and B) for the random variable to be tested.
    The goal of A/B test is to identify the infuence of any changes to a product. An example of this could be
    to identify influence of adding a like button to the user engagement in a webpage.


Q10. What is p-value?

     When performing a hypothesis test in statistic, a p-value can help you determine the strength of the result.
     p-value is the minimum siginificance level that the null hypothesis can be rejected. The lower the p-value, 
     the more likely you can reject the null hypothesis.


Q11. What are the difference between underfitting and overfitting?

     Underfitting is a phenomeno in which the statistical model or machine learning algorithm fails to capture the
     underlying trend/pattern of the data. It mostly occurs when the model is overly simplified, for example when 
     fitting a linear model to non-linear data.

     Overfitting is a phenomeno in which the statistical model or machine learning algorithm follows the noise instead
     of the underlying relationships of the data. It mostly occurs when the model overly complex, such as the too 
     many features with not enough data sample. 


Q12. How to combat underfitting and overfitting?

    To combat underfitting:
    1) try add more features to the model
    2) try other models with larger capacity
    3) increase training iterations

    To combat overfitting:
    1) add more data
    2) try simpler models (reduce model capacity)
    3) regularization
        3.1) l2 or l1 or elasticnet regularization, l2 weight decay, l1 sparsity 
        3.2) early stopping (deep learning)
        3.3) dropout (deep learning)
    4) use k-fold cross validation for model parameter tuning/selection
    5) ensemble (bagging or boosting)
    6) add noise to data
    7) feature selection

Q13. What is the Law of Large Numbers?

    It is a theorem that describes the result of performing the same experiment a large number of times. According to the law,
    the average of the results obtained from a large number of trials should be close to the expected value and will tend to
    become closer to the expected value as more trials are performed.


Q14. What is TF/IDF vectorization?

    TF/IDF is short for term frequency inverted document frequency. It is a numeric statistic that is intended 
    to reflect how important a word is to a document in a collection of document or corpus. It is often used as
    a weighted factor for information retrievel and text mining.

    TF = # of times a word occured in a document / total # of words in the document
    IDF = log(# documents contains the word / total # of documents in corpus)
    
    The TF indicates the relative importance of the word in a given document while IDF help to adjust for the fact 
    that some words are more frequent in general.


Q15. Why we generally use softmax (or sigmoid) non-linearity function as final output in NN? Why RELU in inner layers?
     
    That's because in the final layer, we typically want to output a probability distribution. The softmax
    or sigmoid were designed to scale any real number in the region between [0, 1], which satisfies the 
    probability requirement.

    For inner layers, the issue with softmax or sigmoid is that they quickly saturates when moved away from 
    zero which means the gradient of them will be very small except near zero, in a NN of many layers this 
    leads to a phenomena called vanishing gradient, which will cause the learning algorithm to progress very
    slowly and eventually stops. 

    The Relu activation can solve the vanishing gradient problem because in most regions it has a constant
    gradient of 1, g(x) = max(0, x).


Q16. Train vs validation vs test sets?

    Training
        1) to fit the parameters of a model, i.e. weights in linear regression
        
        Validation
        1) part of the training set
        2) used for model hyperparameter tuning, i.e. regularization strength in linear regression
        3) to avoid overfitting (by plot train & valid error vs different model hyperparameters)

    test
        1) never used for training
        2) to estimate model generalization error


Q17. Cross validation

    Cross validation is a sampling technique used to estimate the generalization power of a model on
    unseen data. A typical scheme of doing cross validation is called k-fold cross validation, in which 
    the data is randomly divided into k subsets called k folds. Then, one of folds is hold out and the
    model is trained on the rest k - 1 folds. This process is repeated k times by rotating the hold out 
    fold. 

    The cross validation provides several benefits:
    1) In case of limited training data, we don't want to have to a hold out test set because it will 
        further reduce the training data.
    2) It allows us to estimate the generalization error on data not used for training.
    3) It can be used for model selection / hyperparameter tuning.
    4) It can help reduce overfitting issues by comparing the training error and validation error in
        different model settings and choose the best one that gives minimum gap in train and validation
        error.
    Other variants:
        1) stratified k fold (data is sampled proportionally according to a group factor).
        2) leave one out, on extreme low data volume setting.


Q18. What is machine learning?

    Machine learning is the study of computer algorithms that improve automatically through
    experience. It is seen as a subset of artificial intelligence. Machine learning explores
    the study and construction of algorithms that can learn from and make predictions on data.


Q19. What is supervised learning?

    In supervised learning, each data point is given a label (target). It is called supervised
    in the sense that the label is served as a supervisor for the algorithm to learn.
    Linear regression, logistic regression, decision tree, naive bayes, knn, svm, nn, etc.

Q20. What is unsupervised learning?

    Unsupervised learning is a type of machine learning algorith used to draw inference from 
    input data without any labels.

    clustering: kmeans, gausian mixture models; anaomly detection; representation learning


Q21. What is "Naive" in Naive Bayes?

    Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes'
    theorem with the "naive" assumption of conditional independence between any pair of features
    given the class variable, that is P(x_i|y, x_1, x_2, ..., x_n) = P(x_i|y)

    Bayes' theorem:
    P(y|x_1,x_2,...x_n) = P(y)*P(x_1,x_2,...,x_n|y) / P(x_1,x_2,...,x_n)
                        = P(y)*P(x_1|y)*P(x_2|y)*...*P(x_n|y) / P(x_1,x_2,...,x_n)

    We can use Maximum a Posteriori (MAP) estimation to estimate P(y) and P(y|x_i) from the observed
    data. 
        P(y|x_1,x_2,...,x_n) ~ P(y)*P(x_1|y)*...*P(x_n|y) =>
        y_hat = argmax_{y} P(y)*P(x_1|y)*...*p(x_n|y) = argmin_{y} -log(P(y)) - sum(logP(x_i|y))


Q22. What is PCA?

    Principal component analysis is a statistical method used in machine learning. It is used for
    projecting higher dimensional data into a lower dimensional space by maximize the variance in
    the projected dimensions.

    There are too ways to do PCA: eigen decomposition or singular value decomposition

    1) eigen decomposition: A = X^TX = UMU^T => B = AU
    2) SVD: A = UDV => B = AV


Q23. SVM

    Assume we have a training set with p features. We can view the data as points in the 
    p dimensional space, so we want to find a p - 1 dimensional hyperplane that seperate 
    data points of one class from the other. There may exists many choice for this hyperplane.
    One reasonable choice as the best hyperplane is the one that has largest separation or
    margin between the two classes. This hyperplane is called maximum margin hyperplane and 
    the corresponding classifier is maximimum margin classifer (SVM).
    
    1) the hyperplane is defined as w*x - b = 0. 
    2) decison boundary is the sign(w*x-b)
    3) margin: 2 // ||w||
    4) support vectors are the points lie on |w*x-b| = 1
    5) hinge loss max(0, 1 - y_i*(wx-b))
        the hinge loss is zero if the data point lies in the correct side of the margin,
        otherwise it increasely linearly with the distance to the margin
    6) kernels
        Linear kernel
        Gausian kernel (RBF), increase gamma increases variance
        Polynomial kernel
        Sigmoid kernel

    Application: text categorization, image classificatin, hand written digit recoginition

Q24. Ensemble algorithms

    Boosting: sequentially improve a model from its previous versions
    1) Adaboost: improve from previous weak learner by assign more weight to incorrect predicted samples
        pros: 
            1.1) robust against in low noise data 
            1.2) few hyperparameter, easy to tune
        cons: slow
    2) Xgboost: gradient tree boosting
        pros: 
            2.1) good performance if tuned correctly
            2.2) very fast due to parallelism
            2.3) regualarization
        cons:
            2.4) multiple hyper parameters, hard to tune
            2.5) may overfit
    Bagging: is non sequential, a collections of weak models are trained using different boostrap samples with replacement from the data
    3) Random Forest: a collection of independent decision trees
        pros:
            3.1) robust against noise, outliers, missing data
            3.2) deal with mixing data types, categorical needs not to be one-hot-encode
            3.3) can do feature selection via feature importance
            3.4) explainability
            3.5) small variance by using large number of trees
        cons:
            3.6) many parameters to tune
            3.7) should be avoided in time series data or look ahead bias should be avoided
                    or the order of data continuity should be ensured.


Q25. How to split data in decison tree?

    metrics:
    gini index: gini = 1 - sum(p_i^2), p_i the proportion of ith class, split on largest decrease in gini index
    cross entropy: -sum(p_ilog(p_i)), split on largest decrease of cross entropy


Q26. What is pruning in Decision Tree?

    Pruning is the process to reduce the size of the decision tree with the purpose of improving model
    generalization power or reduce overfitting. E.g., tree depth, minimum size of leaf nodes.


Q27. Outiliers?

    1) replace with 1% or 99% or quantiles
    2) replace it with mean or median 
    3) normalize the data
    4) log transformation (many outliers)
    5) drop the data point


Q28. Missing value?

    1) analyze the reason of missing, data generation process, any patterns; sometimes missing values provides
       provides valuable information. e.g. in network operations, we want to understand network system behavoriors
       from device data; sometimes the missing data is due to a network failure, these missing value actually
       became a important feature.
    2) drop the feature if too much missing (80%)
    3) replace with mean/median (numeric) or default (categorical)
    4) imputate using a learning algorithm


Q29. How are weights initialized in NN?

    1) random: weights are drawn randomly from a predefine probability distribution, e.g. normal or uniform
    2) Xavier initialization: u(6/sqrt(m), 6/sqrt(n))


Q30. What are hyperparameters in DL?

    1) # of hidden layers
    2) # of hidden units in each layer
    3) learning rate
    4) epochs: # of times the training data should be iterated through
    5) batch size


Q31. What is cost function?

    Cost function is also called loss function or error function. Cost function is a measure
    of how well perfoms when predicting on a given dataset. Most common loss is mean sum of
    squared errors (MSE).
    We can define a loss function using MLE, argmin -logP_model(y|x, y_hat)


Q32. Epoch, Batch, iteration?

    1) Epoch: # of times the model will run through the entire dataset
    2) Batch: data is too large to feed to network at once, divide into smaller batches
    3) Iteration: data_size / batch_size


Q33. Layers in CNN?

    1) Convolutional Layer - performs a convolution operation, creating smaller
       Windows of original image at different location.
       convolution act as an kernel operation, which slides over the input image and
       multiply its elements with the pixels and sum over to a single value at each 
       window of the input image. Each convolution layer will generate a higher level
       feature map, from dot to curved line to cycle, etc.

    2) Activation Layer (ReLU) - it brings non-linearity to the network and converts
       all negative pixels to zero. It follows each convolutional layer.

    3) Pooling layer - pooling is a down sampling operation that reduces the
       dimensionality of the feature map. Stride: how much to slide.

    4) Fully connected layer: output layer, softmax or sigmoid

    Training: back-propagation algorithm
    1) Forward pass
    2) Loss function
    3) Backward pass
    4) Weight update


Q34. What is pooling in CNN?

    Pooling is a filter process used to reduce the spatial dimensions in CNN. It performs
    down sampling on the feature maps created from its previous layer by sliding a filter
    matrix over the input feature map.


Q35. What is a RNN?

    RNNs are a family of neural networks for processing sequential data. The recurrent reflects
    the fact that it can be applied repeatedly to the next value in a sequence to produce an output.

    Parameter sharing: in typical NN, there are many different weights between adjacent layers;
    while in RNNs, we are repeatedly applying the same weights (parameters) to different items in
    the input.

    Another example of parameter sharing is the convolutional layer in CNN, where a single convolution
    kernel matrix is applied repeatedly to different locations of the input image.

    Deep RNN:
        1) add hidden states, one on top of Another
        2) add additional nonlinear layers between input and hidden state
        3) increase depth in the hidden to hidden transition
        4) increase depth in the hidden to output transition
    
    Bidirectional RNN
    Sometimes it's not just about learning from past but future may also influence the prediction
    on current input. For example, speech recognition task, the transcription will change depends 
    on the future context.


Q36. Encoder / Decoder or seq-to-seq RNNs?

    A sequence to sequence model is a model that takes a sequence of items (words, chars, features of images) as input and outputs another sequence of items. 
    
    In machine translation, input sequence would be the words in one language and output is the corresponding sequence of words in the target language.

    Under the hood, the model is composed of an encoder and a decoder. The encoder processes each item in the input sequence and compiles the information it captures into a context vector. After processing the entire sequence, the encoder sends the context vector to the decoder, which begins to producing the output sequence item by item.
    

Q37. Explain LSTM network?

    LSTM is short for long short term memory networks. It is a special kind of RNN which is capable
    of learning long term dependencies in seq data.

    The key to LSTMs is the cell state, which regulates information flow across different time steps
    using structures called gates. 

    Gates are a way to optionally let information through. They are composed out of a sigmoid layer
    and a pointwise multiplication operation. The sigmoid layer outputs numbers between 0 and 1, 
    describing how much each component should be let through. A value of zero means "let nothing 
    through" and a value of one means "let everything through".

    A typical LSTM cell, it consists of the following layers:
        1) forget gate layer: controls how much each compoenent from previous cell state should be forgotten.
           f_{t} = sigmoid(W_f(h_(t-1), x_(t)) + b_f)
        2) input gate layer: controls what new information will be added to the cell state. The gate will multiply the new candidate values and which will then be added to the cell state.
            i_{t} = sigmoid(W_i(h_(t-1), x_(t)) + b_i)
            C'_{t} = tanh(W_C'(h_(t-1), x_(t)) + b_C')
        3) update cell state:
            C_{t} = f_{t} * C_{t-1} + i_{t} * C'_{t}
        4) output layer:
            o_{t} = sigmoid(W_o(h_(t-1), x_(t)) + b_o)
            h_{t} = o_{t} * tanh(C_{t})


Q38. What is MLP?

    A MLP has the same struture as single layer perceptron but with multiple hidden layers.

    A single layer perceptron can only classify linearly separable data but MLP can complicate non linear separable data. 

    Relu as hidden layer activation function and softmax/tanh/sigmoid as final output.


Q39. Explain Gradient Descent?

    Firstly, gradient is a mathematical funtion, which gives the slope&direction for which the original function moves fastest. The gradient only vanishes at stationary points of the function.

    Gradient descent is a first order iterative optimization algorithm for
    finding the minimum of a function. For machine learning, it is used to minimize the loss function of the algorithm. Given a set of training data, it performs the following steps iteratively until converge or stoped:
        1) compute the gradient (slope) which is the first order derivative of the loss function with respect to model parameters at current point.
        2) update the parameters in the opposite direction of the slope increase from current parameter values by the computed amount.

Q40. What is exploding gradient?

    Exploding gradient is a problem found in training deep neural networks using gradient based algorithms and back propagation. As the depth of neural network increases, the gradient growths exponentially with matrix multiplication if the value in these matrix is larger than one. A counter problem is the gradient vanishing problem if multiply many numbers less then one, the gradient shrinks to zero exponentially and eventually completely vashises.

    How to detect gradient exploding?
    1) The model does not learn much with the training data, very poor loss.
    2) The loss changes very much at each updates due to model instability.
    3) The loss became too large or even NaN. 

    How to confirm when observed above?
    1) The model weights quickly become very large during training.
    2) The weights become NaN during training.
    3) The error gradient are consistently above one for each layer and node.

    How to fix gradient exploding?
    1) Redesign model architechture
        1.1) use fewer layers
        1.2) use smaller batch size
        1.3) RNN truncated backpropagation, that is updating across fewer prior time steps during training.
    2) For RNNs, use gate units to learm long-term dependencies, e.g. LSTMs
    3) Gradient clipping: clipping the gradients if their norm exceeds a given threshold.
    4) Weight regularization: add a weight penalty term to the loss function.


Q41. What is vanishing gradient?

    Vanishing gradient is a problem when training a deep neural network using gradient based learning algorithms, which will cause the model weights updates very little or no update at all, effectively stop the learning process. It leads to slow training, poor performance and low accuracy.

    How to fix it?
    1) Use ReLu activation instead of hyperbolic tangent and sigmoid/softmax activation. ReLu has a constant gradient of one when the input is positive. Sigmoid or tangent activation has very small slope except near zero, recursively multiplicaiton will lead to exponentially shrink towards zero.
    2) For RNNs, use LSTM instead. LSTM use gate units to control information flow. If take the differentiation delta(C_{t})/delta(C_{t_1}), the term f_{t} will have the effect of preventing the gradient be either too large or too small.
    3) Use ResNet (residual network): after some layers, add x again: F(x) -> ... -> F(x) + x
    4) Gradient checking: debugging strategy used to numerically track and assess graidents during training.


Q42. Explain back propagation.

    Back propagation is a algorithm used for training neural networks. It works by sending the error made in previous iteration from the output layer backwards to update the weights of the model layer by layer.

    Work flow:
        1) Forward propagation from an input batch towards output layer, calculate model loss.
        2) Calculate graident with respect to model weights by applying chain rule from one layer to its previous layer. Update the weights in opposite direction of gradient.
            w = w - a * g
   

Q43. What are the variants of gradient descent?

    1) Stochastic gradient descent: take only one example at each step for calculating the gradient and updating weights. Loss will fluctuate from example to example will descrease with multiple examples. 
    2) Batch gradient descent: all training data are used in a single step for calculating gradient and updating weights. Take mean gradient, less fluctuations.
    3) Mini-batch gradient descent: It's the most popular choice. Instead of only one example, each step will use a mini-batch of samples to train and updating weights.


Q44. What is the role of Activation Function?

    The activation function is used to introduce non-linearity into the neural network helping it to learn more complex functions. Without activation function would only be able to learn linear combinations of input data.


Q45. What is an auto-encoder?

    An autoencoder is a type of neural network used to learn an efficient data coding in an unsupervised manner. The aim of autoencoder is to learn a representation for a set of data, typically for dimensionality reduciton, by training the network to ignore noise signals. Along with the reduction side, a reconstruction side is learnt, where the autoencoder tries to regenerate from the reduced encoding a representation as close as possible to its original input. 


Q46. What is Dropout?

    Dropout is a technique used during the training a deep neural networks which randomly drop out some nodes (input or hidden) to prevent overfitting. (20%)


Q47. What is BatchNormalization?

    Batch normalization is the technique to improve model performance and stability of neural networks by normalizing the input in every layer of the network, so that they have mean output activation of zero and standard deviation of one.


Q48. Feature selection methods?

    1) Filter methods
       1.1) Linear discrimination analysis
       1.2) ANOVA
       1.3) Chi-Square

    2) Wrapper methods
       2.1) Forward selection
       2.2) Backward selection
       2.3) Recursive feature elimination

    3) Model based
       3.1) Lasso
       3.2) Random Forest  


Q49. How should you maintain a deployed model?

    1) Monitor: constant monitoring the performance of model. When there are changes in the system or data, monitoring helps quickly understand the implications of those changes so we can adjust accordingly.
    2) Evaluate: evaluation is needed for new data and compare to offline benchmark to decide if re-training is required.
    3) Compare: the new models are compared to each other to determine which model performs best.
    4) Rebuild: the best performing model is re-built on the new data.


Q50. What is Generative Adversarial Networks?

    Generative Adversarial Networks is an approach to generative modeling using deep learning methods, such as CNNs. In GANs, two neural networks, called Generator and Discriminator contest with each other in a game. Given a set of training data, the generator learns to generate new data points with the same statistics as the training data while the discrimator will learn to discrinate whether a given data point is generated or real data.


Q51. What are the optimization algorithms used in back propagation?

    1) Gradient descent
    2) RMSProp
    3) Adam


Q52. BERT

    BERT is short for Bidirectional Encoder Representation from Transformers

    pretraining (unsupervised): to understand language
        masked language modeling + next sentence prediction
    
    fine tuning: learn to solve a specific task, fast
        Add addtional output layers for specific task, these layers are trained from scratch but the BERT parameters are only fine tuned.

Q53. Transformer

    Transformers are a model architechture for seq to seq modeling that is solely based on attension mechanism and no RNNs are used.
    
    The encoding part is a stack of encoders, each consists of a self-attention sublayer and a feed forward network.

    The decoding part is again a stack of decoders, each consists of a self-attention sublayer, a multi-head encoder-decoder attension layer and a feed forward network.

    What makes the transformer unique is that it removes the recurrent networks entirely and only use a fixed number of steps. In each of theses steps it uses a self-attention layer to learn the dependencies between words, which outputs a attention score between any pair of input words. Theses scores are used as weights to calculate a weighted average of all words and feed into a feed forward layer to compute the next set of representations. 

    It further refined the attention by using multihead attention, which instead of looking at only one set of attention weights, it uses several, each learning different aspect of the language.

    Encoder:
        input sentence ->
        word vectors + positional encoding vectors (cos + sin) -> 
        multi-head self-attention ->
        weighted sum of attention vectors ->
        feed forword network (parallelism) ->
  ------encoded vectors for every word
  |  
  |     Decoder:
  |     output word vectors + positional encoding vectors -> 
  |     masked (only use previous words as context) self-attension ->
  |---->+weighted sum of attention vectors ->
        encoder-decoder attention layer ->
        feed forward layer ->
        linear layer ->
        softmax layer ->
        output probability: next word

Q54. Attention

    Attension is a technique used in seq to seq modeling which allows the decoder to focus on different parts of the input seq with respect to the context of the target sequence. In the classic encoder decoder architechture, the encoder learns a fixed length context vector (the last hidden state) and send it to the decoder. The problem with this approach is that the fixed length vector may not capture the necessary infomration, especially in cases of very long sequences. The attension mechanism solves this issue by jointly learn an alignment vector along with the encoder and decoder RNNs, which allows the decoder to focus on different parts of the input seq at each time step.

    The attension model differs from a classic seq-to-seq model in two ways:
        1) Instead of sending a single vector from encoder to decoder, the encoder sends all hidden states to the decoder.
        2) The attention decoder does an extra step before producing the output at each time step. In order to focus on the relevant parts of the input seq, the decoder will look at all the encoder hidden states and score them based the alignment vector. The softmax score will then be multiplied to the encoder hidden states, and thus give larger weights to relevant parts of the input seq.


Q55. EM?

    EM is short for expection maximization algorithm. It is an iterative approach mostly commonly used for modeling latent variables in maximum likelihood estimation.


Q56. Design Email Auto Compose?

    Requirements
        1) Latency: to serve in real time applications, as the user type it should provide suggestions within 100ms so it will not affect user experience.
        2) Scale: one billion users, the model should have enough capacity to handle subtle context.
        3) Fairness and privacy: language models learned from large text corpus may contain bias, as such the training process should take it into account. The researcher or engineer should pay special attention to privacy issues, as the user emails should not be exposed to them.
    
    Modeling
        Email auto composing can be formulated as a text generation task. Given the prefix text the user had typed, automatically predict the next word or words to complete the email.

        Classic language modeling models
           1) ngram model
           2) neural bag of words
           3) RNN language model
           4) seq2seq modeling (encoder-decoder)
        One approach to this task is to use a language model to predict the next word giving the prefix text a user has typed so far. Language models such as ngram, neural bag of words or RNNs can be used.

        Another approach is to use other information such as email type, previous email bodies from the same email thread as context; then model it using a seq-2-seq 
        architectures. 
            1) LSTM encoder decoder
            2) Add attention to the encoder decoder architechture
            3) Use transformers
            4) Use Bert 


Q57. Explain word2vec?

    Word2vec is framework for learning distributed work vectors/representations. To explain word2vec, maybe we should start with one hot vectors. So, computers doesn't understand words instead it only understand numeric objects like numbers, vectors and matrices. So we need some way to encode words into numbers. One way to do it is to use one hot encoding for words, which is a vector that has dimensions equal to size of the vocabulary. Each word will have a vector with all zeros except only one dimension being one corresponding to the index of that word. The problem of this approach is that all the vectors are orthogonal to each other, making it very hard to express any semantic similarities betweens, e.g. every word has the same distance or similarities if we use one hot vectors. Another issue is that the dimension can easily go up to hundreds of thousands.

    When we say word2vec is a distributed representation of words, the distributed really means that the signal or the semantic meaning of words are distributed across the dimensions instead of localised into a single dimension. The way how word2vec was constructed is based on a simple idea that similar words will have similar surrounding words.

    In the original paper, two type of models for learning the word2vec was designed:
        1) skip-gram: skip-gram model assumes that we can use a center word to predict its surrounding words. Given the center word, skip-gram is concerned with the conditional probability of P(surrouding words|center word). 
        2) continuous bag of words (CBOW): use context word to predict center word.

    Model training
        The skip-gram model requires computing the conditional probability of a context word given the center word. This softmax computation is very expensive because it will involve computing the dot product between all words in the vocabulary with the vector of the center words. Two approximation algorithm was proposed to estimate the probability efficiently, a sampling based and a modified version of the softmax function.
            1) negative sampling: instead of computing the full softmax probability, the negative sampling methods concerns only to distinguish between positive words and negative words using logistic regression. Here, negative words means it does not appears in the context window of the center word. It achieve this by sampling k negative samples from words not in context of the center word. This simplifies the objective function, only requires computing dot products for k words instead of |V| words.
   
            To further improve training time, the model also subsamples the positive samples.

            2) hierachical softmax: hierarchical softmax makes the computation of the softmax probability faster by groups words into a hierarchical binary tree data structure, where each leaf is one word and each internal node stands for relative probabilities of the children nodes. Each word has a unique path from the root to leaf. The probabilities of picking this word is equivalent to the probability of taking this path from the root down through the tree branches. 

            the probability of turn right at any given interval node is sigmoid(v dot u); turn right is 1 - sigmoid(v dot u) = sigmoid(-v dot u)

Q58 Recommender System?

    Recommender system is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. "Ratings" could be "Buy", "Like", "Watch", or "Dislike".
    
    1) Collaborative Filtering
        Collaborative Filtering is the process of filtering for information or patterns using techniques involving collaboration among multiple users, agents, and data sources.
            1.1) memory based CF
                This technique is nearest neighbor based CF such as user based CF or item based CF. Memory based CF has limitations in dealing with sparse and large scale data since it computes the similarity values based on common items.
            1.2) model based CF
                This technique is based on latent factor models such as matrix factorization machine. Model based methods has better capability in dealing with sparsity and scalability. 
            1.3) hybrid
            
        In general, CF only uses the user-item interaction data to make predictions and recommendations. Besides CF, content-based and context based recommender systems are also useful in incorporating the content descriptions of items/users and contextual signals such as timestamps and locations.

        Matrix Factorization is a class of collaborative filtering models. Specifically, the model factorizes the user-item interaction matrix into the product of two lower-rank matrices, capturing the lower-rank structure of the user-item interactions.

        R(rating matrix) -> P (user matrix), Q (item matrix)
        P -> captures user interests
        Q -> captuers item characteristics
        R_hat = PQ^T ~ R

        This method cannot model user/item bias, for example some users tends to give higher ratings or some items tend to get lower ratings due to poorer quality. To solve this issue, user bias and item bias are introduced:
            R_ui = p_u q_j^T + bu + bi
        Then, it can be trained with the following objective function:
            argmin ||R - R_hat|| + lambda * (||P||2 + ||Q||2 + bu^2 + bi^2)

    2) Recommendation tasks
       2.1) Moives recommendation
       2.2) News recommendation
       2.3) Point-of-interest recommendation

       Recommendation tasks could also be differentiated based on the types of feedback and input data.
       2.4) Explicit rating prediction
       2.5) Top-n recommendation (ranking) ranks all items for each user personally based on implicit feedback (youtube)
       2.6) Click through rate prediction, which is also based on implicit feedback, but various categorical features can be utilized.

       Recommending for new users and recommending new items to existing users are called cold-start recommendation.

    3) AutoRec: Rating predictions with AutoEncoder
        
    4) Personalized Ranking
       4.1) Bayesian Personalized Ranking Loss: maximum posterior estimator
       4.2) Hinge Loss

    5) Neural Collaborative Filtering for Personalized Ranking
       1) AUC
    6) When user item data are extremely sparse, we should seek to include user or item features for recommendations, context/content awareness.

    7) Factorization machines
       7.1) models n-way interactions between features
       7.2) it is reminiscent of support vector machine with a polynomial kernel.
       A fast optimization algorithm associated with factorization machines can reduce the polynomial computation to linear complexity, making it extremely efficient for high dimensional sparse inputs.

    8) Deep factorization machines
       1) 2way FM + MLP