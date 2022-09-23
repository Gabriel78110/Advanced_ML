import numpy as np

class SoftSVM(object):
    def __init__(self, C):

        """
        Soft Support Vector Machine Classifier
        The soft SVM algorithm classifies data points of dimension `d`
        (this dimension includes the bias) into {-1, +1} classes.
        It receives a regularization parameter `C` that
        controls the margin penalization.
        """
        self.C = C

    def predict(self, X: np.ndarray) ->  np.ndarray:
        """
        Input
        ----------
        X: numpy array of shape (n, d)

        Return
        ------
        y_hat: numpy array of shape (n, )
        """

        # TODO: Make predictions
        y_hat = np.sign(np.dot(X,self.w))

        assert y_hat.shape==(len(X),),\
            f'Check your y_hat dimensions they should be {(len(X),)} and are {y_hat.shape}'
        return y_hat

    def subgradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        subgrad: numpy array of shape (d, )
        """

        # TODO: Compute the subgradient

        subgrad = np.zeros(X.shape[1])
        for i in range(X.shape[0]): #check subgradient criterion for each training point
            if y[i]*np.dot(X[i,:],self.w) < 1:
                subgrad -= self.C*y[i]*X[i,:]
        subgrad += self.w  #add the gradient of the main term

        assert subgrad.shape==(X.shape[1],),\
            f'Check your subgrad dimensions they should be {(X.shape[1],)} and are {subgrad.shape}'
        return subgrad

    def loss(self, X: np.ndarray, y: np.ndarray):
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        svm_loss: float
        """

        hinge_term = 0
        reg_term = (1/2)*np.dot(self.w,self.w)
        for i in range(X.shape[0]):

            hinge_term += max(0,1-y[i]*np.dot(self.w,X[i,:]))


        svm_loss = reg_term + self.C * hinge_term
        # TODO: write the soft svm loss that incorporates regularization and hinge loss
        return svm_loss

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        """
        Input
        ----------
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )

        Return
        ------
        accuracy: float
        """
        # TODO: Evaluate the accuracy of the model on the dataset
        y_hat = self.predict(X)
        accuracy = np.sum(np.where(y_hat==y,1,0))
        return accuracy/y.shape[0]

    def train(self,
              X_train: np.ndarray, y_train: np.ndarray,X_test,y_test,
              n_iterations: int, learning_rate: float,
              random_seed=1) -> None:
        """
        Input
        ----------
        X_train: numpy array of shape (n, d)
        y_train: numpy array of shape (n, )
        n_iterations: int
        learning_rate: float
        random_seed: int
        """

        # Check inputs
        assert len(X_train)==len(y_train)
        assert np.array_equal(np.sort(np.unique(y_train)), np.array([-1, 1]))

        # Initialize model
        np.random.seed(random_seed)
        self.d = X_train.shape[1]
        self.w = np.random.normal(size=(self.d,))

        for t in range(n_iterations):
            # TODO: Update weights according to training procedure

            grad = self.subgradient(X_train, y_train) #compute the gradient
            self.w -= learning_rate * grad  # step in the opposite direction of the gradient

            if t==0 or t==(n_iterations - 1):
                print(f'iteration {t}','training loss = ', self.loss(X_train,y_train), 'accuracy = ', self.accuracy(X_train,y_train))
                print(f'iteration {t}','test loss = ', self.loss(X_test,y_test), 'accuracy = ', self.accuracy(X_test,y_test))
                print('-----------------------------------------------------------------------------')
