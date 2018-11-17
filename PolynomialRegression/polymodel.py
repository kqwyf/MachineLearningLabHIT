import numpy as np

class poly_model:
    plot_N=100
    def __init__(self,size,l=0):
        """
        size    size of the model
        l       the regularization coefficient
        """
        self.M=size
        self.l=l
        self.w=np.zeros(self.M+1)
    def _getX(self,x):
        """
        Returns the Vandermonde matrix of x.
        """
        return np.array([x**i for i in range(self.M+1)]).T
    def _get_batch(self,x,y,batch_size):
        """
        Returns a random batch with size batch_size from given data.
        """
        if batch_size==0:
            return x,y
        else:
            index=list(np.random.randint(0,len(x),batch_size))
            return x[index],y[index]
    def loss(self,x,y):
        """
        loss(x,y)=(1/2)||Xw-y||**2+(l/2)||w||**2
        """
        y=y-self._getX(x)@self.w
        return (np.dot(y,y)/2+(self.l/2)*np.dot(self.w,self.w))/len(y)
    def dloss(self,x,y):
        """
        (d/dw)loss=XT(Xw-y)+lw
        """
        X=self._getX(x)
        return X.T@(X@self.w-y)+self.l*self.w
    def train_by_analysis(self,train_set,**datasets):
        """
        Trains the model by solving the equation directly.

        datasets    loss values on these datasets in training progress will be returned as a dict
        """
        datasets['train_set']=train_set
        loss_dict=dict()
        for key in datasets:
            loss_dict[key]=[]
        x,y=train_set
        X=self._getX(x)
        print("training...")
        self.w=np.linalg.inv(X.T@X+self.l*np.eye(self.M+1))@X.T@y
        for key in datasets.keys():
            loss_dict[key].append(self.loss(datasets[key][0],datasets[key][1]))
        print("training finished.")
        return loss_dict
    def train_by_gradient_descent(self,
                          train_set,
                          learning_rate,
                          step,
                          limit=1e-6,
                          batch_size=0,
                          w=None,
                          **datasets):
        """
        Trains the model by gradient descent.
        If the batch size is set to 0, the training method is normal gradient descent,
        else the method is random gradient descent.

        learning_rate   learning rate for every step
        step            number of training steps
        limit           gradient bottom limit
        batch_size      size of each batch. 0 for all data, >0 for random batch
        w               initial value of w, default to 0
        datasets        loss values on these datasets in training progress will be returned as a dict
        """
        datasets['train_set']=train_set
        loss_dict=dict()
        for key in datasets:
            loss_dict[key]=[]
        x,y=train_set
        self.w=np.zeros(self.M+1) if w is None else w
        print("training...")
        for i in range(step):
            train_x,train_y=self._get_batch(x,y,batch_size)
            d=self.dloss(train_x,train_y)
            d2=np.linalg.norm(d,2)
            if d2<limit: break
            self.w-=learning_rate*(d/d2)
            for key in datasets.keys():
                loss_dict[key].append(self.loss(datasets[key][0],datasets[key][1]))
            print("\rtraining process: %d%%    training loss=%.3f"%((i+1)*100//step,self.loss(x,y)),end='')
        print("\ntraining finished.")
        return loss_dict
    def train_by_conjugate_gradient(self,train_set,step=None,w=None,**datasets):
        """
        Trains the model by conjugate gradient method.

        w           initial value of w, default to 0
        step        number of training steps
        datasets    loss values on these datasets in training progress will be returned as a dict
        """
        datasets['train_set']=train_set
        loss_dict=dict()
        for key in datasets:
            loss_dict[key]=[]
        x,y=train_set
        self.w=np.zeros(self.M+1) if w is None else w
        X=self._getX(x)
        print("training...")
        A=X.T@X+self.l*np.eye(self.M+1)
        b=X.T@y
        r=b-A@self.w
        p=r
        for k in range(step if step is not None else self.M+1):
            alpha=np.dot(r,r)/np.dot(p,A@p) # calculate the step length
            self.w+=alpha*p # step once
            beta=np.dot(r,r)
            r=r-alpha*A@p
            beta=np.dot(r,r)/beta
            p=r+beta*p # calculate the direction of the next step
            for key in datasets.keys():
                loss_dict[key].append(self.loss(datasets[key][0],datasets[key][1]))
        print("training finished.")
        return loss_dict
    def predict(self,x):
        """
        Returns the predict value(s) for the given x.
        """
        return sum([self.w[i]*x**i for i in range(self.M+1)])
    def curve(self,L=-1,R=1):
        """
        Returns a curve of this model for plotting.
        """
        x=np.linspace(L,R,self.plot_N)
        return x,self.predict(x)
