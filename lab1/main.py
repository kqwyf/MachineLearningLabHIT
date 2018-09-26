import numpy as np
import math
import time
import matplotlib.pyplot as plt
from polymodel import poly_model

L,R=-1,1

def f(x):
    return np.sin(2*math.pi*x)

def generate_data(n,noise=0):
    x=np.linspace(L,R,n)
    y=np.array([f(i)+np.random.normal(scale=noise) for i in x])
    return x,y

train_set=generate_data(100,noise=0.5)
test_set=generate_data(20,noise=0.5)
validate_set=generate_data(20,noise=0.5)

######################################################

print("Analysis solution without regularization test")
sizes=[5,10,50]
for i in range(3):
    model=poly_model(size=sizes[i])
    loss_dict=model.train_by_analysis(train_set,
                                      test_set=test_set,
                                      validate_set=validate_set)
    print("validate loss=%.3f    test loss=%.3f\n"%(loss_dict['validate_set'][-1],loss_dict['test_set'][-1]))
    plot_x,plot_y=model.curve()
    plt.scatter(train_set[0],train_set[1],c='g',label='training set')
    plt.plot(plot_x,plot_y,c='r',label='model curve')
    plt.plot(np.linspace(-1,1,100),f(np.linspace(-1,1,100)),c='black',label='f(x)')
    plt.legend()
    plt.show()
train_set=generate_data(1000,noise=0.5)
model=poly_model(size=50)
loss_dict=model.train_by_analysis(train_set,
                                  test_set=test_set,
                                  validate_set=validate_set)
print("validate loss=%.3f    test loss=%.3f\n"%(loss_dict['validate_set'][-1],loss_dict['test_set'][-1]))
plot_x,plot_y=model.curve()
plt.scatter(train_set[0],train_set[1],c='g',label='training set')
plt.plot(plot_x,plot_y,c='r',label='model curve')
plt.plot(np.linspace(-1,1,100),f(np.linspace(-1,1,100)),c='black',label='f(x)')
plt.legend()
plt.show()
print("")

######################################################

print("Analysis solution with regularization test")
train_set=generate_data(100,noise=0.5)
sizes=[5,10,50,50,50]
ls=[1e-3,1e-3,1e-3,1e-6,1]
for i in range(5):
    model=poly_model(size=sizes[i],l=ls[i])
    loss_dict=model.train_by_analysis(train_set,
                                      test_set=test_set,
                                      validate_set=validate_set)
    print("validate loss=%.3f    test loss=%.3f\n"%(loss_dict['validate_set'][-1],loss_dict['test_set'][-1]))
    plot_x,plot_y=model.curve()
    plt.scatter(train_set[0],train_set[1],c='g',label='training set')
    plt.plot(plot_x,plot_y,c='r',label='model curve')
    plt.plot(np.linspace(-1,1,100),f(np.linspace(-1,1,100)),c='black',label='f(x)')
    plt.legend()
    plt.show()
print("")

######################################################

print("Gradient descent test")
lrs=[1e-3,2e-1,1]
steps=[100,1000,10000,20000]
for i in range(3):
    for j in range(4):
        model=poly_model(size=50,l=1e-3)
        loss_dict=model.train_by_gradient_descent(train_set,
                                                  learning_rate=lrs[i],
                                                  step=steps[j],
                                                  test_set=test_set,
                                                  validate_set=validate_set)
        print("validate loss=%.3f    test loss=%.3f\n"%(loss_dict['validate_set'][-1],loss_dict['test_set'][-1]))
        plot_x,plot_y=model.curve()
        plt.subplot(2,4,j+1)
        plt.scatter(train_set[0],train_set[1],c='g',label='training set')
        plt.plot(plot_x,plot_y,c='r',label='model curve')
        plt.plot(np.linspace(-1,1,100),f(np.linspace(-1,1,100)),c='black',label='f(x)')
        plt.legend()
        plt.subplot(2,4,j+5)
        plt.plot(range(1,len(loss_dict['validate_set'])+1),loss_dict['validate_set'],label='validate loss')
        plt.plot(range(1,len(loss_dict['train_set'])+1),loss_dict['train_set'],label='train loss')
        plt.legend()
    plt.show()
print("")

######################################################

print("Random gradient descent test")
batches=[1,10,30]
steps=[100,1000,10000,20000]
for i in range(3):
    for j in range(4):
        model=poly_model(size=50,l=1e-3)
        start_time=time.time()
        loss_dict=model.train_by_gradient_descent(train_set,
                                                  learning_rate=2e-1,
                                                  step=steps[j],
                                                  batch_size=batches[i],
                                                  test_set=test_set,
                                                  validate_set=validate_set)
        end_time=time.time()
        print("validate loss=%.3f    test loss=%.3f"%(loss_dict['validate_set'][-1],loss_dict['test_set'][-1]))
        print("training time: %.2fs\n"%(end_time-start_time))
        plot_x,plot_y=model.curve()
        plt.subplot(2,4,j+1)
        plt.scatter(train_set[0],train_set[1],c='g',label='training set')
        plt.plot(plot_x,plot_y,c='r',label='model curve')
        plt.plot(np.linspace(-1,1,100),f(np.linspace(-1,1,100)),c='black',label='f(x)')
        plt.legend()
        plt.subplot(2,4,j+5)
        plt.plot(range(1,len(loss_dict['validate_set'])+1),loss_dict['validate_set'],label='validate loss')
        plt.plot(range(1,len(loss_dict['train_set'])+1),loss_dict['train_set'],label='train loss')
        plt.legend()
    plt.show()
print("")

######################################################

print("Conjugate gradient test")
sizes=[10,50,100,1000]
for i in range(4):
    train_set=generate_data(sizes[i],noise=0.5)
    model=poly_model(size=50,l=1e-3)
    loss_dict=model.train_by_conjugate_gradient(train_set,
                                                test_set=test_set,
                                                validate_set=validate_set)
    print("validate loss=%.3f    test loss=%.3f"%(loss_dict['validate_set'][-1],loss_dict['test_set'][-1]))
    plot_x,plot_y=model.curve()
    w=model.w
    model.train_by_analysis(train_set)
    analysis_x,analysis_y=model.curve()
    plt.subplot(2,4,i+1)
    plt.plot(analysis_x,analysis_y,label='analysis solution')
    plt.scatter(train_set[0],train_set[1],c='g',label='training set')
    plt.plot(plot_x,plot_y,c='r',label='model curve')
    plt.plot(np.linspace(-1,1,100),f(np.linspace(-1,1,100)),c='black',label='f(x)')
    plt.legend()
    plt.subplot(2,4,i+5)
    plt.plot(range(1,len(loss_dict['validate_set'])+1),loss_dict['validate_set'],label='validate loss')
    plt.plot(range(1,len(loss_dict['train_set'])+1),loss_dict['train_set'],label='train loss')
    if np.alltrue(np.abs(w-model.w)<1e-4):
        print("The training result is same as the analysis solution.\n")
    else:
        print("The training result is different from the analysis solution.\n")
    plt.legend()
plt.show()
