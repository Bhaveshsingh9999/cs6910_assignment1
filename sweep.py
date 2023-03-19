"""
code to run sweep in wandb 

"""



import argparse
#!wandb login 44fa96b263794ea519fb29399eb6b8f469eb934b   


#parser=argparse.ArgumentParser()


# parser.add_argument("-wp","--wandb_project",type=str, default="DLassignment1")
# parser.add_argument("-we","--wandb_entity", type=str, default="singhbhavesh999")
# parser.add_argument("-d","--dataset", type=str,choices=["fashion_mnist","mnist"],  default="fashion_mnist")
# parser.add_argument("-e","--epochs",type=int,default=10)
# parser.add_argument("-b","--batch_size",type=int,default=16)
# parser.add_argument("-l","--loss",choices=["mean_squared_error", "cross_entropy"],default="cross_entropy")
# parser.add_argument("-o","--optimizer",choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],default="momentum")
# parser.add_argument("-lr","--learning_rate",type=float,default=0.001)
# parser.add_argument("-a","--activation",choices=["sigmoid", "tanh", "ReLU"],default="tanh")
# parser.add_argument("-sz","--hidden_size",type=int,default=128)
# parser.add_argument("-nhl","--num_layers",type=int,default=3)
# parser.add_argument("-w_i", "--weight_init",choices=["random", "Xavier"],default="Xavier")
# parser.add_argument("-eps", "--epsilon", type=float , default = 1e-3)
# parser.add_argument("-beta2","--beta2",type=float, default=0.999)
# parser.add_argument("-beta1","--beta1",type=float, default=0.9)
# parser.add_argument("-beta","--beta",type=float, default=0.9)
# parser.add_argument("-w_d","--weight_decay",type=float, default=0.0001)
# parser.add_argument("-m","--momentum",type=float, default=0.9)


# arg=parser.parse_args()

# projectn=arg.wandb_project
# entityn=arg.wandb_entity
# datasetn=arg.dataset
# mometum=arg.momentum
# epochs=arg.epochs
# batch_size=arg.batch_size
# loss=arg.loss
# optimizer=arg.optimizer
# learning_rate=arg.learning_rate
# activation_fn=arg.activation
# no_hidden_layer=arg.num_layers
# hidden_layer_size=arg.hidden_size
# weight_init_fn=arg.weight_init
# epsillon=arg.epsilon
# beta=arg.beta
# beta1=arg.beta1
# beta2=arg.beta2
# weight_decay=arg.weight_decay



default_params=dict(
epochs=10,
batch_size=32,
weight_decay=0,
optimizer='nadam',
learning_rate=0.001,
activation_fn='ReLU',
no_hidden_layer=3,
hidden_layer_size=128,
weight_init_fn='Xavier',
)

epsillon=1e-3
loss='cross_entropy'
beta=0.9
beta1=0.9
beta2=0.999
weight_decay=0.0001
datasetn='fashion_mnist'




import wandb
import numpy as np
from sklearn.model_selection import train_test_split

if(datasetn=='fashion_mnist'):
    from keras.datasets import fashion_mnist
    (train_X,train_Y),(test_X,test_Y)=fashion_mnist.load_data()
elif(datasetn=='mnist'):
    from keras.datasets import mnist
    (train_X,train_Y),(test_X,test_Y)=mnist.load_data()


#normalize the train dataset as we values are from 0-255
train_X=train_X/255
test_X=test_X/255

#for validation prepare data
X_train, X_validation, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.1, random_state=10)
run=wandb.init(config=default_params,project='DLassignment1',entity='singhbhavesh999',reinit='true')
#run=wandb.init(project=projectn,entity=entityn,reinit='true')
config=wandb.config


#transforming dataset to 2d matrix whith col representing the data  
no_of_datapoints=X_train.shape[0]
col_dim=X_train.shape[1]*X_train.shape[2]
x_data=X_train.reshape(no_of_datapoints,col_dim)
x_data=x_data.T   #input is x_data

#transform the validation dataset 
no_of_datapoints_val=X_validation.shape[0]
x_val=X_validation.reshape(no_of_datapoints_val,col_dim)
x_val=x_val.T


#transform the testdata set
no_of_datapoints_test=test_X.shape[0]
x_test=test_X.reshape(no_of_datapoints_test,col_dim)
x_test=x_test.T  #to test use this data


def sigmoid(z):
  g = 1 / (1 + np.exp(-z))
  return g

def tanh(a):
  g=np.tanh(a)
  return g

def relu(z):
  return np.maximum(0,z)

def softmax(x):
  x=x.T
  for i in range(x.shape[0]):
    sum=0
    max_ele=np.max(x[i])
    for j in range(x.shape[1]):
      sum+=np.exp(x[i][j]-max_ele)
    x[i]=np.exp(x[i]-max_ele)/sum
  x=x.T

  return x

def sigmoid_der(z):
  t=sigmoid(z)
  return np.multiply(t,(1-t))

def tanh_der(a):
  t=np.tanh(a)
  return 1-np.square(t)

def relu_der(a):
  a[a<=0]=0
  a[a>0]=1
  return a

def cross_entropy_loss(y_hat,y,weight,alpha):
  loss=0
  l2reg=0
  for i in range(y.shape[0]):
    loss+=-np.log2(y_hat[i][y[i]]+1e-8)
  

  for i in range(len(weight)):
    l2reg+=np.sum((weight[i])**2)
  l2reg=((l2reg*alpha)/2)
  

  loss+=l2reg
  
  return loss/y.shape[0]

def softmax_der(a):
    t = softmax(a)
    return np.multiply(t,(1-t))

def mse_loss(y_hat,y,weight,alpha):
  loss=0
  for i in range(y_hat.shape[0]):
    for j in range(y_hat.shape[1]):
      if(j!=y[i]):
        loss+=np.square(y_hat[i][j])
      else:
        loss+=np.square(y_hat[i][j]-1)
  l2reg=0  

  for i in range(len(weight)):
    l2reg+=np.sum((weight[i])**2)
  l2reg=((l2reg*alpha)/2)
 

  loss+=l2reg
  loss=loss/y_hat.shape[0]

  return loss

def forward_propogation(input,weight,bias,no_hidden_layer,output,activation_function,loss_fn):
  activation=[]  #input to above layer
  preactivation=[] 
  #no of layers =  no of layers
  #a=wx+b   and h is sigmoid of a   input data is 
  activation.append(input)
  preactivation.append(input)
  for i in range(no_hidden_layer):
    temp_preactive = np.matmul(weight[i],activation[i])+bias[i]
    
    if(i==0 and activation_function=='ReLU'):
      temp_preactive=temp_preactive.T
      for i in range(temp_preactive.shape[0]):
        temp_preactive[i]=temp_preactive[i]/np.max(temp_preactive[i])
      temp_preactive=temp_preactive.T


    preactivation.append(temp_preactive)
    if(activation_function=='sigmoid'):
      activation.append(sigmoid(np.copy(temp_preactive)))
    elif(activation_function=='tanh'):
      activation.append(tanh(np.copy(temp_preactive)))
    elif(activation_function=='ReLU'):
      activation.append(relu(np.copy(temp_preactive)))
      
  temp_preactive=np.matmul(weight[-1],activation[-1])+bias[-1]
  

  preactivation.append(temp_preactive)
  activation.append(softmax(np.copy(temp_preactive)))


  
  return activation,preactivation

def backward_propogation(activation,output,no_hidden_layer,weights,a,batch_size,activation_function,weight_decay,loss_fn,preactivation):
  #gradient_activation=[]
  gradient_weight=[]
  gradient_bias=[]
  
  # we need one hot vector for output in the format 10*60000 
  hot=np.zeros((activation[-1].shape[1],activation[-1].shape[0]))
  for i in range(activation[-1].shape[1]):
    hot[i][output[i]]=1
  
  hot=hot.T
  gradient_weight=[]
  gradiend_bias=[]
  if(loss_fn=='cross_entropy'):
    gradient_activation=-(hot-activation[-1])
  elif(loss_fn=='mean_squared_error'):
    gradient_activation=np.multiply((activation[-1]-hot),softmax_der(np.copy(preactivation[-1])))




  for i in range(no_hidden_layer,-1,-1):
    weight_temp=np.matmul(gradient_activation,activation[i].T)/batch_size
    bias_temp=(np.sum(gradient_activation,axis=1)/batch_size).reshape(-1,1)
    gradient_weight.append(weight_temp)
    gradient_bias.append(bias_temp)
    if(i!=0):
      gradient_temp=np.matmul(weights[i].T,gradient_activation)
      if(activation_function=='sigmoid'):
        gradient_activation=gradient_temp*sigmoid_der(np.copy(a[i]))
      elif(activation_function=='tanh'):
        gradient_activation=gradient_temp*(tanh_der(np.copy(a[i])))
      elif(activation_function=='ReLU'):
        gradient_activation=gradient_temp*(np.copy(relu_der(a[i])))
  
  
  gradient_weight.reverse()
  gradient_bias.reverse()

  for i in range(len(gradient_weight)):
    gradient_weight[i]+=weight_decay*weights[i]


  return gradient_weight,gradient_bias


def batch_gradient_descent(input,output,epochs,learning_rate,batch_size,activation_function,loss_fn,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay):
  weight,bias = init_layer(no_hidden_layer,hidden_layer_size,weight_init_fn)
  no_batches=input.shape[1]/batch_size
  

  for i in range(epochs):
    # take theinput and spilt it in batch and then provide that as input to
    loss_total=0
    for j in range(0,input.shape[1],batch_size):
      x_batch= input[:,j:j+batch_size]
      y_batch=  output[j:j+batch_size] 
      activation,a=forward_propogation(x_batch,weight,bias,no_hidden_layer,y_batch,activation_function,loss_fn)
      gradient_weight,gradient_bias=backward_propogation(activation,y_batch,no_hidden_layer,weight,a,batch_size,activation_function,weight_decay,loss_fn,a)

      
      
      for k in range(len(weight)):
        weight[k]=weight[k]-learning_rate*(gradient_weight[k])
        bias[k]=bias[k]-learning_rate*gradient_bias[k]

    train_accuracy,train_loss=test_data_accuracy(input,weight,bias,no_hidden_layer,output,activation_function,loss_fn,weight_decay)   
    validation_accuracy,validation_loss=test_data_accuracy(x_val,weight,bias,no_hidden_layer,y_val,activation_function,loss_fn,weight_decay)
    wandb.log({"train_accuracy":train_accuracy,"train_error":train_loss,"val_accuracy":validation_accuracy,"val_error":validation_loss})
    print("epoch is ", i ," train loss ",train_loss,' train accu ',train_accuracy, ' validation accuracy is ',validation_accuracy,' validation loss ',validation_loss)
    

    
def momentum_gradient_descent(input,output,epochs,learning_rate,batch_size,mometum,activation_function,loss_fn,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay):
  weight,bias = init_layer(no_hidden_layer,hidden_layer_size,weight_init_fn)
  ut_weight=[]
  ut_bias=[]
  ut_weight.extend([0 for i in range(len(weight))])
  ut_bias.extend([0 for i in range(len(bias))])
  
  no_batches=input.shape[1]/batch_size
  for i in range(epochs):
    # take theinput and spilt it in batch and then provide that as input to
    loss_total=0
    for j in range(0,input.shape[1],batch_size):
      x_batch= input[:,j:j+batch_size]
      y_batch=  output[j:j+batch_size] 
      activation,a=forward_propogation(x_batch,weight,bias,no_hidden_layer,y_batch,activation_function,loss_fn)
      gradient_weight,gradient_bias=backward_propogation(activation,y_batch,no_hidden_layer,weight,a,batch_size,activation_function,weight_decay,loss_fn,a)
     
      

      for k in range(len(weight)):
        ut_weight[k]=mometum*ut_weight[k]+learning_rate*gradient_weight[k]
        ut_bias[k]=mometum*ut_bias[k]+learning_rate*gradient_bias[k]
        weight[k]=weight[k]-ut_weight[k]
        bias[k]=bias[k]-ut_bias[k]

    train_accuracy,train_loss=test_data_accuracy(input,weight,bias,no_hidden_layer,output,activation_function,loss_fn,weight_decay)   
    validation_accuracy,validation_loss=test_data_accuracy(x_val,weight,bias,no_hidden_layer,y_val,activation_function,loss_fn,weight_decay)
    wandb.log({"train_accuracy":train_accuracy,"train_error":train_loss,"val_accuracy":validation_accuracy,"val_error":validation_loss})
    print("epoch is ", i ," train loss ",train_loss,' train accu ',train_accuracy, ' validation accuracy is ',validation_accuracy,' validation loss ',validation_loss)
  # predict(x_test,weight,bias,no_hidden_layer,activation_fn,loss_fn)


def nesterov_gradient_descent(input,output,epochs,learning_rate,batch_size,mometum,activation_function,loss_fn,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay):
  weight,bias = init_layer(no_hidden_layer,hidden_layer_size,weight_init_fn)
  ut_weight=[]
  ut_bias=[]
  ut_weight.extend([0 for i in range(len(weight))])
  ut_bias.extend([0 for i in range(len(bias))])
  no_batches=input.shape[1]/batch_size
  for i in range(epochs):
    # take theinput and spilt it in batch and then provide that as input to
    loss_total=0
    for j in range(0,input.shape[1],batch_size):
      x_batch= input[:,j:j+batch_size]
      y_batch=  output[j:j+batch_size]
      for k in range(len(weight)):
        weight[k]= weight[k]-mometum*ut_weight[k]
        bias[k]=bias[k]-mometum*ut_bias[k]
      activation,a=forward_propogation(x_batch,weight,bias,no_hidden_layer,y_batch,activation_function,loss_fn)
      gradient_weight,gradient_bias=backward_propogation(activation,y_batch,no_hidden_layer,weight,a,batch_size,activation_function,weight_decay,loss_fn,a)
      #print(loss)

      

      for k in range(len(weight)):
        ut_weight[k]=mometum*ut_weight[k]+learning_rate*gradient_weight[k]
        ut_bias[k]=mometum*ut_bias[k]+learning_rate*gradient_bias[k]
        weight[k]=weight[k]-ut_weight[k]
        bias[k]=bias[k]-ut_bias[k]

    train_accuracy,train_loss=test_data_accuracy(input,weight,bias,no_hidden_layer,output,activation_function,loss_fn,weight_decay)   
    validation_accuracy,validation_loss=test_data_accuracy(x_val,weight,bias,no_hidden_layer,y_val,activation_function,loss_fn,weight_decay)
    wandb.log({"train_accuracy":train_accuracy,"train_error":train_loss,"val_accuracy":validation_accuracy,"val_error":validation_loss})
    print("epoch is ", i ," train loss ",train_loss,' train accu ',train_accuracy, ' validation accuracy is ',validation_accuracy,' validation loss ',validation_loss)


def rmsprop_gradient_descent(input,output,epochs,learning_rate,batch_size,beta,epsillon,activation_function,loss_fn,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay):
  weight,bias = init_layer(no_hidden_layer,hidden_layer_size,weight_init_fn)
  ut_weight=[]
  ut_bias=[]
  ut_weight.extend([0 for i in range(len(weight))])
  ut_bias.extend([0 for i in range(len(bias))])
  no_batches=input.shape[1]/batch_size
  for i in range(epochs):
    # take theinput and spilt it in batch and then provide that as input to
    loss_total=0
    for j in range(0,input.shape[1],batch_size):
      x_batch= input[:,j:j+batch_size]
      y_batch=  output[j:j+batch_size] 
      activation,a=forward_propogation(x_batch,weight,bias,no_hidden_layer,y_batch,activation_function,loss_fn)
      gradient_weight,gradient_bias=backward_propogation(activation,y_batch,no_hidden_layer,weight,a,batch_size,activation_function,weight_decay,loss_fn,a)
      #print(loss)

      

      for k in range(len(weight)):
        ut_weight[k]=beta*ut_weight[k]+(1-beta)*np.square(gradient_weight[k])
        ut_bias[k]=beta*ut_bias[k]+(1-beta)*np.square(gradient_bias[k])
        weight[k]=weight[k]-learning_rate*((gradient_weight[k]/np.sqrt(ut_weight[k]+epsillon)))
        bias[k]=bias[k]-learning_rate*(gradient_bias[k]/np.sqrt(ut_bias[k]+epsillon))

    train_accuracy,train_loss=test_data_accuracy(input,weight,bias,no_hidden_layer,output,activation_function,loss_fn,weight_decay)   
    validation_accuracy,validation_loss=test_data_accuracy(x_val,weight,bias,no_hidden_layer,y_val,activation_function,loss_fn,weight_decay)
    wandb.log({"train_accuracy":train_accuracy,"train_error":train_loss,"val_accuracy":validation_accuracy,"val_error":validation_loss})
    print("epoch is ", i ," train loss ",train_loss,' train accu ',train_accuracy, ' validation accuracy is ',validation_accuracy,' validation loss ',validation_loss)


def adam_gradient_descent(input,output,epochs,learning_rate,batch_size,beta1,beta2,epsillon,activation_function,loss_fn,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay):
  weight,bias = init_layer(no_hidden_layer,hidden_layer_size,weight_init_fn)
  mt_weight=[]
  mt_bias=[]
  vt_weight=[]
  vt_bias=[]
  mt_weight.extend([0 for i in range(len(weight))])
  mt_bias.extend([0 for i in range(len(bias))])
  vt_weight.extend([0 for i in range(len(weight))])
  vt_bias.extend([0 for i in range(len(bias))])

  mt_hat_weight=[]
  vt_hat_weight=[]
  mt_hat_bias=[]
  vt_hat_bias=[]
  mt_hat_weight.extend([0 for i in range(len(weight))])
  vt_hat_weight.extend([0 for i in range(len(weight))])
  mt_hat_bias.extend([0 for i in range(len(weight))])
  vt_hat_bias.extend([0 for i in range(len(weight))])
  no_batches=input.shape[1]/batch_size

  t=0
  for i in range(epochs):
    # take theinput and spilt it in batch and then provide that as input to
    loss_total=0
    for j in range(0,input.shape[1],batch_size):
      t+=1
      x_batch= input[:,j:j+batch_size]
      y_batch=  output[j:j+batch_size] 
      activation,a=forward_propogation(x_batch,weight,bias,no_hidden_layer,y_batch,activation_function,loss_fn)
      gradient_weight,gradient_bias=backward_propogation(activation,y_batch,no_hidden_layer,weight,a,batch_size,activation_function,weight_decay,loss_fn,a)
      #print(loss)

      

      for k in range(len(weight)):
        mt_weight[k]=beta1*mt_weight[k]+(1-beta1)*gradient_weight[k]
        mt_bias[k]=beta1*mt_bias[k]+(1-beta1)*gradient_bias[k]
        
        mt_hat_weight[k]=mt_weight[k]/(1-(beta1**t))
        mt_hat_bias[k]=mt_bias[k]/(1-(beta1**t))



        vt_weight[k]=beta2*vt_weight[k]+(1-beta2)*(np.square(gradient_weight[k]))
        vt_bias[k]=beta2*vt_bias[k]+(1-beta2)*(np.square(gradient_weight[k]))
        
        vt_hat_weight[k]=vt_hat_weight[k]/(1-(beta2**t))
        vt_hat_bias[k]=vt_hat_bias[k]/(1-(beta2**t))

        weight[k]=weight[k]-(learning_rate*(np.divide(mt_hat_weight[k],(np.sqrt(vt_hat_weight[k]+epsillon)))))
        bias[k]=bias[k]-(learning_rate)*(np.divide(mt_hat_bias[k],(np.sqrt(vt_hat_bias[k]+epsillon))))


    train_accuracy,train_loss=test_data_accuracy(input,weight,bias,no_hidden_layer,output,activation_function,loss_fn,weight_decay)   
    validation_accuracy,validation_loss=test_data_accuracy(x_val,weight,bias,no_hidden_layer,y_val,activation_function,loss_fn,weight_decay)
    wandb.log({"train_accuracy":train_accuracy,"train_error":train_loss,"val_accuracy":validation_accuracy,"val_error":validation_loss})
    print("epoch is ", i ," train loss ",train_loss,' train accu ',train_accuracy, ' validation accuracy is ',validation_accuracy,' validation loss ',validation_loss)


def nadam_gradient_descent(input,output,epochs,learning_rate,batch_size,beta1,beta2,epsillon,activation_function,loss_fn,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay):
  weight,bias = init_layer(no_hidden_layer,hidden_layer_size,weight_init_fn)
  mt_weight=[]
  mt_bias=[]
  vt_weight=[]
  vt_bias=[]
  mt_weight.extend([0 for i in range(len(weight))])
  mt_bias.extend([0 for i in range(len(bias))])
  vt_weight.extend([0 for i in range(len(weight))])
  vt_bias.extend([0 for i in range(len(bias))])

  mt_hat_weight=[]
  vt_hat_weight=[]
  mt_hat_bias=[]
  vt_hat_bias=[]
  mt_hat_weight.extend([0 for i in range(len(weight))])
  vt_hat_weight.extend([0 for i in range(len(weight))])
  mt_hat_bias.extend([0 for i in range(len(weight))])
  vt_hat_bias.extend([0 for i in range(len(weight))])
  no_batches=input.shape[1]/batch_size



  t=0
  for i in range(epochs):
    # take theinput and spilt it in batch and then provide that as input to
    loss_total=0
    for j in range(0,input.shape[1],batch_size):
      t+=1
      x_batch= input[:,j:j+batch_size]
      y_batch=  output[j:j+batch_size] 
      activation,a=forward_propogation(x_batch,weight,bias,no_hidden_layer,y_batch,activation_function,loss_fn)
      gradient_weight,gradient_bias=backward_propogation(activation,y_batch,no_hidden_layer,weight,a,batch_size,activation_function,weight_decay,loss_fn,a)
      #print(loss)

      

      for k in range(len(weight)):
        mt_weight[k]=beta1*mt_weight[k]+(1-beta1)*gradient_weight[k]
        mt_bias[k]=beta1*mt_bias[k]+(1-beta1)*gradient_bias[k]
        
        mt_hat_weight[k]=mt_weight[k]/(1-(beta1**t))
        mt_hat_bias[k]=mt_bias[k]/(1-(beta1**t))



        vt_weight[k]=beta2*vt_weight[k]+(1-beta2)*(np.square(gradient_weight[k]))
        vt_bias[k]=beta2*vt_bias[k]+(1-beta2)*(np.square(gradient_weight[k]))
        
        vt_hat_weight[k]=vt_hat_weight[k]/(1-(beta2**t))
        vt_hat_bias[k]=vt_hat_bias[k]/(1-(beta2**t))

        weight[k]=weight[k]-(learning_rate*(np.divide(beta1*mt_hat_weight[k]+((1-beta1)*gradient_weight[k])/(1-(beta1**t)),(np.sqrt(vt_hat_weight[k]+epsillon)))))
        bias[k]=bias[k]-(learning_rate)*(np.divide(beta1*mt_hat_bias[k]+((1-beta1)*gradient_bias[k])/(1-(beta1**t)),(np.sqrt(vt_hat_bias[k]+epsillon))))


    train_accuracy,train_loss=test_data_accuracy(input,weight,bias,no_hidden_layer,output,activation_function,loss_fn,weight_decay)   
    validation_accuracy,validation_loss=test_data_accuracy(x_val,weight,bias,no_hidden_layer,y_val,activation_function,loss_fn,weight_decay)
    wandb.log({"train_accuracy":train_accuracy,"train_error":train_loss,"val_accuracy":validation_accuracy,"val_error":validation_loss})
    print("epoch is ", i ," train loss ",train_loss,' train accu ',train_accuracy, ' validation accuracy is ',validation_accuracy,' validation loss ',validation_loss)

#confusion matrix ke liye 

def conf_matrix(input,weight,bias,no_hidden_layer,activation_fn,loss_fn):
  activation,preactivation=forward_propogation(input,weight,bias,no_hidden_layer,test_Y,activation_fn,loss_fn)
  
  last_layer_data=activation[-1].T
  predicted_output=last_layer_data.argmax(axis=1)
  #cm=confusion_matrix(test_Y,predicted_output)
  #print(cm)
  label_data=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
  actual_y=test_Y
  wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(preds=predicted_output, y_true=actual_y,class_names=label_data)})

  






def init_layer(no_hidden_layer,hidden_layer_size,weight_init_fn):
  weight=[]
  bias=[]
  #initialize h1 weight and bias
  layer=[]
  layer.append(784)
  for i in range(no_hidden_layer):
    layer.append(hidden_layer_size)
  layer.append(10)
  np.random.seed(10)

  if(weight_init_fn=='random'):
     for i in range(no_hidden_layer+1):
       weight.append(np.random.uniform(-0.5,0.5,(layer[i+1],layer[i])))
       bias.append(np.random.uniform(-0.5,0.5,(layer[i+1],1)))

  elif(weight_init_fn=='Xavier'):
    for i in range(no_hidden_layer+1):
      sigma = np.sqrt(6/(layer[i]+layer[i+1]))
      weight.append(np.random.uniform(-sigma,sigma,(layer[i+1],layer[i])))
      bias.append(np.random.uniform(-sigma,sigma,(layer[i+1],1)))
      

  return weight,bias

def test_data_accuracy(test_input,weight,bias,no_hidden_layer,test_output,activation_fn,loss_fn,weight_decay):
  #filal abhi yaha likh de badh mai pass karna 
  

  activation=[]
  activation,preactivation=forward_propogation(test_input,weight,bias,no_hidden_layer,test_output,activation_fn,loss_fn)
  temp=activation[-1].T
  if(loss_fn=='cross_entropy'):
    loss=cross_entropy_loss(temp,test_output,weight,weight_decay)
  elif(loss_fn=='mean_squared_error'):
    loss=mse_loss(temp,test_output,weight,weight_decay)

  y_hat=np.argmax(activation[-1].T,axis=1)
  count=0
  for i in range(test_output.shape[0]):
    if (y_hat[i]==test_output[i]):
      count+=1
  accuracy=count/test_output.shape[0]
  return accuracy,loss



def main_call(x_data,y_train,epochs,learning_rate,batch_size,beta,beta1,beta2,epsillon,activation_fn,loss,no_hidden_layer,hidden_layer_size,optimizer,weight_init_fn,weight_decay):
  
  if(optimizer=='sgd'):
    print('sgd')
    batch_gradient_descent(x_data,y_train,epochs,learning_rate,batch_size,activation_fn,loss,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay)

  elif(optimizer=='momentum'):
    print('momentum')
    momentum_gradient_descent(x_data,y_train,epochs,learning_rate,batch_size,beta,activation_fn,loss,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay)
    



  elif(optimizer=='nag'):
    print('nag')
    nesterov_gradient_descent(x_data,y_train,epochs,learning_rate,batch_size,beta,activation_fn,loss,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay)

  elif(optimizer=='rmsprop'):
    print('rmsprop')
    rmsprop_gradient_descent(x_data,y_train,epochs,learning_rate,batch_size,beta,epsillon,activation_fn,loss,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay)


  elif(optimizer=='adam'):
    print('adam')
    adam_gradient_descent(x_data,y_train,epochs,learning_rate,batch_size,beta1,beta2,epsillon,activation_fn,loss,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay)


  elif(optimizer=='nadam'):
    print('nadam')
    nadam_gradient_descent(x_data,y_train,epochs,learning_rate,batch_size,beta1,beta2,epsillon,activation_fn,loss,no_hidden_layer,hidden_layer_size,weight_init_fn,weight_decay)



epochs=config.epochs
batch_size=config.batch_size

optimizer=config.optimizer
learning_rate=config.learning_rate
activation_fn=config.activation_fn
no_hidden_layer=config.no_hidden_layer
hidden_layer_size=config.hidden_layer_size
weight_init_fn=config.weight_init_fn
weight_decay=config.weight_decay

run.name='hl_'+str(no_hidden_layer)+'_bs_'+str(batch_size)+'_ac_'+activation_fn+'_hls_'+str(hidden_layer_size)+'_lr_'+str(learning_rate)

main_call(x_data,y_train,epochs,learning_rate,batch_size,beta,beta1,beta2,epsillon,activation_fn,loss,no_hidden_layer,hidden_layer_size,optimizer,weight_init_fn,weight_decay)








