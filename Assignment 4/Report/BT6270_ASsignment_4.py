import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def import_image(name):
    '''
    Import image from .txt files and return arrays
    '''
    df=pd.read_csv(name+'.txt',header=None)
    name_arr=np.array(df)
    name_arr=np.sign(name_arr)
    plt.imshow(name_arr,cmap='Greys_r')
    plt.title(f'Image of {name}')
    plt.show()
    return name_arr
    
ball=import_image('ball')
mona=import_image('mona')
cat=import_image('cat')


vec_size=ball.shape[0]*ball.shape[1]
u=np.zeros(vec_size)
ball_S=np.reshape(ball,(vec_size,1))
cat_S=np.reshape(cat,(vec_size,1))
mona_S=np.reshape(mona,(vec_size,1))


class Hopfield_Net():
    def __init__(self,niter):
        self.V = np.zeros((9000,1))
        self.U = np.zeros((9000,1))
        self.weights = np.zeros((9000,9000))
        self.U_d = np.zeros((9000,1))
        self.rmse = np.zeros((niter,1))
        self.flag = 0 # to load all images or only ball
        
    def load_weights(self):
        '''
        loads all images
        '''
        if self.flag==1:
            print('Loading all images')
            self.weights = np.matmul(mona_S,mona_S.T) + np.matmul(ball_S,ball_S.T) + np.matmul(cat_S,cat_S.T)
        if self.flag==0:
            print('Loading the image of the ball')
            self.weights = np.matmul(ball_S,ball_S.T)
        
    def image_loader(self,image):
        '''
        Loads patches of images
        '''
        new_image = np.zeros((90,100))
        new_image[0:45,25:50] = image[0:45,25:50]
        return new_image
        
    def damage_weights(self,p):
        '''
        Damages the weights of the network with probability p
        '''
        indices = np.random.randint(0,9000*9000-1,int(9000*9000*p))
        weights_damaged=np.copy(self.weights)
        weights_damaged=np.reshape(weights_damaged,(9000*9000,1))
        print('Damaging the weights')
        for i in tqdm(range(len(indices))):
            weights_damaged[indices[i]]=0
        weights_damaged = np.reshape(weights_damaged,(9000,9000))
        return weights_damaged
            
                
        
        
def demo(niter,lambdas,flag,p):
    dt=1/(100)
    Hop_net1=Hopfield_Net(niter)
    Hop_net1.flag=flag
    Hop_net1.load_weights()
    Hop_net1.U = np.reshape(Hop_net1.image_loader(ball),(9000,1))
    Hop_net1.weights=Hop_net1.damage_weights(p)
    Hop_net1.weights=Hop_net1.weights/9000
    images_arr=[]
    for i in tqdm(range(niter)):
        Hop_net1.U_d = -Hop_net1.U + np.matmul(Hop_net1.weights,Hop_net1.V)
        Hop_net1.U = Hop_net1.U + (Hop_net1.U_d)*dt
        Hop_net1.V = np.tanh(lambdas*Hop_net1.U)
        Hop_net1.rmse[i]=mean_squared_error(ball_S,Hop_net1.V)
        
        img=np.reshape(Hop_net1.V,(90,100))
        images_arr.append(img)
    images_arr=np.array(images_arr)
    return images_arr,Hop_net1.rmse
    
def show(images_arr,rmse,niter,p):
    images_arr=np.array(images_arr)
    for i in range(int(niter/10)):
        plt.imshow(images_arr[10*i,:,:],'Greys_r')
        plt.title(f'Image after {10*i} iterations for {p*100}% of weight damage')
        plt.show()
        
    plt.plot(rmse)
    plt.title(f'Plot of RMSE for {p*100}% of weight damage')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.grid()
    plt.show()
    
niter=50
images_arr,rmse=demo(niter,10,0,0)   # for loading ball without damage
show(images_arr,rmse,niter,0)

niter=100
images_arr,rmse=demo(niter,10,1,0.25)   # for loading all images with 25% damage
show(images_arr,rmse,niter,0.25)

images_arr,rmse=demo(niter,10,1,0.5)   # for loading all images with 50% damage
show(images_arr,rmse,niter,0.5)

images_arr,rmse=demo(niter,10,1,0.8)   # for loading all images with 75% damage
show(images_arr,rmse,niter,0.8)

