#bs_nn_train.py
#Trains an NeuralNetwork to approximate Black-Scholes European call prices and compares NN delta (autograd)
#Requires:Python 3.8+,numpy,scipy,pandas,matplotlib,torch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from scipy.stats import norm
from scipy.special import ndtr
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)


#Reproducibility
torch.manual_seed(0)
np.random.seed(0)

#Closed form Black-Scholes formula
def bsCallPrice(S,K,T,r,sigma,q):
    eps=1e-12
    sqrtT=np.sqrt(np.clip(T,eps,None)) #set sqrtT to a tiny positive number if T is negative
    d1=(np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
    d2=d1-sigma*sqrtT
    price=S*np.exp(-q*T)*ndtr(d1)-K*np.exp(-r*T)*ndtr(d2)
    return price,d1

#Create feature dataset and expected outputs
def generateDataset(n):
    S=np.random.uniform(10.0,200.0,n)
    K=np.random.uniform(10.0,200.0,n)
    T=np.random.uniform(0.01,2.0,n)
    r=np.random.uniform(0.0,0.10,n)
    sigma=np.random.uniform(0.05,1.00,n)
    q=np.random.uniform(0.0,0.05,n)
    price,d1=bsCallPrice(S,K,T,r,sigma,q)
    delta=np.exp(-q*T)*norm.cdf(d1)
    X=np.stack([S,K,T,r,sigma,q],1)#stack combines arrays into a matrix
    y=price.reshape(-1,1)
    return X.astype(np.float32),y.astype(np.float32),delta.astype(np.float32)


def bsLoss(model,Xb,yb,deltaTrue,stdDev,lambda_delta):
    #price loss
    predPrice=model(Xb) #carry out a forward pass to generate predictions
    lossPrice=nn.MSELoss()(predPrice,yb) #compute MSE

    #delta loss
    Xb.requires_grad_(True)
    predPriceSum=model(Xb).sum()
    grad=torch.autograd.grad(predPriceSum,Xb)[0]

    #dPrice/dS=(grad wrt normalized S)/stdDev[0]
    dprice_dS_nn=grad[:,0]/stdDev[0]

    loss_delta=nn.MSELoss()(dprice_dS_nn,deltaTrue)

    #total loss
    loss=lossPrice+lambda_delta*loss_delta
    return loss,lossPrice.item(),loss_delta.item()


#config
N=120000
batch_size=2048
n_epochs=40
lr=1e-3

#Generate and split
X,y,delta=generateDataset(N)
perm=np.random.permutation(N)
train_idx=perm[:100000]
val_idx=perm[100000:110000]
test_idx=perm[110000:]

Xtrain,yTrain=X[train_idx],y[train_idx]
Xval,yVal=X[val_idx],y[val_idx]
Xtest,yTest=X[test_idx],y[test_idx]
delta_test=delta[test_idx]

#normalize (fit on training)
mean=Xtrain.mean(axis=0)
stdDev=Xtrain.std(axis=0)
XtrainNorm=(Xtrain-mean)/stdDev
XvalNorm=(Xval-mean)/stdDev
XtestNorm=(Xtest-mean)/stdDev

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

XtrainTensor=torch.tensor(XtrainNorm).to(device)
yTrainTensor=torch.tensor(yTrain).to(device)
XvalTensor=torch.tensor(XvalNorm).to(device)
yValTensor=torch.tensor(yVal).to(device)
XtestTensor=torch.tensor(XtestNorm).to(device)
yTestTensor=torch.tensor(yTest).to(device)

trainDataset=TensorDataset(XtrainTensor,yTrainTensor)
valDataset=TensorDataset(XvalTensor,yValTensor)
testDataset=TensorDataset(XtestTensor,yTestTensor)

trainLoader=DataLoader(trainDataset,batch_size,True)
valLoader=DataLoader(valDataset,batch_size,False)
testLoader=DataLoader(testDataset,batch_size,False)

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim=6,hidden=[128,128,64]):
        super().__init__()
        layers=[]
        dim=input_dim
        for h in hidden:
            layers.append(nn.Linear(dim,h))
            layers.append(nn.ReLU())
            dim=h
        layers.append(nn.Linear(dim,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

model=NeuralNetwork().to(device)
optimizer=optim.Adam(model.parameters(),lr=lr)
criterion=nn.MSELoss()

bestValLoss=float('inf')
for epoch in range(1,n_epochs+1):
    model.train()
    trainLosses=[]
    for xb,yb in trainLoader:
        optimizer.zero_grad()

        #clone minibatch and enable gradient for S
        Xb=xb.clone().detach().requires_grad_(True)

        #denormalize features
        S=Xb[:,0] * stdDev[0] + mean[0]
        K=Xb[:,1] * stdDev[1] + mean[1]
        T=Xb[:,2] * stdDev[2] + mean[2]
        r=Xb[:,3] * stdDev[3] + mean[3]
        sigma=Xb[:,4] * stdDev[4] + mean[4]
        q=Xb[:,5] * stdDev[5] + mean[5]

        #analytic delta
        eps=1e-12
        sqrtT=torch.sqrt(torch.clamp(T,min=eps))
        d1=(torch.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
        deltaTrue=torch.distributions.Normal(0,1).cdf(d1)*torch.exp(-q*T)

        #loss
        loss,lp,ld=bsLoss(model, Xb, yb, deltaTrue, stdDev, 0.1)
        loss.backward()
        optimizer.step()
        trainLosses.append(loss.item())

    trainLoss=float(np.mean(trainLosses))

    #validation loop (same as before)
    model.eval()
    valLosses=[]
    with torch.no_grad():
        for xb,yb in valLoader:
            pred=model(xb)
            valLosses.append(criterion(pred,yb).item())
    valLoss=float(np.mean(valLosses))

    print(f"Epoch {epoch}/{n_epochs}  Train Loss:{trainLoss:.6f}  Val Loss:{valLoss:.6f}")

    if valLoss < bestValLoss:
        bestValLoss=valLoss
        torch.save({'model_state_dict':model.state_dict(),'mean':mean,'stdDev':stdDev}, "bs_nn_best.pth")

#evaluate
model.eval()
with torch.no_grad():
    pred_test=model(XtestTensor).cpu().numpy().reshape(-1)
mse_test=np.mean((pred_test-yTest.reshape(-1)) ** 2)
mae_test=np.mean(np.abs(pred_test-yTest.reshape(-1)))
print(f"\nTest MSE:{mse_test:.6f}   Test MAE:{mae_test:.6f}")

#plot predicted vs true
plt.figure(figsize=(6,6))
plt.scatter(yTest.reshape(-1),pred_test,s=6)
plt.plot([yTest.min(),yTest.max()],[yTest.min(),yTest.max()])
plt.xlabel("Black-Scholes True Price")
plt.ylabel("NN Predicted Price")
plt.title("NN vs Black-Scholes (Test Set)")
plt.tight_layout()
plt.savefig("nn_vs_bs.png",dpi=150)
#plt.close()

#Delta comparison
subset=5000
Xsub=Xtest[:subset].copy()
XsubNorm=(Xsub-mean)/stdDev
XsubTensor=torch.tensor(XsubNorm,requires_grad=True).to(device)

#analytic delta (Torch)
S=torch.tensor(Xsub[:,0],dtype=torch.float32).to(device)
K=torch.tensor(Xsub[:,1],dtype=torch.float32).to(device)
T=torch.tensor(Xsub[:,2],dtype=torch.float32).to(device)
r=torch.tensor(Xsub[:,3],dtype=torch.float32).to(device)
sigma=torch.tensor(Xsub[:,4],dtype=torch.float32).to(device)
q=torch.tensor(Xsub[:,5],dtype=torch.float32).to(device)

_,d1_sub=bsCallPrice(S,K,T,r,sigma,q)
delta_analytic=torch.distributions.Normal(0,1).cdf(d1_sub)*torch.exp(-q*T)
delta_analytic=delta_analytic.cpu().numpy()

#NN delta via autograd
model.eval()
predSub=model(XsubTensor)
predSubSum=predSub.sum()
predSubSum.backward()
gradNormS=XsubTensor.grad.cpu().numpy()[:,0]
dprice_dS_nn=gradNormS/stdDev[0]

#scatter plot
plt.figure(figsize=(6,6))
plt.scatter(delta_analytic,dprice_dS_nn,s=6)
plt.plot([delta_analytic.min(),delta_analytic.max()],
         [delta_analytic.min(),delta_analytic.max()])
plt.xlabel("Analytic Delta")
plt.ylabel("NN Estimated Delta")
plt.title("Delta:Analytic vs NN (subset)")
plt.tight_layout()
plt.savefig("delta_vs_nn.png",dpi=150)
#plt.close()

#save test predictions to CSV
outDataframe=pd.DataFrame({
    "S":Xtest[:,0],
    "K":Xtest[:,1],
    "T":Xtest[:,2],
    "r":Xtest[:,3],
    "sigma":Xtest[:,4],
    "q" :Xtest[:,5],
    "BS price":yTest.reshape(-1),
    "NN price":pred_test,
    "BS delta":delta_test,
    "NN delta":np.concatenate([dprice_dS_nn,np.zeros(max(0,len(pred_test)-len(dprice_dS_nn)))])
})
outDataframe.to_csv("bs_nn_test_predictions.csv",index=False)

print("Saved:bs_nn_best.pth,nn_vs_bs.png,delta_vs_nn.png,bs_nn_test_predictions.csv")
