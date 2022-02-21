import numpy as np
import matplotlib.pyplot as plt 

end = 100
loss_range =np.arange(-1, 3, 0.001)
train_x = np.linspace(1,end,end).reshape((-1, 1))
train_y = train_x**1.6

class createModel():
    def __init__(self):
        self.epochs = 6000
        self.learning_rate = 0.00001
        self.w = np.random.rand(1,1)*0.01
    
    def buildModel(self,x,y):
        loss_mem = []
        weight_hist = []
        for e in range(self.epochs):
            hypothesis = x**self.w
            error = hypothesis - y
            loss = np.mean(error*error)/2
            loss_mem.append(loss)
            gradient = np.mean((2*x**self.w * np.log(x)*(x**self.w - y)), axis=0, keepdims= True).T
            # gradient = np.mean(x * error, axis=0, keepdims=True).T
            self.w -= self.learning_rate * gradient
            weight_hist.append(self.w.flatten())
        return (loss_mem, weight_hist)

    def getLoss(self, x, y):
        loss_hist = []
        for w in loss_range:
            hypothesis = x**w
            error = hypothesis - y
            loss = np.mean(error*error)/2
            loss_hist.append(loss)
        return loss_hist
    
    # def predictModel(self,x,y):
    #     hypothesis = x**self.w
    #     error = hypothesis - y
    #     mse = np.mean(error*error)
    #     return np.sqrt(mse)


model = createModel()
loss_mem, weight_hist = model.buildModel(train_x, train_y)
loss_hist = model.getLoss(train_x, train_y)
x_epoch = list(range(len(loss_mem)))
loss_length = list(range(len(loss_hist)))

print(weight_hist[-1])
plt.plot(loss_range, loss_hist)
# plt.plot(x_epoch, weight_hist)
plt.show()