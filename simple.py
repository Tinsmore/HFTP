class Agent:

    def __init__(self):
        self.lastData = [0,0,0,0,0,0] # 0-trade price 1-trade volume 2-bidprice 3-bidvolume 4-askprice 5-askvolume
        self.data = [0,0,0,0,0,0]
        self.step = 0.0001
        print("initialize...")

    def forward(self,newData):
        self.lastData = self.data
        self.data = newData

    def predict(self):
        predictPrice = self.data[0] + self.step*(self.data[3] - self.data[5])

        return predictPrice

    def tune(self,newData):
        predict = self.predict()
        if(predict - newData[0] < 0.01 and newData[0] - predict < 0.01):
            print("step not changed")
            return

        newStep = (newData[0] - self.data[0])/(self.data[3] - self.data[5])
        if newStep > 0:
            self.step = newStep
        print("new step: ",self.step)


def main():
    ag = Agent()
    ag.forward([1.1,1000,1.0,2000,1.2,1000])
    print(ag.predict())

    ag.tune([1.13,2000,1.1,2000,1.3,1200])
    print(ag.predict())


main()