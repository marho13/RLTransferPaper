import torch
def MSE(x, y):
    return x-y

listFiles = []
def vidLoader():
    pass

def normal(epochs=45, batchSize=10):
    videos = [vidLoader(f) for f in listFiles]
    labels = []
    model = None
    for e in epochs:
        for num in len(videos)//batchSize:
            x = videos[batchSize*num: batchSize*(num+1)]
            y = labels[batchSize*num: batchSize*(num+1)]
            pred = model(x)
            loss = MSE(pred, y)

def new(epochs=45, batchSize=10):
    model = None
    for e in epochs:
        for num in len(listFiles) // batchSize:
            x = [vidLoader(f) for f in listFiles[batchSize * num: batchSize * (num + 1)]]
            y = listFiles[batchSize * num: batchSize * (num + 1)]
            pred = model(x)

            torch.no_grad()
            loss = MSE(pred, y)