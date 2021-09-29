
from torch import nn, optim
from numpy import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from split_movielens_padding.distribute_data import Distribute_Youtube
from sklearn import metrics
from splitnn_net import SplitNN
import torch.nn.functional as F

alpha=0.08
input_size = [160, 160]
hidden_sizes = {"client_1": [128], "client_2": [128], "server": [128, 128]}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

watch_vec_size = 64
search_vec_size = 64
other_feat_size = 32
label_path = "/Users/lizhenyu/PycharmProjects/YoutubeDNN/label.csv"
user_watch=np.load('user_watch.npy')
user_search=np.load('user_search.npy')
user_feat=np.load('user_feat.npy')
user_labels=np.load('user_labels.npy')
inputs = np.hstack((user_watch, user_search, user_feat))
x_data = torch.FloatTensor(inputs).to((device))
y_data = torch.FloatTensor(user_labels).to(device)

deal_dataset = TensorDataset(x_data, y_data)
train_size = int(0.91 * len(deal_dataset))
test_size = len(deal_dataset) - train_size



def train(x, target, splitNN):
    splitNN.zero_grads()
    target.to(device)
    pred = splitNN.forward(x).to(device)
    criterion = nn.CrossEntropyLoss()
    temp = target.reshape(-1, pred.shape[0])[0].long().to(device)
    loss = criterion(pred, temp)
    loss.backward()
    splitNN.step()
    return loss.detach().item()


def train_acc(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs = splitnn.forward(data_ptr)
            value, indicies = torch.topk(outputs, 10, dim=1)
            total += label.size(0)
            correct += topK(label, indicies)
    print("Accuracy {:.2f}%".format(100 * correct / total))


def topK(labels, indicy):
    upper = labels.size(0)
    labels = labels.cpu().numpy()
    indicy = indicy.cpu().numpy()
    hit = 0
    for i in range(upper):
        for h in range(10):
            if indicy[i][h] == labels[i][0]:
                hit += 1
                break
    return hit


def print_parameters(model):
    res = {}
    for name, param in models["server"].named_parameters():
        res[name] = param.data.numpy()
        grad_key = name+"_grad"
        res[grad_key] = param.grad.data.numpy()







def data_test_acc(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs = splitnn.forward(data_ptr)
            value, indicies = torch.topk(outputs, 10, dim=1)
            total += label.size(0)
            correct += topK(label, indicies)
    print("Accuracy {:.2f}%".format(100 * correct / total))
    return (100*correct/total)



# def cal_auc_paddle(dataloader):
#     auc=paddle.metric.Auc()
#     label_list=[]
#     acc=0.0
#     auc_res=0.0
#     with torch.no_grad():
#         for data_ptr, label in dataloader:
#             labels=[]
#             outputs=splitnn.cal_auc_forward(data_ptr)
#             # for output in outputs.tolist():
#             #     score.append(output)
#             for i in label.tolist():
#                 temp = []
#                 for j in range(3952):
#                     temp.append(0)
#                 temp[int(i[0])]=1
#                 labels.append(temp)
#             pred_scores = outputs.numpy()
#
#             diff = np.abs(pred_scores-np.array(labels))
#             diff[diff>0.5]=1
#             acc=1-np.mean(diff)
#             class0_preds=1-pred_scores
#             class1_preds=pred_scores
#             preds=np.concatenate((class0_preds, class1_preds),axis=1)
#             auc.update(preds=preds, labels=np.array(label))
#             auc_res=auc.accumulate()
#             print(auc_res)




def cal_auc(dataloader):
    score=[]
    labels=[]
    label_res=[]
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs=splitnn.forward(data_ptr)
            for output in outputs.tolist():
                score.append(output)
            for i in label.tolist():
                label_res.append(int(i[0]))
                temp = []
                for j in range(3952):
                    temp.append(0)
                temp[int(i[0])]=1
                labels.append(temp)
    score=np.array(score)
    labels=np.array(labels)
    print(metrics.roc_auc_score(labels, score, average='micro'))
    return



def train_model(epoch, dataset):
    res=[]
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
    distributed_trainloader = Distribute_Youtube(data_owners=data_owners, data_loader=train_loader)
    distributed_testloader = Distribute_Youtube(data_owners=data_owners, data_loader=test_loader)
    n_count=0
    #file = open('importance.txt', "a+")
    for i in range(epoch):
        running_loss = 0
        splitnn.train()
        for images, labels in distributed_trainloader:
            loss = train(images, labels, splitnn)
            running_loss += loss
        print("Epoch {} - Training loss:{}".format(i, running_loss / len(train_loader)))
        train_acc(distributed_trainloader)
        temp=data_test_acc(distributed_testloader)
        res.append(temp)
    return max(res)

if __name__=="__main__":
    #'''
    res = []
    for j in range(30):
        models = {
            "client_1": nn.Sequential(
                nn.Linear(input_size[0], 128),
                nn.ReLU(),
            ),
            "client_2": nn.Sequential(
                nn.Linear(input_size[1], 128),
                nn.ReLU(),
            ),
            "server": nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 3952),
                nn.LogSoftmax(dim=1)
            )
        }
        model_locations = ['client_1', 'client_2', 'server']
        for location in model_locations:
            models[location].to(device)
        deal_dataset = TensorDataset(x_data, y_data)
        data_owners = ['client_1', 'client_2']

        for location in model_locations:
            for param in models[location]:
                param.to(device)
        optimizers = [optim.Adam(models[location].parameters(), lr=0.01, ) for location in model_locations]
        splitnn = SplitNN(models, optimizers, data_owners).to(device)
        #temp=train_model(30,deal_dataset)
        temp = train_model(15, deal_dataset)
        res.append(temp)
    print("end")
    sum = 0
    for i in res:
        sum += i
    print("Accuracy {:.2f}%".format(sum / len(res)))