
from torch import nn, optim
from numpy import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from distribute_data import Distribute_Youtube
from sklearn import metrics
from splitnn_net1 import SplitNN1
import torch.nn.functional as F

alpha=0.08
input_size = [160, 160]
hidden_sizes = {"client_1": [128], "client_2": [128], "server": [128, 128]}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

watch_vec_size = 64
search_vec_size = 64
other_feat_size = 32
#label_path = "/Users/lizhenyu/PycharmProjects/YoutubeDNN/label.csv"
user_watch=np.load('user_watch.npy')
user_search=np.load('user_search.npy')
user_feat=np.load('user_feat.npy')
user_labels=np.load('user_labels.npy')
inputs = np.hstack((user_watch, user_search, user_feat))
x_data = torch.FloatTensor(inputs).to((device))
y_data = torch.FloatTensor(user_labels).to(device)

target_feat_idx = 21 # is_male

deal_dataset = TensorDataset(x_data, y_data)
target_data_size = int(0.5 * len(deal_dataset))
shadow_data_size = len(deal_dataset) - target_data_size

target_train_size = int(0.8 * target_data_size)
target_test_size = target_data_size - target_train_size

shadow_train_size = int(0.8 * shadow_data_size)
shadow_test_size = shadow_data_size - shadow_train_size

data_owners = ['client_1', 'client_2']

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


def train_acc(network, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs = network.forward(data_ptr)
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







def data_test_acc(network, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs = network.forward(data_ptr)
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



def train_model(network, epoch, dataset, train_size, test_size):
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
        network.train()
        for images, labels in distributed_trainloader:
            loss = train(images, labels, network)
            running_loss += loss
        print("Epoch {} - Training loss:{}".format(i, running_loss / len(train_loader)))
        train_acc(network, distributed_trainloader)
        temp=data_test_acc(network, distributed_testloader)
        res.append(temp)
    return max(res)

def train_splitnn(dataset, train_size, test_size):
    res = []
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
    data_owners = ['client_1', 'client_2']

    for location in model_locations:
        for param in models[location]:
            param.to(device)

    optimizers = [optim.Adam(models[location].parameters(), lr=0.01, ) for location in model_locations]
    splitnn = SplitNN1(models, optimizers, data_owners).to(device)
    #temp=train_model(30,deal_dataset)
    temp = train_model(splitnn, 1, dataset, train_size, test_size)
    res.append(temp)

    print("end")
    sum = 0
    for i in res:
        sum += i
    print("Accuracy {:.2f}%".format(sum / len(res)))

    return splitnn

def train_classifier(epoch, network, dataset, train_size, test_size):

    #define a classifier
    hidden_size = 256
    classifier = nn.Sequential(
                nn.Linear(128 + 1, hidden_size), # 1 dim for label
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 2),
                nn.Sigmoid())

    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
    distributed_trainloader = Distribute_Youtube(data_owners=data_owners, data_loader=train_loader)
    distributed_testloader = Distribute_Youtube(data_owners=data_owners, data_loader=test_loader)

    # train classifier
    adam = optim.Adam(classifier.parameters(), lr=0.01, )
    for i in range(epoch):
        classifier.train()
        step = 0
        for images, labels in distributed_trainloader:
            # use client_2's output and label as classifier's feature
            predict = network.predict_client2(images)
            client2_feat = torch.cat([predict, labels], dim=1)

            out = classifier(client2_feat)
            # use target feature as classifier's label
            target_feat = images["client_2"][:, target_feat_idx]

            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(out, nn.functional.one_hot(target_feat.type(torch.int64), 2))
            adam.zero_grad()
            loss.backward()
            adam.step()

            if step % 100 == 0:
                print("loss for step {} training classifier is {}".format(step, loss.detach().item()))
            step += 1

    return classifier


def attack_eval(target_splitnn, target_dataset, classifier):
    data_loader = DataLoader(dataset=target_dataset, batch_size=32, shuffle=True)
    actual = None
    expect = None

    for image, labels in data_loader:
        predict = target_splitnn.predict_client2(image)
        client2_feat = torch.cat([predict, labels], dim=1)
        infer_out = classifier(client2_feat)
        actual = torch.cat([actual, infer_out], dim=0)

        target_feat = image["client_2"][:, target_feat_idx]
        expect = torch.cat([expect, target_feat], dim=0)
    # calc acc
    correct = expect.numel() - (expect - actual).count_non_zero()
    acc = float(correct) / expect.numel()

    print("acc is {}".format(acc))




if __name__=="__main__":
    #'''
    #torch.seed() = 1024

    # split dataset to target adn shadow dataset
    target_dataset, shadow_dataset = torch.utils.data.random_split(deal_dataset, [target_data_size, shadow_data_size])

    # train target model
    target_splitnn = train_splitnn(target_dataset, target_train_size, target_test_size)

    # train shadow model
    shadow_splitnn = train_splitnn(shadow_dataset, shadow_train_size, shadow_test_size)

    # train classifier
    classifier = train_classifier(10, shadow_splitnn, shadow_dataset, shadow_train_size, shadow_test_size)

    # evaluate attack
    attack_eval(target_splitnn, target_dataset, classifier)

