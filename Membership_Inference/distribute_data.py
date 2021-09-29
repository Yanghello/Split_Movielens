import numpy
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Distribute_Youtube:
    def __init__(self, data_owners, data_loader):
        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)
        self.data_pointer = []
        self.labels = []
        for image, label in self.data_loader:
            curr_data_dict = {}
            self.labels.append(label)
            height=image.shape[0]
            padding_client1=torch.from_numpy(numpy.zeros((height,96))).to(device)
            padding_client2=torch.from_numpy(numpy.zeros((height,64))).to(device)
            curr_data_dict[data_owners[0]] = torch.cat((image[:, 0:64].to(device), padding_client1),1).float().to(device)
            curr_data_dict[data_owners[1]] = torch.cat((padding_client2, image[:, 64:].to(device)),1).float().to(device)
            self.data_pointer.append(curr_data_dict)

    def __iter__(self):
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield (data_ptr, label)

    def __len__(self):
        return len(self.data_loader) - 1