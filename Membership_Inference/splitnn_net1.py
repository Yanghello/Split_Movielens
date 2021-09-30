import torch,random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SplitNN1(torch.nn.Module):
    def __init__(self, models, optimizers, data_owner):
        self.data_owners = data_owner
        self.optimizers = optimizers
        self.models = models
        self.activation=[]
        self.count=[0,0,0,0]
        self.print=False
        super().__init__()


    def forward(self, data_pointer):
        client_output = {}
        remote_outputs = []
        i = 0
        for owner in self.data_owners:
            if i == 0:
                self.models[owner].to(device)
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 160]).to(device))
                #'''
                client_output[owner].to(device)

                remote_outputs.append(client_output[owner].requires_grad_().to(device))
                #print(client_output[owner].requires_grad)
                i += 1
            else:
                self.models[owner].to(device)
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 160]).to(device))
                client_output[owner].to(device)
                remote_outputs.append(client_output[owner].requires_grad_())
        #server_input = torch.min(remote_outputs[0], remote_outputs[1])
        #server_input = torch.cat(remote_outputs,1)
        #server_input = torch.max(remote_outputs[0],remote_outputs[1])
        #server_input = torch.add(remote_outputs[0],remote_outputs[1])
        server_input = torch.add(remote_outputs[0]/2, remote_outputs[1]/2)
        #server_input=torch.mul(remote_outputs[0], remote_outputs[1])
        self.models["server"].to(device)
        server_output = self.models["server"](server_input)
        return server_output

    def test_forward(self, data_pointer):
        client_output = {}
        remote_outputs = []
        i = 0
        for owner in self.data_owners:
            if i == 0:
                self.models[owner].to(device)
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 160]).to(device))
                # '''
                client_output[owner].to(device)
                remote_outputs.append(client_output[owner].requires_grad_().to(device))
                # print(client_output[owner].requires_grad)
                i += 1
            else:
                self.models[owner].to(device)
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 160]).to(device))
                client_output[owner].to(device)
                remote_outputs.append(client_output[owner].requires_grad_())
        #server_input = torch.min(remote_outputs[0], remote_outputs[1])
        # server_input = torch.cat(remote_outputs,1)
        # server_input = torch.max(remote_outputs[0],remote_outputs[1])
        server_input = torch.add(remote_outputs[0],remote_outputs[1])
        # server_input = torch.add(remote_outputs[0]/2, remote_outputs[1]/2)
        self.models["server"].to(device)
        server_output = self.models["server"](server_input)
        return server_output


    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def train(self):
        for loc in self.models.keys():
            for i in range(len(self.models[loc])):
                self.models[loc][i].train()

    def eval(self):
        for loc in self.models.keys():
            for i in range(len(self.models[loc])):
                self.models[loc][i].eval()
    
    def predict_client2(self, data_pointer):
        return self.models["client_2"](data_pointer["client_2"].reshape([-1, 160]).to(device))

    def number_of_certain_probability(self,sequence, probability):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item