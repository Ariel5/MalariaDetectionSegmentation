import torch
import torchvision
import math

class Network(torch.nn.Module):
    def __init__(self, config, doctor_list, initialization=None, latent_space_size=256):
        super(Network, self).__init__()
        self.conv_base = torchvision.models.resnet18(pretrained=True)

        if config.model.category in ['vanilla', 'multi_class']:
            self.FC = torch.nn.Sequential(
                Flatten(),
                torch.nn.Linear(512 +latent_space_size, config.data.num_outputs),
                )
        elif config.model.category=='frank_hall':
            self.FC = torch.nn.Sequential(
                Flatten(),
                torch.nn.Linear(512 +latent_space_size, len(config.data.thresholds)),
                )
        self.config = config
        self.doctor_list = doctor_list

        self.embedding = torch.nn.Embedding(100, latent_space_size) # 100 in doctor_list, 256 dimensional embeddings

        if initialization=='normal':
            # initalize with normal distribution, mean=0, std=1
            # torch.nn.init.normal_(self.embedding.weight, mean=0, std=1)
            torch.nn.init.normal_(self.embedding.weight, mean=0, std=1/math.sqrt(latent_space_size))
        elif initialization=='uniform':
            # initalize with uniform distribution, [0,1]
            torch.nn.init.uniform_(self.embedding.weight, a=0, b=1)


    def forward(self, x, docs):
        x = self.conv_base.conv1(x)
        x = self.conv_base.bn1(x)
        x = self.conv_base.relu(x)
        x = self.conv_base.maxpool(x)
        x = self.conv_base.layer1(x)
        x = self.conv_base.layer2(x)
        x = self.conv_base.layer3(x)
        x = self.conv_base.layer4(x)
        x = self.conv_base.avgpool(x)

        lookup_tensor = []
        for doc in docs:
            lookup_tensor.append(self.doctor_list.index(doc))
        lookup_tensor = torch.tensor(lookup_tensor, dtype=torch.long).cuda()
        doc_embedding = self.embedding(lookup_tensor).unsqueeze(-1).unsqueeze(-1)

        x = torch.cat((x, doc_embedding), 1)
        x = self.FC(x)
        
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

# create a wrapper to the model, to allow it to accept and output dicts
class WrappedNetwork(torch.nn.Module):
    def __init__(self, config, im_key='im', doc_key='ReportDr', doctor_list=None, initialization=None, latent_space_size=256):
        super().__init__()
        # must define the model inside the wrapper, otherwise we'll get GPU device
        # conflicts
        self._model = Network(config, doctor_list, initialization, latent_space_size)
        self.im_key = im_key

        self.doc_key = doc_key

    def forward(self, data_dict):
        # forward just makes sure we return a dict
        data_dict['output'] = self._model(data_dict[self.im_key], data_dict[self.doc_key])
        return data_dict

