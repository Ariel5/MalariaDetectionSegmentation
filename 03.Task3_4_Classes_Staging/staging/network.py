import torch
import torchvision

class Network(torch.nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        
        self.conv_base = torchvision.models.densenet121(pretrained=True)
        self.config = config

        self.FC = torch.nn.Linear(1024, 5)

    def forward(self, x):
        
        x = self.conv_base.features(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.FC(x)

        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

# create a wrapper to the model, to allow it to accept and output dicts
class WrappedNetwork(torch.nn.Module):
    def __init__(self, config, im_key='im'):
        super().__init__()
        # must define the model inside the wrapper, otherwise we'll get GPU device
        # conflicts
        self._model = Network(config)
        self.im_key = im_key

    def forward(self, data_dict):
        # forward just makes sure we return a dict
        data_dict['output'] = self._model(data_dict[self.im_key])
        return data_dict

