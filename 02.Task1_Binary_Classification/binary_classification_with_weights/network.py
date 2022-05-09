import torch
import torchvision

class Network(torch.nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        # self.conv_base = torchvision.models.resnet18(pretrained=True)

        # if config.model.category in ['vanilla', 'multi_class']:
        #     self.FC = torch.nn.Sequential(
        #         Flatten(),
        #         torch.nn.Linear(512, config.data.num_outputs),
        #         )
        # elif config.model.category=='frank_hall':
        #     self.FC = torch.nn.Sequential(
        #         Flatten(),
        #         torch.nn.Linear(512, len(config.data.thresholds)),
        #         )

        # self.conv_base = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.conv_base = torchvision.models.densenet121(pretrained=True)
        self.config = config

        self.FC = torch.nn.Linear(1024, len(config.data.thresholds))

    def forward(self, x):
        # x = self.conv_base.conv1(x)
        # x = self.conv_base.bn1(x)
        # x = self.conv_base.relu(x)
        # x = self.conv_base.maxpool(x)
        # x = self.conv_base.layer1(x)
        # x = self.conv_base.layer2(x)
        # x = self.conv_base.layer3(x)
        # x = self.conv_base.layer4(x)
        # x = self.conv_base.avgpool(x)

        # x = self.FC(x)

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

