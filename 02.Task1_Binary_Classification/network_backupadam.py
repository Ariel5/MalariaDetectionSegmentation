import torch
import torchvision


def make_network(config, num_outputs):
    if config.model.network == 'ResNet18':
        pretrained_model = torchvision.models.resnet18(pretrained=True)
        model = FineTuneResNet(num_outputs, pretrained_model)

    elif config.model.network == 'ResNet50':
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        model = FineTuneResNet(num_outputs, pretrained_model)


    elif config.model.network == 'ResNet18Threshold':
        pretrained_model = torchvision.models.resnet18(pretrained=True)
        model = FineTuneResNetOrdinalThreshold(num_outputs, pretrained_model)
    elif config.model.network == 'ResNet50Threshold':
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        model = FineTuneResNetOrdinalThreshold(num_outputs, pretrained_model)
    else:
        raise('Unknown Network Type')

    return model




class FineTuneResNet(torch.nn.Module):
    def __init__(self, num_outputs, pretrained_model):
        super(FineTuneResNet, self).__init__()
        self.pretrained_model = pretrained_model
        fc_dim = self.pretrained_model.fc.weight.shape[1]
        # turn the last fc layer into an identity
        self.pretrained_model.fc = torch.nn.Identity()
        self.fc = torch.nn.Linear(fc_dim, num_outputs)

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc(x)
        return x


class FineTuneResNetOrdinalThreshold(torch.nn.Module):
    """
    Creates a single output model. If there are K classes, then the num_outputs
    argument should be K-1 (K-1 thresholds needed to split a real line into
    K sections)
    """
    def __init__(self, num_outputs, pretrained_model):
        super(FineTuneResNetOrdinalThreshold, self).__init__()
        self.pretrained_model = pretrained_model
        fc_dim = self.pretrained_model.fc.weight.shape[1]
        # turn the last fc layer into an identity
        self.pretrained_model.fc = torch.nn.Identity()
        self.fc = torch.nn.Linear(fc_dim, 1, bias = False)
        # create the threshold scalar values
        cutpoints = torch.rand(num_outputs).sort()[0]
        # set them as learnable parameters
        self.cutpoints = torch.nn.Parameter(cutpoints)

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc(x)
        # turn single output into num_outputs, by subtracting the thresholds
        x = x - self.cutpoints
        return x


# create a wrapper to the model, to allow it to accept and output dicts
class WrappedNetwork(torch.nn.Module):
    def __init__(self, config, num_outputs=None, im_key='im'):
        super().__init__()
        # must define the model inside the wrapper, otherwise we'll get GPU device
        # conflicts
        if num_outputs is None:
            num_outputs = config.data.num_outputs
        self._model = make_network(config, num_outputs)
        self.im_key = im_key

    def forward(self, data_dict):
        # forward just makes sure we return a dict
        data_dict['output'] = self._model(data_dict[self.im_key])
        return data_dict
