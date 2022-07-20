import torch
from utils import loadImage, getNameFileFromPath

class TransformerModel(torch.nn.Module):
    def __init__(self, _transformer, _tokenizer, opt='adam', lr=5e-5, decay=2e-5, minlr=1e-5, lr_strategy="simple", max_len=50,  cuda="cuda:0"):
        super(TransformerModel, self).__init__()
        self.lr = lr
        self.decay = decay
        self.opt = opt
        self.minlr = minlr
        self.lrStrategy=lr_strategy

        dim= lambda x: x['dim'] if "dim" in x else (x['hidden_size'] if "hidden_size" in x else 768)
        self.encoder = _transformer
        self.tokenizer = _tokenizer

        hiddensize = dim(_transformer.config.to_dict())
        self.dense1 = torch.nn.Sequential(torch.nn.BatchNorm1d(hiddensize),torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())
        self.dense2 = torch.nn.Linear(in_features=64, out_features=32)
        self.drop = torch.nn.Dropout(p=0.25)
        self.classifier = torch.nn.Linear(in_features=32, out_features=2)

        self.dev = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")
        self.max_length = max_len
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.to(device=self.dev)
        self.best_measure = None
        self.best_model_name = None

    def forward(self, x):        
        x = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.dev)
        x = self.encoder(**x)[0][:, 0]
        x = self.dense1(x)
        x = torch.nn.functional.relu(self.dense2(x))
        x = self.classifier(x)
        return x

    def predict_step(self, batch):
            x = batch
            y_hat = self.forward(x)
            preds = torch.max(y_hat, 1).indices
            del x
            del y_hat
            return preds
  
    def configure_optimizers(self):
        print("Configure Optimizers")
        if self.lrStrategy=="dynamic":
            return self.configure_optimizers_dynamic1()
        return self.configure_optimizers_simple()

    def configure_optimizers_simple(self):
        print("Configured Simple {} with lr: {}".format(self.opt, self.lr ))
        if self.opt=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        elif self.opt=='rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.decay)

    def configure_optimizers_dynamic1(self, layers=12):
        print("Configured Dynamic {} with starting lr: {} and Exponential Increase".format(self.opt, self.lr))
        layernum, params, i = layers,[], 0
        diff =abs(self.lr -self.minlr)
        for l in self.encoder.encoder.layer:
            params.append({'params':l.parameters(), 'lr': self.minlr+diff**(layernum)}) 
            layernum-=1
        try:
            params.append({'params':self.encoder.pooler.parameters(), 'lr':self.lr})
        except:
            print('Warning: No Pooler layer found')
        params.append({'params': self.dense1.parameters()})
        params.append({'params': self.dense2.parameters()})
        params.append({'params': self.classifier.parameters()})
        if self.opt == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.decay)
        elif self.opt == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.lr, weight_decay=self.decay)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.dev))

    def save(self, path):
        torch.save(self.state_dict(), path)
# --------------------------------------------------------------------------
class ImageTransformerModel(torch.nn.Module):
    def __init__(self, _itransformer, _itokenizer, opt='adam', lr=5e-5, decay=2e-5, minlr=1e-5, lr_strategy="simple", max_len=50, cuda="cuda:0", imageInventory=None):
        super(ImageTransformerModel, self).__init__()
        self.lr = lr
        self.decay = decay
        self.opt = opt
        self.minlr = minlr
        self.lrStrategy=lr_strategy
        # not the best model...
        dim= lambda x: x['dim'] if "dim" in x else (x['hidden_size'] if "hidden_size" in x else 768)
        self.iencoder=_itransformer
        self.itokenizer = _itokenizer
        hiddensize = dim(_itransformer.config.to_dict())
        self.dense1Image = torch.nn.Sequential(torch.nn.BatchNorm1d(hiddensize),torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())
        self.dense2 = torch.nn.Linear(in_features=64, out_features=32)
        self.drop = torch.nn.Dropout(p=0.25)
        self.classifier = torch.nn.Linear(in_features=32, out_features=2)
        self.dev = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")
        self.max_length = max_len
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.to(device=self.dev)
        self.best_measure = None
        self.best_model_name = None
        self.imageInventory=imageInventory

    def forward(self,p):
        images=[]
        for path in p:
            ima=None
            name=getNameFileFromPath(path)
            if  self.imageInventory!=None and (name in self.imageInventory):
                ima=self.imageInventory[name]
            else: ima=loadImage(path)
            images.append(ima)
        imagesFea = self.itokenizer(images=images, return_tensors="pt")
        pixel_values = imagesFea['pixel_values'].to(device=self.dev)
        x2=self.iencoder(pixel_values)
        x2=x2[0][:, 0, :]
        x2=self.dense1Image(x2)
        x3 = torch.nn.functional.relu(self.dense2(x2))
        x3 = self.classifier(x3)
        return x3

    def predict_step(self, p):
        y_hat = self.forward(p)
        preds = torch.max(y_hat, 1).indices
        del y_hat
        return preds

    def configure_optimizers(self):
        print("Configure Optimizers")
        if self.lrStrategy=="dynamic":
            return self.configure_optimizers_dynamic1()
        return  self.configure_optimizers_simple()

    def configure_optimizers_simple(self):
        print("Configured Simple {} with lr: {}".format(self.opt, self.lr ))
        if self.opt=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        elif self.opt=='rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.decay)

    def configure_optimizers_dynamic1(self, layers=12):
        print("Configured Dynamic {} with starting lr: {} and Exponential Increase".format(self.opt, self.lr))
        params ,i,  layernum, diff = [], 0, layers, abs(self.lr -self.minlr)
        for l in self.iencoder.encoder.layer:
            params.append({'params':l.parameters(), 'lr': self.minlr+diff**(layernum)})
            layernum-=1
        try:
            params.append({'params':self.iencoder.pooler.parameters(), 'lr':self.lr})
        except: print('Warning: No Pooler layer found')
        params.append({'params': self.dense1Image.parameters()})
        params.append({'params': self.dense2.parameters()})
        params.append({'params': self.classifier.parameters()})

        if self.opt == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.decay)
        elif self.opt == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.lr, weight_decay=self.decay)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.dev))

    def save(self, path):
        torch.save(self.state_dict(), path)

class DualTransformerModel(torch.nn.Module):
    def __init__(self, _transformer, _itransformer, _tokenizer, _itokenizer, opt='adam', lr=5e-5, decay=2e-5, minlr=1e-5, lr_strategy="simple", max_len=50, cuda="cuda:0",  imageInventory=None):
        super(DualTransformerModel, self).__init__()
        self.lr = lr
        self.decay = decay
        self.opt = opt
        self.minlr = minlr
        self.lrStrategy=lr_strategy
        dim= lambda x: x['dim'] if "dim" in x else (x['hidden_size'] if "hidden_size" in x else 768)
        self.encoder = _transformer
        self.iencoder=_itransformer
        self.tokenizer = _tokenizer
        self.itokenizer = _itokenizer
        hiddensize = dim(_transformer.config.to_dict())
        self.dense1Text = torch.nn.Sequential(torch.nn.BatchNorm1d(hiddensize),torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())
        self.dense1Image = torch.nn.Sequential(torch.nn.BatchNorm1d(hiddensize),torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())
        self.dense2 = torch.nn.Linear(in_features=2*64, out_features=32)
        self.drop = torch.nn.Dropout(p=0.25)
        self.classifier = torch.nn.Linear(in_features=32, out_features=2)
        self.dev = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")
        self.max_length = max_len
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.to(device=self.dev)
        self.best_measure = None
        self.best_model_name = None
        self.imageInventory=imageInventory

    def forward(self, x,p):
        x = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.dev)
        x1 = self.encoder(**x)[0][:, 0]
        x1 = self.dense1Text(x1)
        #PROCESS THE IMAGES
        images = []
        for path in p:
            ima = None
            name = getNameFileFromPath(path)
            if  self.imageInventory!=None and (name in self.imageInventory):
                ima = self.imageInventory[name]
            else:
                ima = loadImage(path)
            images.append(ima)
        imagesFea = self.itokenizer(images=images, return_tensors="pt")
        pixel_values = imagesFea['pixel_values'].to(device=self.dev)
        x2=self.iencoder(pixel_values)
        x2=x2[0][:, 0, :]
        x2=self.dense1Image(x2)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = torch.nn.functional.relu(self.dense2(x3))
        x3 = self.classifier(x3)
        return x3

    def predict_step(self, x, p):
        y_hat = self.forward(x, p)
        preds = torch.max(y_hat, 1).indices
        del x
        del y_hat
        return preds

    def configure_optimizers(self):
        print("Configure Optimizers")
        if self.lrStrategy=="dynamic":
            return self.configure_optimizers_dynamic1()
        return self.configure_optimizers_simple()

    def configure_optimizers_simple(self):
        print("Configured Simple {} with lr: {}".format(self.opt, self.lr ))
        if self.opt=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        elif self.opt=='rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.decay)

    def configure_optimizers_dynamic1(self, layers=12):
        print("Configured Dynamic {} with starting lr: {} and Exponential Increase".format(self.opt, self.lr))
        params, i, layernum = [], 0 , layers
        diff =abs(self.lr -self.minlr)
        for l in self.iencoder.encoder.layer:
            params.append({'params':l.parameters(), 'lr': self.minlr+diff**(layernum)})
            layernum-=1
        try:
            params.append({'params':self.iencoder.pooler.parameters(), 'lr':self.lr})
        except:
            print('Warning: No Pooler layer found')
        params.append({'params': self.dense1Text.parameters()})
        params.append({'params': self.dense1Image.parameters()})
        params.append({'params': self.dense2.parameters()})
        params.append({'params': self.classifier.parameters()})

        if self.opt == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.decay)
        elif self.opt == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.lr, weight_decay=self.decay)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.dev))

    def save(self, path):
        torch.save(self.state_dict(), path)
# --------------------------------------------------------------------------
class DualTransformerModel2Text(torch.nn.Module):
    def __init__(self, _transformer, _itransformer, _tokenizer, _itokenizer, opt='adam', lr=5e-5, decay=2e-5, minlr=1e-5, lr_strategy="simple", max_len=50,  cuda="cuda:0", imageInventory=None):
        super(DualTransformerModel2Text, self).__init__()
        self.lr = lr
        self.decay = decay
        self.opt = opt
        self.minlr = minlr
        self.lrStrategy=lr_strategy
        # not the best model...
        dim= lambda x: x['dim'] if "dim" in x else (x['hidden_size'] if "hidden_size" in x else 768)
        self.encoder = _transformer
        self.iencoder=_itransformer
        self.tokenizer = _tokenizer
        self.itokenizer = _itokenizer
        hiddensize = dim(_transformer.config.to_dict())
        
        self.dense1Text = torch.nn.Sequential(torch.nn.BatchNorm1d(hiddensize),torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())
        self.dense1TextOCR = torch.nn.Sequential(torch.nn.BatchNorm1d(hiddensize),torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())
        self.dense1Image = torch.nn.Sequential(torch.nn.BatchNorm1d(hiddensize),torch.nn.Linear(in_features=hiddensize, out_features=64), torch.nn.LeakyReLU())

        self.dense2= torch.nn.Sequential(torch.nn.Linear(in_features=3*64, out_features=32), torch.nn.ReLU())
        self.drop = torch.nn.Dropout(p=0.25)
        self.classifier = torch.nn.Linear(in_features=32, out_features=2)

        self.dev = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")
        self.max_length = max_len
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.to(device=self.dev)
        self.best_measure = None
        self.best_model_name = None
        self.imageInventory=imageInventory

    def forward(self, x,z,p):
        x = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.dev)
        x1 = self.encoder(**x)[0][:, 0]
        x1 = self.dense1Text(x1)
        z = self.tokenizer(z, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.dev)
        z1 = self.encoder(**z)[0][:, 0]
        z1 = self.dense1TextOCR(z1)
        #PROCESSING THE IMAGES
        images = []
        for path in p:
            ima = None
            name = getNameFileFromPath(path)
            if  self.imageInventory!=None and (name in self.imageInventory):
                ima = self.imageInventory[name]
            else:
                ima = loadImage(path)
            images.append(ima)
        imagesFea = self.itokenizer(images=images, return_tensors="pt")
        pixel_values = imagesFea['pixel_values'].to(device=self.dev)
        x2=self.iencoder(pixel_values)
        x2=x2[0][:, 0, :]
        x2=self.dense1Image(x2)
        x3 = torch.cat((x1, z1, x2), dim=1)
        x3 = self.dense2(x3)
        x3 = self.classifier(x3)
        return x3

    def predict_step(self, x, z, p):
        y_hat = self.forward(x, z, p)
        preds = torch.max(y_hat, 1).indices
        del x
        del z
        del p
        del y_hat
        return preds

    def configure_optimizers(self):
        print("Configure Optimizers")
        if self.lrStrategy=="dynamic":
            return self.configure_optimizers_dynamic1()
        return self.configure_optimizers_simple()

    def configure_optimizers_simple(self):
        print("Configured Simple {} with lr: {}".format(self.opt, self.lr ))
        if self.opt=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        elif self.opt=='rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.decay)

    def configure_optimizers_dynamic1(self, layers=12):
        print("Configured Dynamic {} with starting lr: {} and Exponential Increase".format(self.opt, self.lr))
        params, i, layernum = [], 0, layers
        diff =abs(self.lr -self.minlr)
        for l in self.iencoder.encoder.layer:
            params.append({'params':l.parameters(), 'lr': self.minlr+diff**(layernum)})
            layernum-=1
        try:
            params.append({'params':self.iencoder.pooler.parameters(), 'lr':self.lr})
        except:
            print('Warning: No Pooler layer found')
        params.append({'params': self.dense1Text.parameters()})
        params.append({'params': self.dense1TextOCR.parameters()})
        params.append({'params': self.dense1Image.parameters()})
        params.append({'params': self.dense2.parameters()})
        params.append({'params': self.classifier.parameters()})

        if self.opt == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.decay)
        elif self.opt == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.lr, weight_decay=self.decay)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.dev))

    def save(self, path):
        torch.save(self.state_dict(), path)

