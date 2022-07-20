import argparse
from sklearn.utils import shuffle
import pandas as pd, numpy as np
import config as conf

from pytorch_lightning import seed_everything
from datamodule import MultiIrony2TextDataModule
from models import  DualTransformerModel2Text
from transformers import ViTFeatureExtractor, ViTModel
from utils import getModelTokenizerConfig
import torch
from sklearn.metrics import classification_report, accuracy_score, fbeta_score
import gc
import os
from utils import loadAllImages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 1234
seed_everything(SEED)
import time

def check_params(args=None):
    parse = argparse.ArgumentParser(description='Method for automatically irony detection in multiple languages')
    parse.add_argument('-i', '--input', help='File with the input data', required=False, default="train.tsv")

    parse.add_argument('-t', '--test', help='File with the input data for testing', required=True,
                       default="test.tsv")

    parse.add_argument('-v', '--validation', help='File with the input data for testing', required=True,
                       default="valid.tsv")

    parse.add_argument('-o', '--output', help='File path for writing the results and metrics', required=False,
                       default="output")
    parse.add_argument('-d', '--dropout', help='Dropout values used to reduce the overfitting ', required=False,
                       type=float, default=conf.defaultDp)
    parse.add_argument('-ml', '--maxlength', help='Max length of the sequences used for training', required=False,
                       type=int, default=conf.defaultMlenght)
    parse.add_argument('-p', '--epochs', help='Number of epoch used in the training phase', required=False, type=int,
                       default=conf.defaultEpoch)
    parse.add_argument('-b', '--batchsize', help='Batch size used in the training process', required=False, type=int,
                       default=conf.defaulBatchsize)
    parse.add_argument('-z', '--optimizer', help='Method used for parameters optimizations', required=False, type=str,
                       default=conf.defaultOp)
    parse.add_argument('-r', '--learning', help='Value for the learning rate in the optimizer', required=False,
                       type=float, default=conf.defaultLr)
    parse.add_argument('-sr', '--lrstategy', help='Value for the learning rate in the optimizer', required=False,
                       type=str, choices=['dynamic', 'simple'], default=conf.defaulLrStrategy)
    parse.add_argument('-dr', '--lrdecay', help='Value for the weigth decay in the optimizer', required=False,
                       type=float, default=conf.defaultLr)
    parse.add_argument('-mr', '--minlearning', help='Value for the minimal learning rate in the optimizer',
                       required=False, type=float, default=conf.defaultLr)
    parse.add_argument('-md', '--model', help='Model', required=False,
                       choices=["mbert", "xlm-roberta", "beto", "spanbert", "roberta", "bert", 'bertweet', "alberto",
                                'umberto','deberta'],
                       default='bertweet')
    parse.add_argument('-l', '--language', help='Language of the model', required=False, default="en",
                       choices=['ml', 'es', 'en', 'it'])
    parse.add_argument('-w', '--patience', help='Patience for the early stopping criterion', required=False, type=int,
                       default=conf.defaultEpatience)
    parse.add_argument('-db', '--database', help='name of the folder for storing the local database', required=False, type=str,
                       default=conf.database)
    parse.add_argument('-c', '--cuda', help='Number of GPU to use', required=False, type=int,default=0)

    parse.add_argument('-im', '--imagefolder', help='name of the folder with the images', required=False, type=str,
                       default="dataset_images/dataset_images")
    results = parse.parse_args(args)
    return results


if __name__ == "__main__":
    import sys
    from utils import SqliteDB
    parameters = check_params(sys.argv[1:])
    database = f"{parameters.database}/urls.sqlite3"
    sql3 = SqliteDB(database)
    fileoutname = parameters._get_kwargs()
    fileoutname.sort()
    shorten = "%%%".join([f'{x[0]}_{str(x[1])}' for x in fileoutname])
    fileoutname = sql3.shorten(shorten)

    print(fileoutname)
    #sys.exit(1)
    sql3.close()
    imageInventory=None
    if conf.READ_IMAGES==True:
        print("Reading the full images folder to RAM")
        imageInventory= loadAllImages(parameters.imagefolder)

    ironyDataTrain = pd.read_csv(parameters.input, sep="\t")
    ironyDataTrain =ironyDataTrain.fillna("No text Available")
    ironyDataTrain = shuffle(ironyDataTrain, random_state=SEED)
    ironyTrainID, ironyTrainLabels, ironyDataTrain = ironyDataTrain["id"], ironyDataTrain["irony"], ironyDataTrain[["irony", "preprotext", 'images','tags']]
    trainlab = ironyTrainLabels.to_numpy()

    ironyDataTest = pd.read_csv(parameters.test, sep="\t")
    ironyDataTest =ironyDataTest.fillna("No text Available")
    ironyDataTest = shuffle(ironyDataTest, random_state=SEED)
    #Es importante el orden en que se filtran los datos.
    ironyTestID, ironyTestLabels, ironyDataTest = ironyDataTest["id"], ironyDataTest["irony"], ironyDataTest[["irony", "preprotext",'images','tags']]
    testlab = ironyTestLabels.to_numpy()


    ironyDataVal = pd.read_csv(parameters.validation, sep="\t")
    ironyDataVal = ironyDataVal.fillna("No text Available")
    ironyDataVal = shuffle(ironyDataVal, random_state=SEED)
    ironyValID, ironyValLabels, ironyDataVal = ironyDataVal["id"], ironyDataVal["irony"], ironyDataVal[["irony", "preprotext",'images','tags']]
    vallab = ironyValLabels.to_numpy()

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    overallACC, overallF1 = 0.0, 0.0

    history = []
    os.makedirs(parameters.output, exist_ok=True)
    with open(parameters.output + os.sep + fileoutname, 'w')  as myout:
        history.append({'loss': [], 'acc': [], 'dev_loss': [], 'dev_acc': []})
        torch.cuda.empty_cache()

        pretrainedmodel, tokenizer = getModelTokenizerConfig(parameters.model, parameters.language)
        imodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # Obtaining the data split for training and test.
        print("#" * 50)
        print("TRAINING:", sum(trainlab), "/", len(trainlab))
        print("TEST ", sum(testlab), "/", len(testlab))
        print("#" * 50)
        data = MultiIrony2TextDataModule(train=ironyDataTrain, image_folder=parameters.imagefolder, test=ironyDataTest, val_data=ironyDataVal,
                               batch_size=parameters.batchsize)  # transform=data_transform)
        model = DualTransformerModel2Text(pretrainedmodel, imodel, tokenizer, feature_extractor, lr=parameters.learning, opt=parameters.optimizer,
                                 lr_strategy=parameters.lrstategy, minlr=parameters.minlearning, cuda=f'cuda:{parameters.cuda}', imageInventory=imageInventory)
        optimizer = model.configure_optimizers()
        data.setup("fit")
        trainloader = data.train_dataloader()
        devloader = data.val_dataloader()
        test = data.test_dataloader()
        del data
        patience = 0
        batches = len(trainloader)
        for epoch in range(parameters.epochs):
            if patience >= parameters.patience: break
            running_loss = 0.0
            perc = 0
            acc = 0
            model.train()
            for j, data in enumerate(trainloader, 0):
                torch.cuda.empty_cache()
                inputs, images, labels, tags = data['X'], data["I"], data['y'].to(model.dev),data["O"]
                optimizer.zero_grad()
                outputs = model(inputs,tags,images)
                loss = model.loss_criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if j == 0:
                        acc = ((torch.max(outputs, 1).indices == labels).sum() / len(labels)).cpu().numpy()
                        running_loss = loss.item()
                    else:
                        acc = (acc + ((torch.max(outputs, 1).indices == labels).sum() / len(
                            labels)).cpu().numpy()) / 2.0
                        running_loss = (running_loss + loss.item()) / 2.0

                if (j + 1) * 100.0 / batches - perc >= 1 or j == batches - 1:
                    perc = (1 + j) * 100.0 / batches
                    print('\r Epoch:{} step {} of {}. {}% loss: {}'.format(epoch + 1, j + 1, batches,
                                                                           np.round(perc, decimals=1),
                                                                           np.round(running_loss, decimals=3)),
                          end="")
            model.eval()
            history[-1]['loss'].append(running_loss)
            with torch.no_grad():
                out = None
                log = None
                for k, data in enumerate(devloader, 0):
                    torch.cuda.empty_cache()
                    inputs, images, label, tags = data['X'], data["I"], data['y'].to(model.dev), data["O"]
                    dev_out = model(inputs,tags, images)
                    if k == 0:
                        out = dev_out
                        log = label
                    else:
                        out = torch.cat((out, dev_out), 0)
                        log = torch.cat((log, label), 0)
                dev_loss = model.loss_criterion(out, log).item()
                dev_acc = ((torch.max(out, 1).indices == log).sum() / len(log)).cpu().numpy()

                history[-1]['acc'].append(acc)
                history[-1]['dev_loss'].append(dev_loss)
                history[-1]['dev_acc'].append(dev_acc)
            if model.best_measure is None or model.best_measure < dev_acc:
                model.best_measure = dev_acc
                model.best_model_name = f'models/{fileoutname}.pt'
                model.save(model.best_model_name)
                patience = 0
            patience += 1
            print(" acc: {} ||| dev_loss: {} dev_acc: {}".format(np.round(acc, decimals=3),
                                                                 np.round(dev_loss, decimals=3),
                                                                 np.round(dev_acc, decimals=3)))


        # ---------------------------------------------------------------------------------------------------------------------------------------------
        model.load(model.best_model_name)
        model.eval()
        with torch.no_grad():
            predictions = None
            groundTruth = None
            allData, allIma, allTag=None, None,None
            for x, batch in enumerate(test):
                data, ima, labels, tags = batch['X'], batch["I"],batch['y'].to(model.dev), batch["O"]
                pred = model.predict_step(data,tags, ima)
                if x == 0:
                    predictions = pred
                    groundTruth = labels
                    allIma=ima
                    allData=data
                    allTag=tags
                else:
                    predictions = torch.cat((predictions, pred), 0)
                    groundTruth = torch.cat((groundTruth, labels), 0)
                    allData+=data
                    allIma+=ima
                    allTag+=tags
            target_names = ['no irony', 'irony']
            metrics = classification_report(groundTruth.cpu().detach().numpy(), predictions.cpu().detach().numpy(),
                                            target_names=target_names, digits=4)
            df=pd.DataFrame({"Text":allData, "Image":allIma, "Tags":allTag, "GroundTruth":groundTruth.cpu().detach().numpy(),"Prediction": predictions.cpu().detach().numpy()})
            df.to_csv(f"predictions_{fileoutname}.csv")
            print(metrics)
            myout.write(metrics + '\n')
            overallACC += accuracy_score(groundTruth.cpu().detach().numpy(), predictions.cpu().detach().numpy())
            overallF1 += fbeta_score(groundTruth.cpu().detach().numpy(), predictions.cpu().detach().numpy(),
                                     average='macro', beta=0.5)
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # Releasing the MEMORY
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        del tokenizer
        del test
        del model
        del trainloader
        del devloader
        gc.collect()
        print(f'Overall ACC {overallACC}')
        print(f'Overall F1-Macro {overallF1}')
