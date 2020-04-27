'''
  Author: Julia Jeng, Shu Wang, Arman Anwar
  Brief: AIT 726 Homework 4
  Usage:
      Put file 'semantic_role_labeler.py' and folder 'data.wsj' in the same folder.
      -semantic_role_labeler.py
      -data.wsj
        |---ne                                  # ne : Named Entities.
        |---props                               # props : Target verbs and correct propositional arguments.
        |---synt.cha                            # synt.cha : PoS tags and full parses of Charniak.
        |---words                               # words : words.
      -data
        |---test-set.txt
        |---train-set.txt
      -temp
      -make-testset.sh                          # run with bash to get test set.
      -make-trainset.sh                         # run with bash to get train set.
      -srl-eval.pl
      -semantic_role_labeler.py
  Command to run:
      python semantic_role_labeler.py
  Description:
      Build and train a recurrent neural network (RNN) with hidden vector size 256.
      Loss function: Adam loss.
      Embedding vector: 128-dimensional.
      Learning rate: 0.0001.
      Batch size: 256
'''

import sys
import re
import os
import math
import random
import pandas as pd
from collections import defaultdict
from gensim import models
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as torchdata
from sklearn.metrics import accuracy_score
import torch

# global path
rootPath = './'
dataPath = rootPath + '/data/'
tempPath = rootPath + '/temp/'
outsPath = rootPath + '/outputs/'
modsPath = rootPath + '/models/'
# global variable
maxEpoch = 100
perEpoch = 1
judEpoch = 5 # > 1
# marco variable
BatchSize = 16
LearningRate = 0.0001
ExtraDims = 3
NumLabels = 39

pd.options.display.max_columns = None
pd.options.display.max_rows = None
#np.set_printoptions(threshold=np.inf)

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def main():
    # initialize the log file.
    #sys.stdout = Logger('semantic_role_labeler.txt')
    print("-- AIT726 Homework 4 from Julia Jeng, Shu Wang, and Arman Anwar --")
    # read data from files.
    # [0/WORD, 1/POS, 2/FULL_SYNT, 3/NE, 4/TARGETS, (5/PROP)....]
    trainSents = ReadData('Train')
    testSents = ReadData('Test')
    # get vocabulary and dictionary.
    wordDict, posDict, neDict, propsDict, maxLen = GetVocab(trainSents, testSents)
    # pad the data.
    PadData(trainSents, maxLen)
    PadData(testSents, maxLen)
    # split data to train and valid.
    trainSents, validSents = SplitData(trainSents)
    # get mapping.
    # [0/WORD, 1/POS, 2/NE, 3/TARGETS, (4/PROP)....]
    trainSentsIndex = GetMapping(trainSents, wordDict, posDict, neDict, propsDict)
    validSentsIndex = GetMapping(validSents, wordDict, posDict, neDict, propsDict)
    testSentsIndex = GetMapping(testSents, wordDict, posDict, neDict, propsDict)
    # separate targets and generate dataset.
    dTrain, lTrain = GetDataset(trainSentsIndex)
    dValid, lValid = GetDataset(validSentsIndex)
    dTest, lTest = GetDataset(testSentsIndex)
    # get preWeights
    preWeights = GetEmbedding(wordDict)
    # train model.
    TRAIN = 0
    if TRAIN:
        model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights)
    else:
        preWeights = torch.from_numpy(preWeights)
        model = LongShortTermMemoryNetworks(preWeights, preTrain=True, bidirect=True, hiddenSize=256)
        model.load_state_dict(torch.load(modsPath + '/model.pth'))
    # test model.
    TestRNN(model, dTest, lTest, wordDict, propsDict)
    # output the format.
    OutputEval(model, 'Test', wordDict, posDict, neDict, propsDict)
    return

def ReadData(dataset):

    def PreProc(sentence):
        # relabel NE and PROPS.
        def ReLabel(sentence, k):
            mark = [0, '']
            #for word in sentence:
            for word in sentence:
                if word[k][0] == '(' and word[k][-2:] == '*)':  # (_*)
                    word[k] = word[k][1:-2]
                elif word[k][0] == '(' and word[k][-1] == '*':  # (_*
                    word[k] = word[k][1:-1]
                    mark = [1, word[k]]
                elif word[k] == '*' and mark[0] == 1:
                    word[k] = mark[1]
                elif word[k] == '*)':                           # *)
                    word[k] = mark[1]
                    mark = [0, '']
            return sentence

        # delabel NE and PROPS.
        def DeLabel(sentence, k):
            # get mark.
            mark = np.zeros(numWords+1)
            for i in range(1, numWords):
                if sentence[i][k] == sentence[i-1][k]:
                    mark[i] = mark[i-1]
                else:
                    mark[i] = mark[i-1] + 1
            print(mark)
            # process.
            for i in range(numWords):
                if sentence[i][k] == '*':
                    continue
                sign = '*'
                if i == 0 or mark[i] != mark[i-1]:
                    sign = '(' + sentence[i][k] + sign  # (_*
                if i == numWords-1 or mark[i] != mark[i+1]:
                    sign = sign + ')'  # (_*) or *)
                sentence[i][k] = sign
            return sentence

        # number of words.
        numWords = len(sentence)
        # process TARGETS.
        for word in sentence:
            if word[4] == '-':
                word[4] = 0
            else:
                word[4] = 1
        # process NE.
        ReLabel(sentence, 3)
        # process PROPS.
        numProps = len(sentence[0]) - 5
        if numProps:
            for i in range(5, 5+numProps):
                ReLabel(sentence, i)
        return sentence

    # input validation.
    if dataset.lower() not in ['train', 'valid', 'test']:
        print('[Error] Input invalid! [' + dataset + ']')
        return

    # check if there is a cache file.
    if os.path.exists(tempPath + '/' + dataset.lower() + 'Sentences.npy'):
        print('[Info] Load ' + dataset.lower() + ' data from ' + tempPath + '/' + dataset.lower() + 'Sentences.npy')
        return np.load(tempPath + '/' + dataset.lower() + 'Sentences.npy', allow_pickle=True)

    # file name.
    filename = dataPath + '/' + dataset.lower() + '-set.txt'
    print('[Info] Read data from ' + filename)

    # read data from file and store at sentences
    sentences = []
    sentence = []
    file = open(filename).readlines()
    for line in file:
        segments = line.split()
        # print(segments)
        if len(segments): # line is not empty.
            sentence.append(segments)
        else: # line is empty.
            if len(sentence):
                sentences.append(sentence)
            sentence = []
    if len(sentence):
        sentences.append(sentence)
    # show sentences[1]
    # sentence = sentences[1]
    # df = pd.DataFrame(sentence)
    # print(df)

    # for each sentence.
    for sentence in sentences: # [0/WORD, 1/POS, 2/FULL_SYNT, 3/NE, 4/TARGETS, 5/PROP....]
        PreProc(sentence)
        # print(sentence)
        # numProps = sum([word[4] for word in sentence])
        # print(numProps)

    # save data.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    np.save(tempPath + '/' + dataset.lower() + 'Sentences.npy', sentences)

    return sentences

def GetVocab(trainSents, testSents):
    # combine train and test set.
    words = []
    words.extend([word for sent in trainSents for word in sent])
    words.extend([word for sent in testSents for word in sent])

    # [0/WORD, 1/POS, 2/FULL_SYNT, 3/NE, 4/TARGETS, 5/PROP....]
    if os.path.exists(tempPath + '/wordVocab.npy'):
        wordVocab = np.load(tempPath + '/wordVocab.npy', allow_pickle=True)
    else:
        wordList = [word[0] for word in words]
        wordVocab = list(set(wordList))
        wordVocab.sort(key=wordList.index)
        wordVocab.insert(0, '<pad>')
        np.save(tempPath + '/wordVocab.npy', wordVocab)
    print('[Info] Get %d vocabulary WORD successfully.' % (len(wordVocab)))

    if os.path.exists(tempPath + '/posVocab.npy'):
        posVocab = np.load(tempPath + '/posVocab.npy', allow_pickle=True)
    else:
        posList = [word[1] for word in words]
        posVocab = list(set(posList))
        posVocab.sort(key=posList.index)
        posVocab.insert(0, '<pad>')
        np.save(tempPath + '/posVocab.npy', posVocab)
    print('[Info] Get %d vocabulary POS successfully.' % (len(posVocab)))

    if os.path.exists(tempPath + '/syntVocab.npy'):
        syntVocab = np.load(tempPath + '/syntVocab.npy', allow_pickle=True)
    else:
        syntList = [word[2] for word in words]
        syntVocab = list(set(syntList))
        syntVocab.sort(key=syntList.index)
        syntVocab.insert(0, '<pad>')
        np.save(tempPath + '/syntVocab.npy', syntVocab)
    print('[Info] Get %d vocabulary FULL_SYNT successfully.' % (len(syntVocab)))

    if os.path.exists(tempPath + '/neVocab.npy'):
        neVocab = np.load(tempPath + '/neVocab.npy', allow_pickle=True)
    else:
        neList = [word[3] for word in words]
        neVocab = list(set(neList))
        neVocab.sort(key=neList.index)
        neVocab.insert(0, '<pad>')
        np.save(tempPath + '/neVocab.npy', neVocab)
    print('[Info] Get %d vocabulary NE successfully.' % (len(neVocab)))

    if os.path.exists(tempPath + '/propsVocab.npy'):
        propsVocab = np.load(tempPath + '/propsVocab.npy', allow_pickle=True)
    else:
        propsList = []
        for word in words:
            propsList.extend(word[5:])
        propsVocab = list(set(propsList))
        propsVocab.sort(key=propsList.index)
        propsVocab.insert(0, '<pad>')
        np.save(tempPath + '/propsVocab.npy', propsVocab)
    print('[Info] Get %d vocabulary PROPS successfully.' % (len(propsVocab)))

    # get dictionary.
    wordDict = {item: index for index, item in enumerate(wordVocab)}
    posDict = {item: index for index, item in enumerate(posVocab)}
    neDict = {item: index for index, item in enumerate(neVocab)}
    propsDict = {item: index for index, item in enumerate(propsVocab)}

    # combine train and test set.
    trainMaxLen = max([len(sent) for sent in trainSents])
    testMaxLen = max([len(sent) for sent in testSents])
    maxLen = max(trainMaxLen, testMaxLen)

    return wordDict, posDict, neDict, propsDict, maxLen

def PadData(sentences, maxLen):
    for sentence in sentences:
        dims = len(sentence[0])
        pads = ['<pad>', '<pad>', '<pad>', '<pad>', 0]
        if dims > 5:
            pads.extend(['<pad>' for ind in range(dims-5)])
        for ind in range(maxLen - len(sentence)):
            sentence.append(pads)
    return sentences

def SplitData(trainSents):
    # train/valid
    splitRate = 0.8
    numSents = len(trainSents)
    numTrain = round(numSents * splitRate)
    # random list
    listSents = [ind for ind in range(numSents)]
    random.shuffle(listSents)
    # split.
    train = [trainSents[ind] for ind in listSents[:numTrain]]
    valid = [trainSents[ind] for ind in listSents[numTrain:]]
    return train, valid

def GetMapping(sentences, wordDict, posDict, neDict, propsDict):
    newSents = []
    for sentence in sentences:
        # print(sentence)
        dims = len(sentence[0])
        # print(dims)
        nSent = []
        for word in sentence:
            nWord = []
            nWord.append(wordDict[word[0]])
            nWord.append(posDict[word[1]])
            nWord.append(neDict[word[3]])
            nWord.append(word[4])
            if dims > 5:
                for ind in range(5, dims):
                    nWord.append(propsDict[word[ind]])
            nSent.append(nWord)
        # print(nSent)
        newSents.append(nSent)
    return newSents

def GetDataset(sentsIndex):
    dataset = []
    labels = []
    # for each sentence.
    for sent in sentsIndex:
        npSent = np.array(sent)
        npSent = npSent.T
        #print(npSent)
        # [0/WORD; 1/POS; 2/NE; 3/TARGETS; (4/PROP)....]
        targets = npSent[3]
        index = np.array(np.where(targets == 1))[0]
        numTargets = len(index)
        if numTargets != 0:
            for ind in range(numTargets):
                target = np.zeros(targets.shape, dtype=int)
                target[index[ind]] = 1
                # construct data
                data = npSent[0:3]
                data = np.row_stack((data, target))
                #print(data)
                label = npSent[4+ind]
                #print(label)
                dataset.append(data)
                labels.append(label)
    # print(np.array(dataset))
    # print(np.array(labels))
    return np.array(dataset), np.array(labels)

def GetEmbedding(wordDict):
    '''
    Get the embedding vectors from files.
    :param wordDict: word dictionary
    :return: pre-trained weights.
    '''
    # load preWeights.
    weightFile = 'preWeights.npy'
    if not os.path.exists(tempPath + '/' + weightFile):
        # find embedding file.
        embedFile = 'GoogleNews.txt'
        if not os.path.exists(tempPath + '/' + embedFile):
            # path validation.
            modelFile = 'GoogleNews-vectors-negative300.bin'
            if not os.path.exists(tempPath + '/' + modelFile):
                print('[Error] Cannot find %s/%s.' % (tempPath, modelFile))
                return
            # find word2vec file.
            model = models.KeyedVectors.load_word2vec_format(tempPath + '/' + modelFile, binary=True)
            model.save_word2vec_format(tempPath + '/' + embedFile)
            print('[Info] Get the word2vec format file %s/%s.' % (tempPath, embedFile))

        # read embedding file.
        embedVec = {}
        file = open(tempPath + '/' + embedFile, encoding = 'utf8')
        for line in file:
            seg = line.split()
            word = seg[0]
            embed = np.asarray(seg[1:], dtype = 'float32')
            embedVec[word] = embed
        np.save(tempPath+'/embedVec.npy', embedVec)

        # get mapping to preWeights.
        numWords = len(wordDict)
        numDims = 300
        preWeights = np.zeros((numWords, numDims))
        for ind, word in enumerate(wordDict):
            if word in embedVec:
                preWeights[ind] = embedVec[word]
            else:
                preWeights[ind] = np.random.normal(size=(numDims,))

        # save the preWeights.
        np.save(tempPath + '/' + weightFile, preWeights)
        print('[Info] Get pre-trained word2vec weights.')
    else:
        preWeights = np.load(tempPath + '/' + weightFile)
        print('[Info] Load pre-trained word2vec weights from %s/%s.' % (tempPath, weightFile))
    return preWeights

class LongShortTermMemoryNetworks(nn.Module):
    '''
    LSTM model.
    '''
    def __init__(self, preWeights, preTrain=True, bidirect=True, hiddenSize=256):
        super(LongShortTermMemoryNetworks, self).__init__()
        # sparse parameters.
        numWords, numDims = preWeights.size()
        numBiDirect = 2 if bidirect else 1
        # embedding layer.
        self.embedding = nn.Embedding(num_embeddings=numWords, embedding_dim=numDims)
        self.embedding.load_state_dict({'weight': preWeights})
        if preTrain:
            self.embedding.weight.requires_grad = False
        # LSTM layer.
        self.lstm = nn.LSTM(input_size=numDims+ExtraDims, hidden_size=hiddenSize, batch_first=True, bidirectional=bidirect)
        # fully-connected layer.
        self.fc = nn.Linear(in_features=hiddenSize*numBiDirect, out_features=NumLabels)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        embeds = self.embedding(x[:,0,:])
        # print(embeds.size())
        features = x[:,1:,:]
        features = features.permute(0,2,1)
        inputs = torch.cat((embeds.float(), features.float()), 2)
        rnn_out, hidden = self.lstm(inputs)
        out = rnn_out.contiguous().view(-1, rnn_out.shape[2])
        a = self.fc(out)
        # a = self.sm(a)
        return a

def TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, batchsize=BatchSize, learnRate=LearningRate):
    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=True)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=True)

    # get training weights
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(1, NumLabels):
        weights.append( 1 - (lbTrain.count(lb)) / (len(lbTrain) - lbTrain.count(0)))
    #weights.insert(0, 0.75)
    weights.insert(0, 0)
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWeights = torch.from_numpy(preWeights)
    model = LongShortTermMemoryNetworks(preWeights, preTrain=True, bidirect=True, hiddenSize=256)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('[Demo] --- RNNType: LSTM | HiddenNodes: 256 | Bi-Direction: True | Pre-Trained: True ---')
    print('[Para] BatchSize=%d, LearningRate=%.5f, MaxEpoch=%d, JudEpoch=%d, PerEpoch=%d.' % (batchsize, learnRate, maxEpoch, judEpoch, perEpoch))
    # optimizing with Adam.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as cross entropy loss.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)

    # run on each epoch.
    accList = [0]
    for epoch in range(maxEpoch):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        lossTrain /= len(lbTrain)
        # train accuracy.
        padIndex = [ind for ind, lb in enumerate(labels) if lb == 0]
        for ind in sorted(padIndex, reverse=True):
            del predictions[ind]
            del labels[ind]
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        # valid accuracy.
        padIndex = [ind for ind, lb in enumerate(labels) if lb == 0]
        for ind in sorted(padIndex, reverse=True):
            del predictions[ind]
            del labels[ind]
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # output information.
        if 0 == (epoch + 1) % perEpoch:
            print('[Epoch %03d] loss: %.3f, train acc: %.3f%%, valid acc: %.3f%%.' % (epoch + 1, lossTrain, accTrain, accValid))
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + '/model.pth')
        # stop judgement.
        if (epoch + 1) >= judEpoch and accList[-1] < min(accList[-judEpoch:-1]):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model.pth'))
    return model

def TestRNN(model, dTest, lTest, wordDict, classDict):
    # test period
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=BatchSize, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # testing phase.
    model.eval()
    words = []
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            yhat = model.forward(data)  # get output
            # statistic
            words.extend(data[:,0,:].contiguous().view(-1).int().tolist())
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    # testing accuracy.
    padIndex = [ind for ind, wd in enumerate(words) if wd == 0]
    for ind in sorted(padIndex, reverse=True):
        del words[ind]
        del predictions[ind]
        del labels[ind]
    accuracy = accuracy_score(labels, predictions) * 100
    print('[Eval] Testing accuracy: %.3f%%.' % (accuracy))

    # get inverse index dictionary.
    wordIndDict = {ind: item for ind, item in enumerate(wordDict)}
    classIndDict = {ind: item for ind, item in enumerate(classDict)}
    # print(wordIndDict)
    # print(classIndDict)
    # output preparation.
    outWords = [wordIndDict[item] for item in words]
    outLabels = [classIndDict[item] for item in labels]
    # predictions = [(1 if 0 == item else item) for item in predictions]
    outPredictions = [classIndDict[item] for item in predictions]

    # file operation.
    if not os.path.exists(outsPath):
        os.mkdir(outsPath)
    filename = 'output.txt'
    fout = open(outsPath + '/' + filename, 'w')
    for i in range(len(outLabels)):
        fout.write(outWords[i] + ' ' + outLabels[i] + ' ' + outPredictions[i] + '\n')
    fout.close()

    return accuracy

def OutputEval(model, dataset, wordDict, posDict, neDict, propsDict):
    # input validation.
    if dataset.lower() not in ['train', 'valid', 'test']:
        print('[Error] Input invalid! [' + dataset + ']')
        return

    # load data set.
    Sents = ReadData(dataset)
    numSents = len(Sents)

    # set model environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # for each sentence.
    #for sent in Sents:
    if 1:
        sent = Sents[2]

        ###
        # dims = len(sent[0])
        print(sent)

        sentIndex = GetMapping([sent], wordDict, posDict, neDict, propsDict)
        print(sentIndex)

        data, label = GetDataset(sentIndex)
        print(data)
        print(label)

        numTargets = len(data)
        print(numTargets)

        if numTargets:
            Tdata = torch.from_numpy(data).long().cuda()
            Tlabel = torch.from_numpy(label).long().cuda()
            print(Tdata)
            print(Tlabel)
            TDataset = torchdata.TensorDataset(Tdata, Tlabel)
            TLoader = torchdata.DataLoader(TDataset, batch_size=256, shuffle=False)
            print(TLoader)
            with torch.no_grad():
                for iter, (data, label) in enumerate(TLoader):
                    print(data)
                    print(label)
        else:
            pass




    return

if __name__ == '__main__':
    main()