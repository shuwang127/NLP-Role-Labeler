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
        |---test-set.txt                        # get from .sh file.
        |---train-set.txt                       # get from .sh file.
      -temp
        |---GoogleNews-vectors-negative300.bin  # embedding file.
      -models
        |---model.pth                           # best model we get.
      -outputs
        |---outputs.txt                         # model outputs.
        |---test_outputs.txt                    # outputs that satisfies HW requirement.
      -make-testset.sh                          # run with bash to get test set.
      -make-trainset.sh                         # run with bash to get train set.
      -senmantic_role_labeler.txt               # log file.
      -srl-eval.pl

  Command to run:
      python semantic_role_labeler.py
  Description:
      Build and train a recurrent neural network (RNN) with hidden vector size 256.
      Loss function: Adam loss.
      Embedding vector: 300-dimensional.
      Learning rate: 0.0001.
      Batch size: 16
'''

# dependencies
import sys
import os
import random
import pandas as pd
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
# print setting.
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def main():
    # initialize the log file.
    sys.stdout = Logger('semantic_role_labeler.txt')
    print("-- AIT726 Homework 4 from Julia Jeng, Shu Wang, and Arman Anwar --")
    # read data from files.
    # [0/WORD, 1/POS, 2/FULL_SYNT, 3/NE, 4/TARGETS, (5/PROP)....]
    trainSents, _ = ReadData('Train')
    testSents, _ = ReadData('Test')
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
    MODEL_TRAIN = True
    if MODEL_TRAIN:
        model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights)
    else:
        preWeights = torch.from_numpy(preWeights)
        model = LongShortTermMemoryNetworks(preWeights, preTrain=True, bidirect=True, hiddenSize=256)
        model.load_state_dict(torch.load(modsPath + '/model.pth'))
    # test model.
    TestRNN(model, dTest, lTest, wordDict, propsDict)
    # output the format.
    OutputEval(model, 'Test', wordDict, posDict, neDict, propsDict, maxLen)
    return model

def ReadData(dataset):
    '''
    Read data from file and get pre-process contents and targets.
    :param dataset: indicates 'train' or 'test' dataset.
    :return: sentences - preprocess variables.
             targets - original targets.
    '''

    def PreProc(sentence):
        '''
        Pre-process each sentence.
        :param sentence: input sentence. numWords * dims.
        :return: sentence - preprocessed sentence.
        '''
        # relabel NE and PROPS.
        def ReLabel(sentence, k):
            '''
            Relabel props and ne
            :param sentence: input sentence. numWords * dims.
            :param k: process k-column.
            :return: sentence - preprocessed sentence.
            '''
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
            '''
            Delabel props and ne, invert from Relabel.
            :param sentence: input sentence. numWords * dims.
            :param k: process k-column.
            :return: sentence - preprocessed sentence.
            '''
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
    if os.path.exists(tempPath + '/' + dataset.lower() + 'Sentences.npy')\
            and os.path.exists(tempPath + '/' + dataset.lower() + 'Targets.npy'):
        print('[Info] Load ' + dataset.lower() + ' data from ' + tempPath + '/' + dataset.lower() + 'Sentences.npy')
        sentences = np.load(tempPath + '/' + dataset.lower() + 'Sentences.npy', allow_pickle=True)
        targets = np.load(tempPath + '/' + dataset.lower() + 'Targets.npy', allow_pickle=True)
        return sentences, targets

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

    # output targets.
    targets = []
    for sentence in sentences:
        target = []
        for word in sentence:
            target.append([word[4]])
        targets.append(target)
    #print(targets)

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
    np.save(tempPath + '/' + dataset.lower() + 'Targets.npy', targets)

    return sentences, targets

def GetVocab(trainSents, testSents):
    '''
    Get vocabulary from training and testing set.
    :param trainSents: training set.
    :param testSents: testing set.
    :return: wordDict - word dictionary
             posDict - POS dictionary
             neDict - NE dictionary
             propsDict - PROPS dictionary
             maxLen - max sentence length
    '''
    # combine train and test set.
    words = []
    words.extend([word for sent in trainSents for word in sent])
    words.extend([word for sent in testSents for word in sent])

    # [0/WORD, 1/POS, 2/FULL_SYNT, 3/NE, 4/TARGETS, 5/PROP....]
    # WORD
    if os.path.exists(tempPath + '/wordVocab.npy'):
        wordVocab = np.load(tempPath + '/wordVocab.npy', allow_pickle=True)
    else:
        wordList = [word[0] for word in words]
        wordVocab = list(set(wordList))
        wordVocab.sort(key=wordList.index)
        wordVocab.insert(0, '<pad>')
        np.save(tempPath + '/wordVocab.npy', wordVocab)
    print('[Info] Get %d vocabulary WORD successfully.' % (len(wordVocab)))

    # POS
    if os.path.exists(tempPath + '/posVocab.npy'):
        posVocab = np.load(tempPath + '/posVocab.npy', allow_pickle=True)
    else:
        posList = [word[1] for word in words]
        posVocab = list(set(posList))
        posVocab.sort(key=posList.index)
        posVocab.insert(0, '<pad>')
        np.save(tempPath + '/posVocab.npy', posVocab)
    print('[Info] Get %d vocabulary POS successfully.' % (len(posVocab)))

    # FULL_SYNT
    if os.path.exists(tempPath + '/syntVocab.npy'):
        syntVocab = np.load(tempPath + '/syntVocab.npy', allow_pickle=True)
    else:
        syntList = [word[2] for word in words]
        syntVocab = list(set(syntList))
        syntVocab.sort(key=syntList.index)
        syntVocab.insert(0, '<pad>')
        np.save(tempPath + '/syntVocab.npy', syntVocab)
    print('[Info] Get %d vocabulary FULL_SYNT successfully.' % (len(syntVocab)))

    # NE
    if os.path.exists(tempPath + '/neVocab.npy'):
        neVocab = np.load(tempPath + '/neVocab.npy', allow_pickle=True)
    else:
        neList = [word[3] for word in words]
        neVocab = list(set(neList))
        neVocab.sort(key=neList.index)
        neVocab.insert(0, '<pad>')
        np.save(tempPath + '/neVocab.npy', neVocab)
    print('[Info] Get %d vocabulary NE successfully.' % (len(neVocab)))

    # PROP
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
    '''
    For each sentence, pad it to the max sentence length.
    :param sentences: the list of sentences.
    :param maxLen: max sentence length.
    :return: processed sentences.
    '''
    for sentence in sentences:
        dims = len(sentence[0])
        pads = ['<pad>', '<pad>', '<pad>', '<pad>', 0]
        if dims > 5:
            pads.extend(['<pad>' for ind in range(dims-5)])
        for ind in range(maxLen - len(sentence)):
            sentence.append(pads)
    return sentences

def SplitData(trainSents):
    '''
    Split the training set into training and valid set.
    :param trainSents: original training set.
    :return: train - splited training set.
             valid - splited valid set.
    '''
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
    '''
    Mapping the sentences from string to index-form.
    :param sentences: list of sentences.
    :param wordDict: word dictionary.
    :param posDict: POS dictionary.
    :param neDict: NE dictionary.
    :param propsDict: PROP dictionary.
    :return: index-formed sentences.
    '''
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
    '''
    Convert the index-form data to pytorch acceptable form, and divide original data to data and label.
    :param sentsIndex: index-form data.
    :return: dataset - numpy dataset.
             label - numpy label.
    '''
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
    '''
    train the model.
    :param dTrain: train data.
    :param lTrain: train label.
    :param dValid: valid data.
    :param lValid: valid label.
    :param preWeights: pre-trained weights.
    :param batchsize: batch size.
    :param learnRate: learning rate.
    :return: trained model
    '''
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
    '''
    run the model on test set and get the accuracy.
    :param model: trained model.
    :param dTest: test data.
    :param lTest: test label.
    :param wordDict: word dictionary
    :param classDict: props dictionary.
    :return: accuracy - the testing accuracy.
    '''
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
    filename = 'outputs.txt'
    fout = open(outsPath + '/' + filename, 'w')
    for i in range(len(outLabels)):
        fout.write(outWords[i] + ' ' + outLabels[i] + ' ' + outPredictions[i] + '\n')
    fout.close()

    return accuracy

def OutputEval(model, dataset, wordDict, posDict, neDict, propsDict, maxLen):
    '''
    Eval the test result and convert result into the HW required form.
    :param model: trained model.
    :param dataset: 'train' or 'test'
    :param wordDict: word dictionary.
    :param posDict: POS dictionary.
    :param neDict: NE dictionary.
    :param propsDict: PROP dictionary.
    :param maxLen: max sentence length.
    :return: outputs - HW required form outputs.
    '''
    # delabel NE and PROPS.
    def DeLabel(sentence, k):
        '''
        Decode the sentence in k-column.
        :param sentence: input sentence.
        :param k: process the k-column.
        :return: processed sentence.
        '''
        # number of words.
        numWords = len(sentence)
        # get mark.
        mark = np.zeros(numWords + 1)
        for i in range(1, numWords):
            if sentence[i][k] == sentence[i - 1][k]:
                mark[i] = mark[i - 1]
            else:
                mark[i] = mark[i - 1] + 1
        #print(mark)
        # process.
        for i in range(numWords):
            if sentence[i][k] == '*':
                continue
            sign = '*'
            if i == 0 or mark[i] != mark[i - 1]:
                sign = '(' + sentence[i][k] + sign  # (_*
            if i == numWords - 1 or mark[i] != mark[i + 1]:
                sign = sign + ')'  # (_*) or *)
            sentence[i][k] = sign
        return sentence

    # input validation.
    if dataset.lower() not in ['train', 'valid', 'test']:
        print('[Error] Input invalid! [' + dataset + ']')
        return

    # load data set.
    Sents, Targets = ReadData(dataset)
    numSents = len(Sents)

    # set model environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    torch.no_grad()

    outputs = []
    InvPropsDict = {ind: item for ind, item in enumerate(propsDict)}
    #print(InvPropsDict)
    # for each sentence.
    for ind in range(numSents):
        # get sentence and targets.
        sent = Sents[ind]
        targ = Targets[ind]
        # print(sent)
        # print(targ)
        # get info.
        length = len(sent)
        dims = len(sent[0])
        # pad the data
        sentPad = PadData([sent], maxLen)
        # map to index form.
        sentIndex = GetMapping(sentPad, wordDict, posDict, neDict, propsDict)
        # convert to model form.
        data, label = GetDataset(sentIndex)
        #print(data)
        #print(label)

        # judge if there are verb targets.
        numTargets = len(data)
        if numTargets:
            # run model.
            x = torch.from_numpy(data).long().cuda()
            y = torch.from_numpy(label).long().cuda()
            #print(y)
            x = x.to(device)
            y = y.contiguous().view(-1)
            y = y.to(device)
            yhat = model.forward(x)  # get output
            # statistic
            preds = yhat.max(1)[1]
            #print(preds)
            predictions = preds.tolist()
            predictions = np.reshape(predictions, (numTargets, -1))
            predictions = predictions[:, :length]
            #print(predictions)
            torch.cuda.empty_cache()
            # cat the targ.
            for nTarget in range(numTargets):
                for nWord in range(length):
                    prop = predictions[nTarget][nWord]
                    prop = InvPropsDict[prop]
                    targ[nWord].append(prop)
        else:
            pass
        outputs.append(targ)
    #print(outputs)

    # Delabel.
    for out in outputs:
        numProps = len(out[0]) - 1
        #print(numProps)
        if numProps:
            for ind in range(1, 1 + numProps):
                DeLabel(out, ind)

    #print(outputs)
    #out = outputs[0]
    #df = pd.DataFrame(out)
    #print(df)

    # file operation.
    if not os.path.exists(outsPath):
        os.mkdir(outsPath)
    filename = dataset.lower() + '_outputs.txt'
    fout = open(outsPath + '/' + filename, 'w')
    print('[Info] Outputs for testing are shown as follows:')
    for out in outputs:
        for word in out:
            for item in word:
                print('%-16s\t' % (item), end='')
                fout.write('%-16s\t' % (item))
            print('')
            fout.write('\n')
        print('')
        fout.write('\n')
    fout.close()
    print('[Info] Outputs have been saved in ' + outsPath + '/' + filename)

    return outputs

if __name__ == '__main__':
    main()