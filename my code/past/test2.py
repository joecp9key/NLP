import nltk
import pycrfsuite
train_sent = list(nltk.corpus.conll2000.iob_sents('train.txt'))
test_sent = list(nltk.corpus.conll2000.iob_sents('test.txt'))


def word2features(sent, i):
    word = sent[i][0] #token
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2]
    ]
    
    # look up the previous word
    if i > 0:
        word1 =sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag[:2] #we're interested in normal form 
        ])
    else:
        features.append('BOS')
        
    # loop up the next wrd
    if i < len(sent) -1:
        word2 = sent[i + 1][0]
        postag2 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word2.lower(),
            '+1:word.istitle=%s' % word2.istitle(),
            '+1:word.isupper=%s' % word2.isupper(),
            '+1:postag=' + postag2,
            '+1:postag[:2]=' +postag2[:2]
        ])
    else:
        features.append('EOS')
    return features
    
def sent2features(sent):
    return [ word2features(sent,i) for i in range(len(sent)) ]
    
def sent2labels(sent):
    return [ label for token, pos, label in sent]

def sent2tokens(sent):
    return [ token for token, pos, label in sent]

def train():
	X_train = [ sent2features(s) for s in train_sent]
	Y_train = [ sent2labels(s) for s in train_sent]

	trainer = pycrfsuite.Trainer(verbose=False)
	trainer.set_params({
	    'c1': 1.0,
	    'c2': 1e-3,
	    'max_iterations': 50,
	    'feature.possible_transitions': True
	})

	for xseq, yseq in zip(X_train, Y_train):
	    trainer.append(xseq, yseq)
	    
	trainer.train('mytrain_model')

def predict():
    tagger = pycrfsuite.Tagger()
    tagger.open('mytrain_model')
    example_set = test_sent[3]
    print(' '.join(sent2tokens(example_set)), end='\n\n')
    print("Predicted:", ' '.join(tagger.tag(sent2features(example_set))))
    print("Correct:  ", ' '.join(sent2labels(example_set)))


train()
predict()