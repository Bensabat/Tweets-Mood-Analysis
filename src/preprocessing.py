import pandas as pd
import re 
import emoji
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from string import punctuation
punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~...'''
...

tknzr = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
notstopwords = set(('not', 'no'))
stopwords = set( stopwords.words('english')) - notstopwords




def load_data_semeval(path_tweet, label):
	try:
		if label=='True':
			tweet = pd.read_csv(path_tweet, encoding='utf-8',sep='\t')
			text = tweet['turn1']+" "+tweet['turn2']+" "+tweet['turn3']
			return text, tweet['label']
		else:
			tweet = pd.read_csv(path_tweet, encoding='utf-8',sep='\t')
			text = tweet['turn1']+" "+tweet['turn2']+" "+tweet['turn3']
			return text			

	except IOError: 
		print ("Could not read file:", path_tweet) 
		

def standardization(tweet):
	tweet = re.sub(r"\\u2019", "'", tweet)
	tweet = re.sub(r"\\u002c", ",", tweet)
	tweet=emoji.str2emoji(tweet)
	tweet = re.sub(r"(http|https)?:\/\/[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4}(/\S*)?", " ", tweet)
	tweet = re.sub(r"u r "," you are ",tweet)
	tweet = re.sub(r"U r "," you are ",tweet)
	tweet = re.sub(r" u(\s|$)"," you ",tweet)
	tweet = re.sub(r"didnt","did not",tweet)
	tweet = re.sub(r"\'ve", " have", tweet)
	tweet = re.sub(r" can\'t", " cannot", tweet)
	tweet = re.sub(r"n\'t", " not", tweet)
	tweet = re.sub(r"\'re", " are", tweet)
	tweet = re.sub(r"\'d", " would", tweet)
	tweet = re.sub(r"\'ll", " will", tweet)
	tweet = re.sub(r"\'s", "", tweet)
	tweet = re.sub(r"\'n", "", tweet)
	tweet = re.sub(r"\'m", " am", tweet)
	tweet = re.sub(r"@\w+", r' ',tweet)
	tweet = re.sub(r"#\w+", r' ',tweet)
	tweet = re.sub(r" [0-9]+ "," ",tweet)
	tweet = re.sub(r" plz[\s|$]", " please ",tweet)
	tweet = re.sub(r"^([1-9] |1[0-9]| 2[0-9]|3[0-1])(.|-)([1-9] |1[0-2])(.|-|)20[0-9][0-9]"," ",tweet)
	tweet = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(i) for i,j in pos_tag(tknzr.tokenize(tweet))]
	tweet = [ i for i in tweet if (i not in stopwords) and (i not in punctuation ) ]
	tweet = ' '.join(tweet)
	return tweet.lower()


def data_preprocessing(path_tweet, label):
	if label=='True':
		texts, labels = load_data_semeval(path_tweet,label)
		texts = texts.apply(lambda x: standardization(x))
		labels = labels.apply(lambda x:0 if x=='angry' else (1 if x=='happy' else (2 if x=='sad' else 3) ) )	
		return texts, labels
	else:
		
		texts = load_data_semeval(path_tweet,label)
		texts = texts.apply(lambda x: standardization(x))	
		return texts
	
def external_data(path_tweet):
	tweet = pd.read_csv(path_tweet, encoding='utf-8',sep='\t',names=['id_x','text','label'])
	texts = tweet['text'].apply(lambda x: standardization(x))
	labels = tweet['label'].apply(lambda x:0 if x=='angry'  else (1 if x=='happy'  else 2 ))      
	return texts, labels
	

def data_semeval3(path_tweet):
	data = pd.read_csv(path_tweet, encoding='utf-8',sep='\t', names=['id','class','text'])
	data['class'] = data['class'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2 ))  # 0: 	negative, 1: neutral, 2: positive
	data['text'] = data['text'].apply(lambda x: standardization(x))
	return data['text'], data['class']

		
#text,label=extrnal_data("/home/bouche_a/semeval2019/external_data.txt")

#for i in range(len(text)):
#	print(text[i]+"\t"+str(label[i]))

#print(standardization("My mummy is your grandma	Hey Soul Sister on Repeat <3 :')	Why bc"))
