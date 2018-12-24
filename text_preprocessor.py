import re
import imp
import io
import string
from nltk.corpus import stopwords

def process(p):

    # cache the stopwords from nltk to save time
    cachedStopWords = set(stopwords.words("english"))

    # removes any alphanumeric character in a particular paragraph
    pl = "".join([c if c.isalnum() else " " for c in p])

    # converts all letters to lowercase in a particular paragraph
    lc = pl.lower()

    # removes stopwords from a particular paragraph
    sl = re.compile(r'\b(' + r'|'.join(cachedStopWords) + r')\b\s*').sub('', lc)

    # replaces any double or more spaces with a single space
    fp = " ".join(sl.split())

    return fp

def keyword_in_para(keywords, p):
    # given list of keywords to look for, true if paragraph contains any of them (false otherwise)
    for keyword in keywords:
        if keyword in p:
            return True
    return False

def create_output(filename):

	# initialize strings that contain the output corresponding to each list of keywords
	emailOutput = creditCardOutput = SSNOutput = adsOutput = locationOutput = PIIOutput = lawOutput = changeOutput = controlOutput = aggOutput = ""

	# initialize the lists of keywords
	emailFactors = ["email", "mail", "third", "party", "share", "sell", "promote", "affiliate"]
	creditCard = ["credit", "debit", "card", "bill", "pay", "third", "party", "share", "sell", "promote", "affiliate"]
	SSN = ["social", "security", "number", "ssn", "third", "party", "share", "sell", "promote", "affiliate"]
	ads = ["ad", "ads", "market", "third", "party", "share", "sell", "promote", "affiliate"]
	location = ["locate", "location", "geo", "mobile", "gps", "third", "party", "share", "sell", "promote", "affiliate"]
	childrenPII = ["age", "child"]
	law = ["law", "regulate", "legal", "government", "warrant", "subpoena", "court", "judge"]
	change = ["notice", "change", "update", "post"]
	dataControl = ["data", "choice", "edit", "delete", "limit", "setting", "account", "access", "update", "control"]
	dataAgg = ["data", "aggregate", "identity", "identifiable", "identify"]

	# create lists of factors
	factors = [emailFactors, creditCard, SSN, ads, location, childrenPII, law, change, dataControl, dataAgg]

	# open an example privacy policy and split it by newline to make a list of paragraphs
	paras = open(filename,"r").read().splitlines()

	# define this function to use for a labeled break procedure
	# build an output for each factor
	outputs = []
	for factor in factors:
	    outputs.append([process(p) for p in paras if keyword_in_para(factor, process(p))])

	return outputs

def create_dictionary():
    
    # initialize an empty string that will hold every word in every text file
	words = ""
    
    # initialize a dictionary with the key "<PAD>" corresponding to the value 0, this is to utulize sequence padding from keras
	d = {}
	d["<PAD>"] = 0

    # read in every word from every file
	for i in range(2, 456):
		try:
			words = words + process(open("UTA/%i.txt" % i,"r").read()) + " "
		except:
			continue
    
    # split the string of every word into a list by whitespace
	words = words.split(" ")

    # fill the dictionary with unique word-value pairs
	value = 1
	for word in words:
		if word in d:
			pass
		else:
			d[word] = value
			value += 1
            
    # return the dictionary
	return d
