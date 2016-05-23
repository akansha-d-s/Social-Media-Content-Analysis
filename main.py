import util
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
import sys
import os
import glob

def main():
	init()

	# get the dataset 
	print colored("Enter path to the dataset?", 'cyan', attrs=['bold'])
	ans = sys.stdin.readline()
	# remove any newlines or spaces at the end of the input
	path = ans.strip('\n')
	if path.endswith(' '):
		path = path.rstrip(' ')

	# preprocess data into two folders instead of 6
	reorganize_dataset(path)


	print '\n\n'

	# do the main test
	main_test(path)

def reorganize_dataset(path):
	appropriate = ['rec.sport.hockey', 'sci.crypt', 'sci.electronics']
	inappropriate = ['sci.space', 'rec.motorcycles', 'misc.forsale']

	folders = glob.glob(path + '/*')
	if len(folders) == 2:
		return
	else:
		# create `appropriate` and `inappropriate` directories
		if not os.path.exists(path + '/' + 'appropriate'):
			os.makedirs(path + '/' + 'appropriate')
		if not os.path.exists(path + '/' + 'inappropriate'):
			os.makedirs(path + '/' + 'inappropriate')

		for like in appropriate:
			files = glob.glob(path + '/' + like + '/*')
			for f in files:
				parts = f.split('/')
				name = parts[len(parts) -1]
				newname = like + '_' + name
				os.rename(f, path+'/appropriate/'+newname)
			os.rmdir(path + '/' + like)
		
		for like in inappropriate:
			files = glob.glob(path + '/' + like + '/*')
			for f in files:
				parts = f.split('/')
				name = parts[len(parts) -1]
				newname = like + '_' + name
				os.rename(f, path+'/inappropriate/'+newname)
			os.rmdir(path + '/' + like)

def main_test(path = None):
	dir_path = path or 'dataset'

	remove_incompatible_files(dir_path)

	print '\n\n'

	# load data
	print colored('Loading files into memory', 'green', attrs=['bold'])
	files = sklearn.datasets.load_files(dir_path)

	# refine all emails
	print colored('Refining all files', 'green', attrs=['bold'])
	util.refine_all_emails(files.data)

	# calculate the BOW representation
	print colored('Calculating BOW', 'green', attrs=['bold'])
	word_counts = util.bagOfWords(files.data)

	# TFIDF
	print colored('Calculating TFIDF', 'green', attrs=['bold'])
	tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
	X = tf_transformer.transform(word_counts)


	print '\n\n'

	# create classifier
	# clf = sklearn.naive_bayes.MultinomialNB()
	# clf = sklearn.svm.LinearSVC()
	n_neighbors = 11
	weights = 'uniform'
	weights = 'distance'
	clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

	# test the classifier
	print '\n\n'
	print colored('Testing classifier with train-test split', 'magenta', attrs=['bold'])
	test_classifier(X, files.target, clf, test_size=0.2, y_names=files.target_names, confusion=False)

def remove_incompatible_files(dir_path):
	# find incompatible files
	incompatible_files = util.find_incompatible_files(dir_path)


	# delete them
	if(len(incompatible_files) > 0):

		util.delete_incompatible_files(incompatible_files)

def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
	#train-test split
	print 'test size is: %2.0f%%' % (test_size*100)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)

	if not confusion:
		print colored('Classification report:', 'magenta', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)

if __name__ == '__main__':
	main()
