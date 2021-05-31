import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import metrics
import csv
from flask import Flask,request,render_template,redirect,url_for,json
app=Flask(__name__)


import os
cwd=os.getcwd()
print(cwd)
app.config["UPLOAD_PATH"]=cwd

@app.route("/index")
def index_page():
		return render_template("index.html");

@app.route("/classification",methods=["GET","POST"])  #Decorator
def home_page():
	if request.method=='POST':
		f1=request.files['modelname']
		f2=request.files['test_data']
		f1.save(os.path.join(app.config['UPLOAD_PATH'],f1.filename))
		f2.save(os.path.join(app.config['UPLOAD_PATH'],f2.filename))
		return redirect(url_for("evaluate"))
	else:
		return render_template("classification.html",msg="Upload your files")


@app.route("/evaluate")
def evaluate():
	model_file=''
	dataset_file=''
	for filename in os.listdir(cwd):
		if filename.endswith(".sav"):
			model_file=filename
		if filename.endswith(".csv"):
			dataset_file=filename

	print(model_file)
	print(dataset_file)

	with open(dataset_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		list_of_column_names = []
		for row in csv_reader:
			list_of_column_names.append(row)
			break
	print(list_of_column_names[0])
	loaded_model = pickle.load(open(model_file, 'rb'))
	col_names = list_of_column_names[0]
	pima=pd.read_csv(dataset_file,header=None,names=col_names,skiprows=1)
	feature_cols=col_names[0:-1]
	label=col_names[-1]
	x=pima[feature_cols]
	y_pred=loaded_model.predict(x)
	probs=loaded_model.predict_proba(x)
	y_actual=pima[label]
	acc=metrics.accuracy_score(y_actual,y_pred)
	precision_score=metrics.precision_score(y_actual,y_pred)
	recall=metrics.recall_score(y_actual,y_pred)
	f1=metrics.f1_score(y_actual,y_pred)
	log_loss=metrics.log_loss(y_actual,probs)
	data=[]
	data.append(acc)
	data.append(precision_score)
	data.append(recall)
	data.append(f1)
	data.append(log_loss)
	labels=["Accuracy","Precision","Recall","F1 Score","Logg Loss"]

	for filename in os.listdir(cwd):
		if filename.endswith(".sav"):
			os.remove(filename)
		if filename.endswith(".csv"):
			os.remove(filename)
	return render_template("evaluate.html",msg="files have been Uploaded",data=data,labels=labels)


@app.route('/regression',methods=["GET","POST"])
def regression():
	if request.method=='POST':
		f1=request.files['modelname']
		f2=request.files['test_data']
		f1.save(os.path.join(app.config['UPLOAD_PATH'],f1.filename))
		f2.save(os.path.join(app.config['UPLOAD_PATH'],f2.filename))
		return redirect(url_for("evaluate_regression"))
	else:
		return render_template("regression.html",msg="Upload your files")

@app.route('/evaluate_regression')
def evaluate_regression():
	model_file=''
	dataset_file=''
	for filename in os.listdir(cwd):
		if filename.endswith(".sav"):
			model_file=filename
		if filename.endswith(".csv"):
			dataset_file=filename

	print(model_file)
	print(dataset_file)

	with open(dataset_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		list_of_column_names = []
		for row in csv_reader:
			list_of_column_names.append(row)
			break
	print(list_of_column_names[0])
	loaded_model = pickle.load(open(model_file, 'rb'))
	col_names = list_of_column_names[0]
	dataset=pd.read_csv(dataset_file,header=None,names=col_names,skiprows=1)

	feature_cols= ['ZN', 'INDUS', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'LSTAT']
	label=col_names[-1]

	print(feature_cols)
	print(label)
	X = dataset.loc[:,feature_cols]
	print(X.shape)
	y_test_pred=loaded_model.predict(X)
	y_test=dataset[label]

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler()
	y_test_pred_rs = y_test_pred.reshape(-1,1)

	scaler.fit(y_test_pred_rs)
	y_test_pred_scaled = scaler.transform(y_test_pred_rs)
	k = y_test_pred_scaled.shape
	y_pred = y_test_pred_scaled.reshape(k[0])

	y_test_np = np.asarray(y_test)
	y_test_rs = y_test_np.reshape(-1,1)

	scaler.fit(y_test_rs)
	y_test_scaled = scaler.transform(y_test_rs)
	k = y_test_scaled.shape
	y_test_new = y_test_scaled.reshape(k[0])

	# Metrics
	r2 = metrics.r2_score(y_test, y_test_pred)
	ar2 = 1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)
	mae = metrics.mean_absolute_error(y_test, y_test_pred)
	mse = metrics.mean_squared_error(y_test, y_test_pred)
	rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
	rmsle = np.sqrt(metrics.mean_squared_log_error( y_test_new, y_pred ))

	for filename in os.listdir(cwd):
		if filename.endswith(".sav"):
			os.remove(filename)
		if filename.endswith(".csv"):
			os.remove(filename)
	return render_template("evaluate_regression.html",msg="files have been Uploaded",mae=mae,mse=mse,rmse=rmse,rmsle=rmsle,r2=r2,ar2=ar2)



app.run(debug=True)
