import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import metrics
import csv
from flask import Flask,request,render_template,redirect,url_for
app=Flask(__name__)


import os
cwd=os.getcwd()
print(cwd)
app.config["UPLOAD_PATH"]=cwd

@app.route("/index")
def index_page():
    	return render_template("indexx.html");

@app.route("/classification",methods=["GET","POST"])  #Decorator
def home_page():
	if request.method=='POST':
		f1=request.files['modelname']
		f2=request.files['test_data']
		f1.save(os.path.join(app.config['UPLOAD_PATH'],f1.filename))
		f2.save(os.path.join(app.config['UPLOAD_PATH'],f2.filename))
		return redirect(url_for("evaluate"))
	else:
		return render_template("index.html",msg="Upload your files")


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
	feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
	x=pima[feature_cols]
	y_pred=loaded_model.predict(x)
	y_actual=pima.Outcome
	acc=metrics.accuracy_score(y_actual,y_pred)
	precision_score=metrics.precision_score(y_actual,y_pred)
	recall=metrics.recall_score(y_actual,y_pred)
	f1=metrics.f1_score(y_actual,y_pred)
	r2=metrics.r2_score(y_actual,y_pred)
	return render_template("evaluate.html",msg="files have been Uploaded",a=acc,p=precision_score,r=recall,f1=f1,r2=r2)




app.run(debug=True)