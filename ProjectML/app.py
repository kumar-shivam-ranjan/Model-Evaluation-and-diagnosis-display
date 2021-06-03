import numpy as np
import pandas as pd
from flask_restful import Api
from db import db
import requests

import pickle
from sklearn import metrics
import csv
from flask import Flask,request,render_template,redirect,url_for, Response
from resources.evaluation import Evaluate, EvaluateList
from models.evaluation import EvalModel
from jinja2 import Template
from jinja2.filters import FILTERS, environmentfilter
@environmentfilter
def do_reverse_by_word(environment, value, attribute=None):
	k = [list(value.split('\\'))]
	return k[-1][-1]


FILTERS["reverse_by_word"] = do_reverse_by_word

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '#521637819082308ryfbbjdwd89'
api = Api(app)

@app.before_first_request
def create_tables():
	db.create_all()


@app.route("/")
def index_page():
	return render_template("index.html");

@app.route("/evaluatelist",methods=["GET"])  #Decorator
def dashboard():
	hostaddr = request.host
	r = requests.get('http://'+hostaddr+'/evaluate')
	evaluation_entities = r.json()['evaluation_entities']
	result = []
	for evaluation in evaluation_entities:
		item = evaluation
		result.append(item)
	return render_template('all_evaluations.html',entities=result)

@app.route("/neweval",methods=["GET","POST"])
def new_eval():
	if request.method == "GET":
		return render_template("evalform.html")
	else:
		hostaddr = request.host
		name = request.form['name']
		model_type = request.form['model_type']
		model_path = request.form['model_path']
		dataset_path = request.form['dataset_path']
		payload = {
			"name":name,
			"model_type":model_type,
			"model_path":model_path,
			"dataset_path":dataset_path
		}
		r = requests.post('http://'+hostaddr+'/evaluate',data=payload)
		return redirect(url_for("dashboard"))
	return {"message":"An error occured"}


@app.route("/evaluate/regression/<int:eval_id>")
def evaluate_regression(eval_id):
	hostaddr = request.host
	r = requests.get('http://'+hostaddr+'/evaluate/'+str(eval_id))
	eval_dict = r.json()
	metrics = eval_dict["metadata"]
	# print(metrics,type(metrics))
	name = eval_dict["name"]
	model_type = eval_dict["model_type"]
	if metrics:
		return render_template("evaluate_regression.html",
			name=name,
			model_type=model_type,
			mae=metrics["mean_absolute_error"],
			mse=metrics["mean_squared_error"],
			rmse=metrics["root_mean_squared_error"],
			rmsle=metrics["root_mean_squared_log_error"],
			r2=metrics["Coefficient_of_Determination"],
			ar2=metrics["Adjusted_r_squared"]
		)
	return {"message":"metrics are empty"}



@app.route("/evaluate/classification/<int:eval_id>")
def evaluate_classification(eval_id):
	hostaddr = request.host
	r = requests.get('http://'+hostaddr+'/evaluate/'+str(eval_id))
	eval_dict = r.json()
	metrics = eval_dict["metadata"]
	name = eval_dict["name"]
	model_type = eval_dict["model_type"]
	if metrics:
		return render_template("evaluate_classification.html",
			name=name,
			model_type=model_type,
			id=eval_id,
			acc=metrics["accuracy_score"],
			precision_score=metrics["precision_score"],
			recall=metrics["recall"],
			f1=metrics["f1-score"],
			log_loss=metrics["log_loss"]
		)
	return {"message":"metrics are empty"}



@app.route("/evaluate/classification/<int:eval_id>/auc")
def evaluate_classification_auc(eval_id):
	hostaddr = request.host
	r = requests.get('http://'+hostaddr+'/evaluate/'+str(eval_id))
	eval_dict = r.json()
	metrics = eval_dict["metadata"]
	name = eval_dict["name"]
	model_type = eval_dict["model_type"]
	if metrics:
		return render_template("evaluate_auc.html",
			name=name,
			model_type=model_type,
			id=eval_id,
			fpr=metrics["fpr"],
			tpr=metrics["tpr"],
			auc=metrics["roc_auc"]
		)
	return {"message":"metrics are empty"}

@app.route("/evaluate/classification/<int:eval_id>/cmatrix")
def evaluate_confusion_matrix(eval_id):
	hostaddr = request.host
	r = requests.get('http://'+hostaddr+'/evaluate/'+str(eval_id))
	eval_dict = r.json()
	metrics = eval_dict["metadata"]
	name = eval_dict["name"]
	model_type = eval_dict["model_type"]
	if metrics:
		print(metrics['confusion_matrix'])
		return render_template("evaluate_confusion_matrix.html",
			name=name,
			model_type=model_type,
			id=eval_id,
			cmatrix=metrics['confusion_matrix']
		)
	return {"message":"metrics are empty"}


api.add_resource(Evaluate,"/evaluate/<int:eval_id>")
api.add_resource(EvaluateList,"/evaluate")

if __name__=="__main__":
	db.init_app(app)
	app.run(debug=True)
