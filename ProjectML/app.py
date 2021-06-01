import numpy as np
import pandas as pd
from flask_restful import Api
from db import db

import pickle
from sklearn import metrics
import csv
from flask import Flask,request,render_template,redirect,url_for
from resources.evaluation import Evaluate, EvaluateList
from models.evaluation import EvalModel

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '#521637819082308ryfbbjdwd89'
api = Api(app)

@app.before_first_request
def create_tables():
    db.create_all()

# import os
# cwd=os.getcwd()
# print(cwd)
# app.config["UPLOAD_PATH"]=cwd

@app.route("/")
def index_page():
	return render_template("index.html");

# @app.route("/classification",methods=["GET","POST"])  #Decorator
# def home_page():
# 	if request.method=='POST':
# 		f1=request.files['modelname']
# 		f2=request.files['test_data']
# 		f1.save(os.path.join(app.config['UPLOAD_PATH'],f1.filename))
# 		f2.save(os.path.join(app.config['UPLOAD_PATH'],f2.filename))
# 		return redirect(url_for("evaluate"))
# 	else:
# 		return render_template("classification.html",msg="Upload your files")
#
# @app.route('/regression',methods=["GET","POST"])
# def regression():
# 	if request.method=='POST':
# 		f1=request.files['modelname']
# 		f2=request.files['test_data']
# 		f1.save(os.path.join(app.config['UPLOAD_PATH'],f1.filename))
# 		f2.save(os.path.join(app.config['UPLOAD_PATH'],f2.filename))
# 		return redirect(url_for("evaluate"))
# 	else:
# 		return render_template("regression.html",msg="Upload your files")

api.add_resource(Evaluate,"/evaluate/<int:eval_id>")
api.add_resource(EvaluateList,"/evaluate")

if __name__=="__main__":
	db.init_app(app)
	app.run(debug=True)