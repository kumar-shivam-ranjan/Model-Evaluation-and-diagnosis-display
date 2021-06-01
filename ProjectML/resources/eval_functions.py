from flask import Flask,request,render_template,redirect,url_for
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import csv
class EvaluationFunctions():
	def __init__(self, model_type, model_path, dataset_path):
		self.model_path = model_path
		self.model_type = model_type
		self.dataset_path = dataset_path

	def evaluate_classification(self):
		model_file=self.model_path
		dataset_file=self.dataset_path

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

		return {"accuracy_score":acc,
		"precision_score":precision_score,
		"recall":recall,
		"f1-score":f1,
		"log_loss":log_loss}

	def evaluate_regression(self):
		model_file=self.model_path
		dataset_file=self.dataset_path

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

		return {
			"Coefficient_of_Determination":r2,
			"Adjusted_r_squared":ar2,
			"mean_absolute_error":mae,
			"mean_squared_error":mse,
			"root_mean_squared_error":rmse,
			"root_mean_squared_log_error":rmsle
		}
