import sqlite3
from flask_restful import Resource, reqparse
from models.evaluation import EvalModel
from resources.eval_functions import EvaluationFunctions
from flask import Flask,request,render_template,redirect,url_for
class Evaluate(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('model_path',
        type=str,
        required=True,
        help="Please provide a model path"
    )
    parser.add_argument('dataset_path',
        type=str,
        required=True,
        help="Please provide a datset path"
    )
    parser.add_argument('model_type',
        type=str,
        required=True,
        help="Please define the type of model"
    )
    parser.add_argument('name',
        type=str,
        required=True,
        help="Please define the name of model"
    )

    def get(self,eval_id):
        evaluation_entity = EvalModel.find_by_id(eval_id)
        if evaluation_entity:
            eval_dict = evaluation_entity.json()
            evaluation_object = EvaluationFunctions(eval_dict['model_type'], eval_dict['model_path'], eval_dict['dataset_path'])
            if eval_dict['model_type'] == 'regression':
                return evaluation_object.evaluate_regression()
            else:
                return evaluation_object.evaluate_classification()
        return {"message":"Requested evaluation entity doesn't exist"}, 404


class EvaluateList(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('model_path',
        type=str,
        required=True,
        help="Please provide a model path"
    )
    parser.add_argument('dataset_path',
        type=str,
        required=True,
        help="Please provide a datset path"
    )
    parser.add_argument('model_type',
        type=str,
        required=True,
        help="Please define the type of model"
    )
    parser.add_argument('name',
        type=str,
        required=True,
        help="Please define the name of model"
    )
    def get(self):
        return {"evaluation_entities":[x.json() for x in EvalModel.query.all()]}

    def post(self):
        data = EvaluateList.parser.parse_args()

        item = EvalModel(**data)

        try:
            item.save_to_db()
        except:
            return {"message":"An error occured inserting the evaluation"}, 500

        return item.json(), 201
