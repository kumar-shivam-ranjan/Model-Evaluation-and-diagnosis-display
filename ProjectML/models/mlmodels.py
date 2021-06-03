from db import db
from sqlalchemy.dialects.postgresql import JSON

class MLModel(db.Model):
    __tablename__ = "Models"

    # model_id, model_name, model_path, created
    model_id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(80))
    model_path = db.Column(db.String(200))
    created = db.Column() # insert timestamp datatype

    # Insert the foreign key relationship

    def __init__(self):
        pass
