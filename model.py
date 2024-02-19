import os
from fastai.vision.all import *
from label_studio_ml.model import LabelStudioMLBase
from fastai.vision.widgets import *
from fastbook import *
import json
import numpy


class ImageClassifierAPI(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(ImageClassifierAPI, self).__init__(**kwargs)

        self.image_model = load_learner(self.path/'export.pkl')

        # self.label_map = {
        #     1: "Positive",
        #     0: "Negative"}

    def predict(self, tasks, **kwargs):
        predictions = []
   
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config
        #  to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
        data_name = schema['inputs'][0]['value']

        for task in tasks:
            # load the data and make a prediction with the model
            image_path = task['data'][data_name]
            ans = self.image_model.predict(image_path)
            label, index = ans[0], ans[1].numpy()
            print(label, index)

            prediction = {
                    'score': float(index),
                    'result': [{
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'choices',
                        'value': {
                            'choices': [
                                label
                            ]
                        },
                    }]
                }

            predictions.append(prediction)
        return predictions

  
