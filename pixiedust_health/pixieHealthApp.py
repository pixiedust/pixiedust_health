# -------------------------------------------------------------------------------
# Copyright IBM Corp. 2017
# 
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------------

from pixiedust.display.app import *
from pixiedust.utils import Logger
from .cohort import *
import pandas as pd

@PixieApp
@Logger()
class PixieHealthApp():

    def getDialogOptions(self):
        return {
            'title': 'PixieHealthApp Demo',
            'maximize': 'true'
        }

    def selectedDisease(self, disease):
        self.debug('selectedDisease: {}'.format(disease))
        if disease is not None and self.alldiseases is not None:
            for d in self.alldiseases:
                if d[0] == disease:
                    self.selecteddisease = d
                    break

        if self.selecteddisease is not None:
            # TODO: get features, data for charting, etc (from API) for the selected disease
            # hard code values for now tuple of (feature, selected)
            self.diseasefeatures = [('STD_GENDER', 'true'), ('STD_RACE', 'true'), ('AGE', 'true')]
            self.dataframe1 = pd.DataFrame({'num1': range(5), 
                           'str1': ['a', 'b', 'c', 'd', 'e'],
                           'num2': range(2, 7),
                           'str2': ['w', 'v', 'x', 'y', 'z']})
            self.dataframe2 = pd.DataFrame({'num1': range(50, 55), 
                           'state1': ['MA', 'NC', 'MD', 'OK', 'CA']})

    def selectedFeatures(self, features):
        self.debug('selectedFeatures: {}'.format(features))
        selectedfeatures = features.split(',')

        # update feature selection
        if len(selectedfeatures) > 0:
            self.diseasefeatures = [(f, 'false') if f not in selectedfeatures else (f, 'true') for (f, s) in self.diseasefeatures]



    @route(page="classification")
    def page_classification(self):
        # TODO: update model (accuracy, recall, precision, ROC)
        # hard code values for now
        self.dataframe3 = pd.DataFrame({'num1': [10, 15, 3, 17, 16], 'str1': ['a', 'b', 'c', 'd', 'e']})
        modelapr = { 'accuracy': 80, 'precision': 70, 'recall': 75 }
        featureimportance = ['AGE', 'STD_GENDER']
        

        self._addHTMLTemplate('page-classification.html', disease=self.selecteddisease, features=self.diseasefeatures, modelapr=modelapr, featureimportance=featureimportance)

    @route(page="analytics")
    def page_analytics(self):
        self._addHTMLTemplate('page-analytics.html', disease=self.selecteddisease)

    @route()
    def page_start(self):
        # TODO: get list of diseases from API
        # hard code diseases for now
        self.alldiseases = [('Diabetes', 'xyz123'), ('Hypertension', 'abc789'), ('Alzheimer', 'lmn456')]

        self._addHTMLTemplate('page-start.html', diseases=self.alldiseases)
