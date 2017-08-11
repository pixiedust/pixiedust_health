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
            #self.diseasefeatures = [('STD_GENDER', 'true'), ('STD_RACE', 'true'), ('AGE', 'true')]
            self.demographicDF = self.cohorts.getDemographics(self.selecteddisease[1])['pos']
            # self.dataframe1 = pd.DataFrame({'num1': range(5), 
            #                'str1': ['a', 'b', 'c', 'd', 'e'],
            #                'num2': range(2, 7),
            #                'str2': ['w', 'v', 'x', 'y', 'z']})
            # self.dataframe2 = pd.DataFrame({'num1': range(50, 55), 
            #                'state1': ['MA', 'NC', 'MD', 'OK', 'CA']})
            self.geodf = self.cohorts.geoFormatPostal(self.selecteddisease[1])
            self.featureDF = self.cohorts.getFeatureVectors(self.selecteddisease[1])
            self.selectedfeatureDF = self.featureDF

    def selectedFeatures(self, features):
        self.debug('selectedFeatures: {}'.format(features))
        selectedfeatures = features.split(',')

        # update feature selection
        if len(selectedfeatures) > 0:
            self.diseasefeatures = [(f, 'false') if f not in selectedfeatures else (f, 'true') for (f, s) in self.diseasefeatures]
            self.selectedfeatureDF = self.cohorts.getFeaturesForModel(self.featureDF, selectedfeatures)


    @route(page="classification")
    def page_classification(self):
        # TODO: update model (accuracy, recall, precision, ROC)
        # hard code values for now
        self.x_train, self.y_train, self.x_test, self.y_test = self.cohorts.getTrainTestSets(self.selectedfeatureDF)
        self.featurenames = self.cohorts.getFeatureNames(self.selectedfeatureDF)
        self.clf, self.y_preds = self.cohorts.getRandomForestClassifier(self.x_train, self.y_train, self.x_test, self.featurenames)
        self.featureimportance = self.cohorts.featureImportance(self.clf, self.x_train, self.featurenames)
        self.accuracy, self.precision, self.recall = self.cohorts.getClassifierMetrics(self.clf, self.y_test, self.y_preds)

        #self.dataframe3 = pd.DataFrame({'num1': [10, 15, 3, 17, 16], 'str1': ['a', 'b', 'c', 'd', 'e']})
        modelapr = { 'accuracy': self.accuracy, 'precision': self.precision, 'recall': self.recall }
        #featureimportance = ['AGE', 'STD_GENDER']
        

        self._addHTMLTemplate('page-classification.html', disease=self.selecteddisease, features=self.diseasefeatures, modelapr=modelapr, featureimportance=featureimportance)

    @route(page="analytics")
    def page_analytics(self):
        self._addHTMLTemplate('page-analytics.html', disease=self.selecteddisease)

    @route()
    def page_start(self):
        # TODO: get list of diseases from API
        # hard code diseases for now
        self.cohorts = self.pixieapp_entity
        self.alldiseases = [('Diabetes', 1), ('Hypertension', 2), ('Alzheimer', 3)]
        self.diseasefeatures =[('HEIGHT', 'true'), ('WEIGHT', 'true'), ('HBA1C', 'true'), ('STD_GENDER', 'true'), ('AGE', 'true'), ('STD_ETHNICITY', 'true'), ('STD_RACE', 'true'), ('BMI', 'true')]
        #self.diseasefeatures = [('STD_GENDER', 'true'), ('STD_RACE', 'true'), ('AGE', 'true')]
        self._addHTMLTemplate('page-start.html', diseases=self.alldiseases)
