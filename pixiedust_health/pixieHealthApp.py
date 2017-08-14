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
            self.demographicDF = self.cohorts.getDemographics(self.selecteddisease[1])['pos']
            self.geodf = self.cohorts.geoFormatPostal(self.selecteddisease[1])
            self.allfeaturesDF = self.cohorts.getFeatureVectors(self.selecteddisease[1])
            self.selectedfeatureDF = self.allfeaturesDF
            self.allfeaturesnames = self.cohorts.getFeatureNames(self.allfeaturesDF)


    def selectedFeatures(self, features):
        self.debug('selectedFeatures: {}'.format(features))
        selectedfeatures = features.split(',')

        if len(selectedfeatures) > 0:
            self.selectedfeatureDF = self.cohorts.getFeaturesForModel(self.allfeaturesDF, selectedfeatures)


    @route(page="classification")
    def page_classification(self):
        x_train, y_train, x_test, y_test = self.cohorts.getTrainTestSets(self.selectedfeatureDF)
        featurenames = self.cohorts.getFeatureNames(self.selectedfeatureDF)
        clf, y_preds = self.cohorts.getRandomForestClassifier(x_train, y_train, x_test, featurenames)

        accuracy, precision, recall = self.cohorts.getClassifierMetrics(clf, y_test, y_preds)
        modelapr = { 'accuracy': accuracy, 'precision': precision, 'recall': recall }
        featureimportance = self.cohorts.featureImportance(clf, x_train, featurenames)

        featuresbyselection = [(f, 'false') if f not in featurenames else (f, 'true') for f in self.allfeaturesnames]

        self._addHTMLTemplate('page-classification.html', disease=self.selecteddisease[0], features=featuresbyselection, modelapr=modelapr, featureimportance=featureimportance)


    @route(page="analytics")
    def page_analytics(self):
        self._addHTMLTemplate('page-analytics.html', disease=self.selecteddisease[0])


    @route()
    def page_start(self):
        self.cohorts = self.pixieapp_entity
        self.alldiseases = self.cohorts.getDiseases()

        self._addHTMLTemplate('page-start.html', diseases=self.alldiseases)
