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

from six import iteritems
import pandas

# Hopefully find a better way to do this
diseaseMap = {1: {'DisplayName': 'Diabetes',
                  'SnomedIds': ['44054006', '359642000', '81531005', '237599002', '199230006', '609567009', '237627000',
                                '9859006', '190331003', '703138006', '314903002', '314904008', '190390000', '314772004',
                                '314902007', '190389009', '313436004', '1481000119100']},
              2: {'DisplayName': 'Hypertension', 'SnomedIds': ['38341003']}}


# Class for storing data and generating DataFrames and matrices for UI and machine learning
class Cohorts:
    def __init__(self, pathToData):
        self.demographics = pandas.read_csv(pathToData + '/create_v_demographic.csv', delimiter="\t")
        self.__filterDemographics()
        self.histories = pandas.read_csv(pathToData + '/create_v_medical_history.csv', delimiter="\t")
        self.__filterHistories()

    # Filter preliminary demographics data frame
    # No input or output
    def __filterDemographics(self):
        filteredDemographics = self.demographics[['EXPLORYS_PATIENT_ID', 'STD_GENDER', 'BIRTH_YEAR', 'STD_RACE']]
        ages = filteredDemographics['BIRTH_YEAR'].map(lambda x: 2017 - int(x))
        filteredDemographics['AGE'] = ages.values
        self.demographics = filteredDemographics.drop('BIRTH_YEAR', 1)

    # Filter preliminary medical history data frame
    # No input or output
    def __filterHistories(self):
        filteredHistories = self.histories[['EXPLORYS_PATIENT_ID', 'SNOMED_IDS']]
        snomedIDs = filteredHistories['SNOMED_IDS'].map(lambda x: tuple(x.split(',')))
        filteredHistories['SNOMED_IDS'] = snomedIDs.values
        self.histories = filteredHistories

    # Returns list of (DisplayName, ID) for all diseases we have data for
    # Designed for drop-down menu in initial window of UI
    def getDiseases(self):
        return [(value['DisplayName'], key) for key, value in iteritems(diseaseMap)]

    # Returns Data Frame with demographics for each patient
    # Filters by patients with a certain disease if disease ID is given
    # Designed for descriptive analytics window of UI
    def getDemographics(self, diseaseID=None):
        if diseaseID:
            patients = self.getPatientsWithDisease(diseaseID)
            return self.demographics.loc[self.demographics['EXPLORYS_PATIENT_ID'].isin(patients)]
        return self.demographics

    # Returns list of IDs for patients with a given disease
    # Requires disease ID
    # Facilitates getDemographics()
    def getPatientsWithDisease(self, diseaseID):
        snomedIDs = diseaseMap[diseaseID]['SnomedIds']
        filtered = self.histories.loc[
            [not (set(history).isdisjoint(snomedIDs)) for history in self.histories['SNOMED_IDS']]]
        return list(set(filtered['EXPLORYS_PATIENT_ID'].values))

    # Returns a numpy matrix of features
    # Filters by a certain disease if disease ID is given
    # Returns only certain features if features are specified
    def getFeatureMatrix(self, diseaseID=None, features=None):
        ret = self.getDemographics(diseaseID)
        if features:
            return ret[features].as_matrix()
        return ret[['STD_GENDER', 'STD_RACE', 'AGE']].as_matrix()
