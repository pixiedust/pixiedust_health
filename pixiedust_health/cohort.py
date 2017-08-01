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

# Hopefully find a better way to do this
diseaseMap = {1: {'DisplayName': 'Diabetes',
                  'SnomedIds': ['44054006', '359642000', '81531005', '237599002', '199230006', '609567009', '237627000',
                                '9859006', '190331003', '703138006', '314903002', '314904008', '190390000', '314772004',
                                '314902007', '190389009', '313436004', '1481000119100']},
              2: {'DisplayName': 'Hypertension', 'SnomedIds': ['38341003']}}


# Class for storing data and generating DataFrames and matrices for UI and machine learning
class Cohorts:
    def __init__(self, pathToData):
        demographics = pandas.read_csv(pathToData + '/create_v_demographic.csv', delimiter="\t")
        self.demographics = self.__filterDemographics(demographics)

        diagnosis = pandas.read_csv(pathToData + '/create_v_diagnosis.csv', delimiter="\t")
        self.histories = self.__filterHistories(diagnosis)
        self.diagnosis = self.__filterDiagnosis(diagnosis)

        observations = pandas.read_csv(pathToData + '/create_v_observation.csv', delimiter="\t")
        self.demographics = self.__getFeatures(observations)

    # Returns filtered preliminary demographics data frame
    # Requires original demographics chart
    def __filterDemographics(self, demographics):
        filteredDemographics = demographics[['EXPLORYS_PATIENT_ID', 'STD_GENDER', 'BIRTH_YEAR', 'STD_RACE']]
        ages = filteredDemographics['BIRTH_YEAR'].map(lambda x: 2017 - int(x))
        filteredDemographics['AGE'] = ages.values
        return filteredDemographics.drop('BIRTH_YEAR', 1)

    # Returns data frame of diseases each patient has been diagnosed with
    # Requires original diagnosis histories chart
    def __filterHistories(self, histories):
        filteredHistories = histories[['EXPLORYS_PATIENT_ID', 'SNOMED_IDS']]
        snomedIDs = filteredHistories['SNOMED_IDS'].map(
            lambda x: tuple(x.split(',')) if isinstance(x, str) else tuple())
        filteredHistories['SNOMED_IDS'] = snomedIDs.values
        filteredHistories = filteredHistories.groupby(['EXPLORYS_PATIENT_ID'], as_index=False).sum()
        return filteredHistories

    # Returns diagnosis history data frame
    # Requires original diagnosis histories chart
    def __filterDiagnosis(self, histories):
        filteredHistories = histories[['EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE', 'SNOMED_IDS']]
        snomedIDs = filteredHistories['SNOMED_IDS'].map(
            lambda x: tuple(x.split(',')) if isinstance(x, str) else tuple())
        filteredHistories['SNOMED_IDS'] = snomedIDs.values
        filteredHistories = filteredHistories.groupby(['EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE'], as_index=False).sum()
        return filteredHistories

    # Returns filtered observation data frame
    # Requires original observations chart
    def __getFeatures(self, observations):
        demographicCopy = self.demographics
        filteredObservations = observations[['EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID', 'STD_VALUE']]
        filteredObservations = filteredObservations.groupby(['EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID'],
                                                            as_index=False).mean()
        loincIDs = [('HBA1C', '4548-4'), ('WEIGHT', '29463-7')]
        for label, loinc in loincIDs:
            justThisLabel = filteredObservations.loc[filteredObservations['LOINC_TEST_ID'] == loinc]
            justThisLabel = justThisLabel.rename(columns={'STD_VALUE': label})
            demographicCopy = demographicCopy.merge(justThisLabel.drop('LOINC_TEST_ID', 1))
        return demographicCopy

    # Returns list of (DisplayName, ID) for all diseases we have data for
    # Designed for drop-down menu in initial window of UI
    def getDiseases(self):
        return [(value['DisplayName'], key) for key, value in iteritems(diseaseMap)]

    # Returns Data Frame with demographics for each patient
    # Filters by patients with a certain disease if disease ID is given
    # Designed for descriptive analytics window of UI
    def getDemographics(self, diseaseID=None, negative=False):
        if diseaseID:
            if negative:
                patients = self.getPatientsWithoutDisease(diseaseID)
            else:
                patients = self.getPatientsWithDisease(diseaseID)
            # Cache last dataframe generated
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

    # Returns list of IDs for patients without a given disease
    # Requires disease ID
    # Facilitates getDemographics()
    def getPatientsWithoutDisease(self, diseaseID):
        snomedIDs = diseaseMap[diseaseID]['SnomedIds']
        filtered = self.histories.loc[[set(history).isdisjoint(snomedIDs) for history in self.histories['SNOMED_IDS']]]
        return list(set(filtered['EXPLORYS_PATIENT_ID'].values))

    # Returns a numpy matrix of features
    # Filters by a certain disease if disease ID is given
    # Returns only certain features if features are specified
    def getFeatureVectors(self, diseaseID=None, features=None, negative=False):
        ret = self.getDemographics(diseaseID, negative)
        if features:
            return ret[features].as_matrix()
        return ret[['STD_GENDER', 'STD_RACE', 'AGE', 'HBA1C', 'WEIGHT']].as_matrix()