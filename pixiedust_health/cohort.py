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

# Hopefully find a better way to do this if we expand to other diseases
diseaseMap = {1: {'DisplayName': 'Diabetes',
                  'SnomedIds': ['44054006', '359642000', '81531005', '237599002', '199230006', '609567009', '237627000',
                                '9859006', '190331003', '703138006', '314903002', '314904008', '190390000', '314772004',
                                '314902007', '190389009', '313436004', '1481000119100'],
                  'loincIDs': [('HBA1C', '4548-4'), ('WEIGHT', '29463-7')]},
              2: {'DisplayName': 'Hypertension', 'SnomedIds': ['38341003']}}


# Class for storing data and generating DataFrames and matrices for UI and machine learning
# Requires 3 pandas data frames from create_v_demographic.csv, create_v_diagnosis.csv, and create_v_observation.csv
# Columns expected in demogaphics: 'EXPLORYS_PATIENT_ID', 'STD_GENDER', 'BIRTH_YEAR', 'STD_RACE'
# Columns expected in diagnosis: 'EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE', 'SNOMED_IDS'
# Columns expected in observations: 'EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID', 'STD_VALUE'
class Cohorts:
    # --------------------------------------- PRIVATE METHODS ---------------------------------------#

    ##-------------------------------------- Initialization --------------------------------------##

    #     def __init__(self, pathToData):
    def __init__(self, demographics, diagnosis, observations):
        #         demographics = pandas.read_csv(pathToData + '/create_v_demographic.csv', delimiter="\t")
        # print "demographics starting"
        self.demographics = self.__filterDemographics(demographics)
        # print "demographics done"

        #         diagnosis = pandas.read_csv(pathToData + '/create_v_diagnosis.csv', delimiter="\t")
        # print "histories starting"
        self.histories = self.__filterDiagnosis(diagnosis)
        #         self.diagnosis = self.__filterDiagnosis(diagnosis)
        #         self.histories, self.diagnosis = self.__filterDiagnosis(diagnosis)
        # print "histories done"

        #         observations = pandas.read_csv(pathToData + '/create_v_observation.csv', delimiter="\t")
        # print "observations starting"
        self.observations = observations
        self.demographics = self.__getFeatures(observations)
        # print "observations done"

    # Returns filtered demographics data frame
    # Requires original demographics chart
    # Used in initial processing of data
    def __filterDemographics(self, demographics):
        filteredDemographics = demographics[['EXPLORYS_PATIENT_ID', 'STD_GENDER', 'BIRTH_YEAR', 'STD_RACE']]
        ages = filteredDemographics['BIRTH_YEAR'].map(lambda x: 2017 - int(x))
        filteredDemographics['AGE'] = ages.values
        return filteredDemographics.drop('BIRTH_YEAR', 1)

    # Returns diagnosis history data frame
    # Requires original diagnosis histories chart
    # Used in initial processing of data
    def __filterDiagnosis(self, histories):
        filteredHistories = histories[['EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE', 'SNOMED_IDS']]
        snomedIDs = filteredHistories['SNOMED_IDS'].map(
            lambda x: tuple(x.split(',')) if isinstance(x, str) else tuple())
        filteredHistories['SNOMED_IDS'] = snomedIDs.values
        return filteredHistories

    # Returns the BMI of a patient given their information in demographics
    # Designed to be used in the apply function in getFeatures()
    def __bmi(self, row):
        return (row['WEIGHT'] / row['HEIGHT'] ** 2) * 10000

    # Returns filtered observation data frame
    # Requires original observations chart
    # Used in initial processing of data
    def __getFeatures(self, observations):
        demographicCopy = self.demographics
        filteredObservations = observations[['EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID', 'STD_VALUE']]
        filteredObservations = filteredObservations.groupby(['EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID'],
                                                            as_index=False).mean()
        loincIDs = [('HBA1C', '4548-4'), ('WEIGHT', '29463-7'), ('HEIGHT', '8302-2')]
        for label, loinc in loincIDs:
            justThisLabel = filteredObservations.loc[filteredObservations['LOINC_TEST_ID'] == loinc]
            justThisLabel = justThisLabel.rename(columns={'STD_VALUE': label})
            demographicCopy = demographicCopy.merge(justThisLabel.drop('LOINC_TEST_ID', 1))
        bmis = demographicCopy[['WEIGHT', 'HEIGHT']].apply(self.__bmi, axis=1)
        demographicCopy['BMI'] = bmis.values
        return demographicCopy

    ##------------------------ Utility functions for filtering by disease ------------------------##

    # Returns a list of all patients we have a medical history for
    # Facilitates getPatientsWithoutDisease()
    def __getPatients(self):
        return set(self.histories['EXPLORYS_PATIENT_ID'].values)

        # Returns list of IDs for patients with a given disease

    # Requires disease ID
    # Facilitates getDemographics() and used in machine learning component
    def __getPatientsWithDisease(self, diseaseID):
        snomedIDs = diseaseMap[diseaseID]['SnomedIds']
        filtered = self.histories.loc[
            [not (set(history).isdisjoint(snomedIDs)) for history in self.histories['SNOMED_IDS']]]
        return set(filtered['EXPLORYS_PATIENT_ID'].values)

    # Returns list of IDs for patients without a given disease
    # Requires disease ID
    # Facilitates getDemographics() and used in machine learning component
    def __getPatientsWithoutDisease(self, diseaseID):
        return self.__getPatients() - self.__getPatientsWithDisease(diseaseID)

    # --------------------------------------- PUBLIC METHODS ---------------------------------------#

    # Returns list of (DisplayName, ID) for all diseases we have data for
    # Designed for drop-down menu in initial window of UI
    # Example return: [('Diabetes', 1), ('Hypertension', 2)]
    def getDiseases(self):
        return [(value['DisplayName'], key) for key, value in iteritems(diseaseMap)]

    # Returns a dictionary of Data Frames, one for demographics and stats for patients with the given disease (key 'pos')
    #   and one for patients who don't have that disease (key 'neg')
    # Columns in returned data frames: 'EXPLORYS_PATIENT_ID', 'STD_GENDER', 'STD_RACE', 'AGE', 'HBA1C', 'WEIGHT', 'HEIGHT', 'BMI'
    # Had to calculate age  from birth year and BMI from weight and height
    # Designed for descriptive analytics window of UI and to use in getFeatureVectors()
    # This part is split into 2 separate data frames so the information shown in the pixieapp is just the people with the disease
    def getDemographics(self, diseaseID):
        pos_patients = self.__getPatientsWithoutDisease(diseaseID)
        neg_patients = self.__getPatientsWithDisease(diseaseID)
        pos_demographics = self.demographics.loc[self.demographics['EXPLORYS_PATIENT_ID'].isin(pos_patients)]
        neg_demographics = self.demographics.loc[self.demographics['EXPLORYS_PATIENT_ID'].isin(neg_patients)]
        return {'pos': pos_demographics, 'neg': neg_demographics}

    # Returns a data frame including both patients with and without given disease
    # HAS_DISEASE column indicates whether a patient has the disease or not
    # Returns only certain features if features are specified
    # Columns returned by default: 'STD_GENDER', 'STD_RACE', 'AGE', 'HBA1C', 'WEIGHT', 'HEIGHT', 'BMI', 'HAS_DISEASE'
    # Features argument must be some subset of the default columns
    # Out of the positive and negative patients, the bigger group is cut down to be the same size as the smaller group
    # Used in machine learning component
    def getFeatureVectors(self, diseaseID, features=None):
        dataFrames = self.getDemographics(diseaseID)
        size = min(len(dataFrames['pos'].index), len(dataFrames['neg'].index))
        for k in dataFrames:
            dataFrames[k] = dataFrames[k].drop('EXPLORYS_PATIENT_ID', 1)
            dataFrames[k] = dataFrames[k][:size]
            if features:
                dataFrames[k] = dataFrames[k][features]
            dataFrames[k]['HAS_DISEASE'] = k
        return pandas.concat([dataFrames['pos'], dataFrames['neg']])