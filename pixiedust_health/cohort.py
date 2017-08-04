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
# Requires 3 pandas data frames from create_v_demographic.csv, create_v_diagnosis.csv, and create_v_observation.csv
# List of columns expected in demogaphics: 'EXPLORYS_PATIENT_ID', 'STD_GENDER', 'BIRTH_YEAR', 'STD_RACE'
# Columns expected in diagnosis: 'EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE', 'SNOMED_IDS'
# Columns expected in observations: 'EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID', 'STD_VALUE'
class Cohorts:
    #     def __init__(self, pathToData):
    def __init__(self, demographics, diagnosis, observations):
        #         demographics = pandas.read_csv(pathToData + '/create_v_demographic.csv', delimiter="\t")
        print "demographics starting"
        self.demographics = self.__filterDemographics(demographics)
        print "demographics done"

        #         diagnosis = pandas.read_csv(pathToData + '/create_v_diagnosis.csv', delimiter="\t")
        print "histories starting"
        self.histories = self.__filterDiagnosis(diagnosis)
        #         self.diagnosis = self.__filterDiagnosis(diagnosis)
        #         self.histories, self.diagnosis = self.__filterDiagnosis(diagnosis)
        print "histories done"

        #         observations = pandas.read_csv(pathToData + '/create_v_observation.csv', delimiter="\t")
        print "observations starting"
        self.observations = observations
        self.demographics = self.__getFeatures(observations)
        print "observations done"

    # Returns filtered demographics data frame
    # Requires original demographics chart
    def __filterDemographics(self, demographics):
        filteredDemographics = demographics[['EXPLORYS_PATIENT_ID', 'STD_GENDER', 'BIRTH_YEAR', 'STD_RACE']]
        ages = filteredDemographics['BIRTH_YEAR'].map(lambda x: 2017 - int(x))
        filteredDemographics['AGE'] = ages.values
        return filteredDemographics.drop('BIRTH_YEAR', 1)

    # Returns diagnosis history data frame
    # Requires original diagnosis histories chart
    def __filterDiagnosis(self, histories):
        filteredHistories = histories[['EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE', 'SNOMED_IDS']]
        snomedIDs = filteredHistories['SNOMED_IDS'].map(
            lambda x: tuple(x.split(',')) if isinstance(x, str) else tuple())
        filteredHistories['SNOMED_IDS'] = snomedIDs.values
        return filteredHistories

    def __bmi(self, row):
        return (row['WEIGHT'] / row['HEIGHT'] ** 2) * 10000

    # Returns filtered observation data frame
    # Requires original observations chart
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

    # Returns a list of all patients we have a medical history for
    # Facilitates getPatientsWithoutDisease()
    def __getPatients(self):
        return set(self.histories['EXPLORYS_PATIENT_ID'].values)

    # Returns list of IDs for patients with a given disease
    # Requires disease ID
    # Facilitates getDemographics()
    def __getPatientsWithDisease(self, diseaseID):
        snomedIDs = diseaseMap[diseaseID]['SnomedIds']
        filtered = self.histories.loc[
            [not (set(history).isdisjoint(snomedIDs)) for history in self.histories['SNOMED_IDS']]]
        return set(filtered['EXPLORYS_PATIENT_ID'].values)

    # Returns list of IDs for patients without a given disease
    # Requires disease ID
    # Facilitates getDemographics()
    def __getPatientsWithoutDisease(self, diseaseID):
        return self.__getPatients() - self.__getPatientsWithDisease(diseaseID)

    # Returns a Data Frame of diagnosis dates and corresponding snomed IDs for a given patient
    # Requires a patient ID
    # Facilitates getYearPriorToDiagnosis()
    def __getDiagnosisHistory(self, patientID):
        patientHistory = self.histories.loc[self.histories['EXPLORYS_PATIENT_ID'] == patientID].drop(
            'EXPLORYS_PATIENT_ID', 1)
        return patientHistory.groupby('DIAGNOSIS_DATE', as_index=False).sum()

    # Returns list of (DisplayName, ID) for all diseases we have data for
    # Designed for drop-down menu in initial window of UI
    # Example return: [('Diabetes', 1), ('Hypertension', 2)]
    def getDiseases(self):
        return [(value['DisplayName'], key) for key, value in iteritems(diseaseMap)]

    # Returns Data Frame with demographics for each patient
    # Filters by patients with a certain disease if disease ID is given
    # Designed for descriptive analytics window of UI
    def getDemographics(self, diseaseID, negative=False):
        if negative:
            patients = self.__getPatientsWithoutDisease(diseaseID)
        else:
            patients = self.__getPatientsWithDisease(diseaseID)
        # Cache last dataframe generated?
        return self.demographics.loc[self.demographics['EXPLORYS_PATIENT_ID'].isin(patients)]

    # Returns a numpy matrix of features
    # Filters by a certain disease if disease ID is given
    # Returns only certain features if features are specified
    # Designed to be used for machine learning component
    # Matrix dimensions: number of patients with disease x number of features given (default 5)
    def getFeatureVectors(self, diseaseID, features=None, negative=False):
        ret = self.getDemographics(diseaseID, negative).drop('EXPLORYS_PATIENT_ID', 1)
        if features:
            return ret[features].as_matrix()
        return ret.as_matrix()

    # Returns a Data Frame of diagnosis dates and corresponding snomed IDs for a given patient
    # Requires a patient ID
    # Facilitates getMosPriorToDiagnosis()
    def getDiagnosisHistory(self, patientID):
        patientHistory = self.histories.loc[self.histories['EXPLORYS_PATIENT_ID'] == patientID].drop(
            'EXPLORYS_PATIENT_ID', 1)
        return patientHistory.groupby('DIAGNOSIS_DATE', as_index=False).sum()

    # Unfinished function being worked on by Katie
    # Gets the average value of given measurement (loincID) for twelve months before patient's diagnosis
    # May be included in demographics/feature vectors
    def getYearPriorToDiagnosis(self, patientID, loincID):
        patientObservations = self.getDiagnosisHistory(patientID)