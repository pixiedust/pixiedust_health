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
from __future__ import print_function
from six import iteritems
import re
import functools
from sklearn import metrics, model_selection, ensemble, svm, linear_model
import pandas as pd
import numpy as np


# Hopefully find a better way to do this if we expand to other diseases
diseaseMap = {1: {'DisplayName': 'Diabetes',
                  'SnomedIDs': ['44054006', '359642000', '81531005','237599002',
                                '199230006','609567009', '237627000', '9859006',
                                '190331003','703138006', '314903002','314904008',
                                '190390000','314772004', '314902007','190389009',
                                '313436004','1481000119100'],
                  'LoincIDs': ['4548-4', '29463-7','8302-2'],
                  'Features': ['HBA1C','WEIGHT','HEIGHT']}, 
              2: {'DisplayName': 'Hypertension',
                  'SnomedIDs': ['38341003'],
                  'LoincIDs': [],
                  'Features': ['BLOOD_PRESSURE']}
             }
loinc_dict = dict(zip(diseaseMap[1]['LoincIDs'], diseaseMap[1]['Features']))


# Utility Functions
def classifaction_report_df(report):
    report_data = []
    lines = report.split('\n')
    for idx in np.r_[2:4, 5]:
        line = lines[idx]
        row = {}
        row_data = re.split(r'\s\s+', line.strip())
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    df = df[['class', 'precision', 'recall', 'f1_score', 'support']].set_index('class')
    return df


# Class for storing data and generating DataFrames and matrices for UI and machine learning
# Requires 3 pandas data frames from create_v_demographic.csv, create_v_diagnosis.csv, and create_v_observation.csv
# Columns expected in demogaphics: 'EXPLORYS_PATIENT_ID', 'STD_GENDER', 'BIRTH_YEAR', 'STD_RACE'
# Columns expected in diagnosis: 'EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE', 'SNOMED_IDS'
# Columns expected in observations: 'EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID', 'STD_VALUE'
class Cohorts:
    
    #--------------------------------------- PRIVATE METHODS ---------------------------------------#
    
    ##-------------------------------------- Initialization --------------------------------------##
    
    def __init__(self, demographics, diagnosis, observations):
        print("diagnosis starting....", end="")
        self.__filterDiagnosis(diagnosis)
        print("diagnosis done")


        print("demographics starting....", end="")
        self.__filterDemographics(demographics)
        print("demographics done")
        
        print("observations starting....", end="")
        self.__getObservations(observations)
        print("observations done")

        self.__getDemographicFeatures(observations)

        # Memoization
        self._get_demog_call = None
        self._get_fvctor_call = None
    
    
    # Returns diagnosis history data frame
    # Requires original diagnosis histories chart
    # Used in initial processing of data
    def __filterDiagnosis(self, diagnosis):
        df_diag = diagnosis[['EXPLORYS_PATIENT_ID', 'DIAGNOSIS_DATE',
                             'SNOMED_IDS']]
        df_diag['DIAGNOSIS_DATE'] = pd.to_datetime(df_diag['DIAGNOSIS_DATE'])
        df_diag.SNOMED_IDS.fillna('', inplace=True)
        df_diag.SNOMED_IDS.astype('str', inplace=True)

        self.df_diag = df_diag
        return

        
    # Returns filtered demographics data frame
    # Requires original demographics chart
    # Used in initial processing of data
    def __filterDemographics(self, demographics):
        df_demog = demographics[['EXPLORYS_PATIENT_ID', 'STD_GENDER', 
                                 'BIRTH_YEAR', 'STD_ETHNICITY', 
                                 'STD_RACE', 'POSTAL_CODE_3']]
        df_demog['BIRTH_YEAR'] = 2017 - df_demog['BIRTH_YEAR']
        df_demog.rename(columns={'BIRTH_YEAR':'AGE'}, inplace=True)
                
        # ethnicity to indices
        dict_ethnicity = {'hispanic':1, 'non-hispanic':2, 'other':3, 'declined':4, 'unknown':4}
        df_demog["STD_ETHNICITY"].replace(dict_ethnicity, inplace=True)
        # df_demog['POSTAL_CODE_3'] = df_demog.POSTAL_CODE_3.to_string(na_rep='', float_format=':03.f'.format)
        # race to indices
        dict_race = {101:1, 615:2, 203:3, 699:4}
        df_demog["STD_RACE"].replace(dict_race, inplace=True)

        self.df_demog = df_demog
        return
    
    
    def __getObservations(self, observations):
        df_obsv = observations[['EXPLORYS_PATIENT_ID', 'LOINC_TEST_ID', 'STD_VALUE',
                                'OBSERVATION_DATE']]
        df_obsv['OBSERVATION_DATE'] = pd.to_datetime(df_obsv['OBSERVATION_DATE'])

        self.df_obsv = df_obsv
        return    
    
    # Returns filtered observation data frame
    # Requires original observations chart
    # Used in initial processing of data
    def __getDemographicFeatures(self, observations):
        df_demog = self.df_demog
        df_obsv = self.df_obsv

        df_comm_obsv = df_obsv[df_obsv.LOINC_TEST_ID.isin(list(loinc_dict.keys()))]
        df_comm_obsv.LOINC_TEST_ID.replace(loinc_dict, inplace=True)


        ### Analytics page
        # `getDemographic`: return both positive and negative demographics
        # ### demographic features
        tmp = df_comm_obsv.pivot_table(index='EXPLORYS_PATIENT_ID', values='STD_VALUE', 
                                                  columns='LOINC_TEST_ID')
        tmp.columns.name = None
        tmp.reset_index(inplace=True)

        demographicsCDF = pd.merge(tmp, df_demog, on='EXPLORYS_PATIENT_ID')
        demographicsCDF['BMI'] = (demographicsCDF.WEIGHT / ((demographicsCDF.HEIGHT) ** 2)) * 10000
        
        self.df_comm_obsv = df_comm_obsv
        self.demographicsFeatures =  demographicsCDF
        return
    
    
    #--------------------------------------- PUBLIC METHODS ---------------------------------------#
    
    # Page 0: Designed for drop-down menu in initial window of UI
    # Returns list of (DisplayName, ID) for all diseases we have data for
    # Example return: [('Diabetes', 1), ('Hypertension', 2)]
    def getDiseases(self):
        return [(value['DisplayName'], key) for key, value in iteritems(diseaseMap)]
    
    
    # Returns a dictionary of Data Frames, one for demographics and stats for patients with the given disease (key 'pos') 
    #   and one for patients who don't have that disease (key 'neg')
    # Columns in returned data frames: 'EXPLORYS_PATIENT_ID', 'STD_GENDER', 'STD_ETHNICITY', 'STD_RACE', 'AGE', 'HBA1C', 'WEIGHT', 'HEIGHT', 'BMI'
    # Had to calculate age  from birth year and BMI from weight and height
    # Designed for descriptive analytics window of UI and to use in getFeatureVectors()
    # This part is split into 2 separate data frames so the information shown in the pixieapp is just the people with the disease
    def getDemographics(self, diseaseID):
        df_diag = self.df_diag

        dis_snomed_ids = diseaseMap[diseaseID]['SnomedIDs']
        dis_snomed_regex = re.compile('|'.join(dis_snomed_ids))

        df_pos_diag = df_diag[df_diag.SNOMED_IDS.str.contains(dis_snomed_regex)]
        df_pos_diag = df_pos_diag.groupby('EXPLORYS_PATIENT_ID')['DIAGNOSIS_DATE'].min().reset_index()
        df_pos_diag.rename(columns={'DIAGNOSIS_DATE': 'FIRST_DIAGNOSIS_DATE'},
                           inplace=True)
        df_pos_diag['ONE_YEAR_PREV'] = df_pos_diag.FIRST_DIAGNOSIS_DATE.apply(lambda x: x - pd.DateOffset(years=1))
        
        all_patients = df_diag.EXPLORYS_PATIENT_ID.unique()
        pos_patients = df_pos_diag.EXPLORYS_PATIENT_ID.unique()
        neg_patients = set(all_patients) -  set(pos_patients)
       
        # updating disease state
        self.demographicsFeatures['HAS_DISEASE'] = self.demographicsFeatures.EXPLORYS_PATIENT_ID.isin(pos_patients)
        
        pos_demographics = self.demographicsFeatures[self.demographicsFeatures['EXPLORYS_PATIENT_ID'].isin(pos_patients)]
        neg_demographics = self.demographicsFeatures[self.demographicsFeatures['EXPLORYS_PATIENT_ID'].isin(neg_patients)]
        
        self.pos_patients = pos_patients
        self.neg_patients = neg_patients
        self.df_pos_diag = df_pos_diag
        return {'pos': pos_demographics, 'neg': neg_demographics}
    
    
    # Returns a data frame including both patients with and without given disease
    # HAS_DISEASE column indicates whether a patient has the disease (1) or not (0)
    # Returns only certain features if features are specified
    # Columns returned by default: 'STD_GENDER', 'AGE', 'STD_ETHNICITY', 'STD_RACE', 'HBA1C', 'WEIGHT', 'HEIGHT', 'BMI', 'HAS_DISEASE'
    # Out of the positive and negative patients, the bigger group is cut down to be the same size as the smaller group
    # Used in machine learning component
    def getFeatureVectors(self, diseaseID):
        # ### featurevectors
        df_comm_obsv = self.df_comm_obsv
        pos_patients = self.pos_patients
        neg_patients = self.neg_patients
        df_pos_diag = self.df_pos_diag
        df_demog = self.df_demog

        # **positive cases**
        df_pos_obsv = df_comm_obsv[df_comm_obsv.EXPLORYS_PATIENT_ID.isin(pos_patients)]

        tmp = pd.merge(df_pos_obsv, df_pos_diag, on='EXPLORYS_PATIENT_ID', how='inner')
        tmp = tmp.query('ONE_YEAR_PREV <= OBSERVATION_DATE <= FIRST_DIAGNOSIS_DATE').iloc[:, :3]
        posLoincFeatures = tmp.pivot_table(index='EXPLORYS_PATIENT_ID', values='STD_VALUE', 
                                      columns='LOINC_TEST_ID')
        posLoincFeatures.columns.name = None
        posLoincFeatures.reset_index(inplace=True)
        posFeatures = pd.merge(posLoincFeatures, df_demog, on='EXPLORYS_PATIENT_ID')
        posFeatures['HAS_DISEASE'] = True


        # **negative features**
        df_neg_obsv = df_comm_obsv[df_comm_obsv.EXPLORYS_PATIENT_ID.isin(neg_patients)]
        idx_date = (df_neg_obsv.groupby('EXPLORYS_PATIENT_ID')['OBSERVATION_DATE'].max()
                    - pd.DateOffset(years=1)).reset_index()
        idx_date.columns = ['EXPLORYS_PATIENT_ID', 'IDX_DATE']

        tmp = pd.merge(df_neg_obsv, idx_date, on='EXPLORYS_PATIENT_ID', how='inner')
        tmp = tmp.query('IDX_DATE <= OBSERVATION_DATE').iloc[:, :3]

        negLoincFeatures = tmp.pivot_table(index='EXPLORYS_PATIENT_ID', values='STD_VALUE', 
                                           columns='LOINC_TEST_ID')
        negLoincFeatures.columns.name = None
        negLoincFeatures.reset_index(inplace=True)
        negFeatures = pd.merge(negLoincFeatures, df_demog, on='EXPLORYS_PATIENT_ID')
        negFeatures['HAS_DISEASE'] = False


        allFeatures = pd.concat((negFeatures, posFeatures))
        allFeatures['BMI'] = (allFeatures.WEIGHT / ((allFeatures.HEIGHT) ** 2)) * 10000
        allFeatures.drop(['POSTAL_CODE_3'], axis=1, inplace=True)
        
        cols = allFeatures.columns.tolist()
        cols = cols[:-2] + [cols[-1], cols[-2]]
        allFeatures = allFeatures[cols]
        return allFeatures
    
    
    # Returns a data frame including both patients with and without given disease to be fed into the model 
    # Purpose of this function is to accomodate the ability to select features in the app without the need to recompute all columns
    # Feature Columns returned by default: 'STD_GENDER', 'AGE', 'STD_ETHNICITY', 'STD_RACE', 'HBA1C', 'WEIGHT', 'HEIGHT', 'BMI'
    # features argument must be some subset of the default columns in an array (that you wish to include)
    def getFeaturesForModel(self, data, features=None):
        if features != None:
            features.insert(0, 'EXPLORYS_PATIENT_ID')  # first column in patient id
            features.append('HAS_DISEASE')             # last column is disease target
            return data[features]
        return data
    
    
    #-------------------------------------- Building the Model -------------------------------------#
    
    # Returns array of the features to be used in the model from the data set
    # data is the table returned from getFeaturesForModel()
    def getFeatureNames(self, data):
        # Create a list of the feature column's names
        features = [str(item).encode('utf8') for item in data.columns]
        features = features[1:len(features)-1]
        return features

    def getClassification(self, data):
        # TODO:ensure correct data is passed
        allFeatures = data
        
        # feat_names = ['HBA1C', 'HEIGHT', 'WEIGHT', 'STD_GENDER', 'AGE', 'STD_ETHNICITY', 'STD_RACE', 'BMI']
        feat_names = allFeatures.columns[1:-1]
        # print(feat_names)

        # HACK to get a good result
        filt_idx = np.random.choice(allFeatures.index, 20000, replace=False)
        filtFeatures = allFeatures.loc[filt_idx, :]

        X, Y = filtFeatures[feat_names], filtFeatures['HAS_DISEASE'].astype('int')
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)


        # Train the classifier to take the training features and learn how they relate to the training y
        clf = ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(x_train, y_train) 

        probas_ = clf.predict_proba(x_test)
        y_pred = clf.predict(x_test)

        # Compute ROC curve and area the curve
        fpr, tpr, _ = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = metrics.auc(fpr, tpr)    

        clf_report = classifaction_report_df(metrics.classification_report(y_test, y_pred)).reset_index()
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'baseline': fpr})
        feat_df = pd.DataFrame({'feat_imp': clf.feature_importances_,
                                'feat_name':feat_names})
        return clf_report, roc_df, feat_df
  

