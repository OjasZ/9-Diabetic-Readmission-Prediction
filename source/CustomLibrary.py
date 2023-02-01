import pandas as pd
pd.set_option('display.max_columns', 200) #set to show all columns
pd.set_option('display.max_rows', 200) 
import numpy as np

class clean():
    
    def __init__(self, dataframe):
        """

        Parameters
        ----------
        dataframe : TYPE pandas dataframe
            Initialize with dataframe to generate necessary column or row based data
            for data cleaning.

        Returns
        -------
        None.

        """
        # assign dataframe
        self.dataframe = dataframe
        
        # bad columns to drop
        self.bad_columns = ['encounter_id', 'patient_nbr', 'payer_code', 'weight', 
                   'medical_specialty', 'acetohexamide', 'tolbutamide',
                  'troglitazone', 'examide', 'citoglipton', 'glipizide-metformin',
                  'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                  'metformin-pioglitazone']
        
        # columns to make dummy columns
        self.dummy_columns = ["admission_type_id", "discharge_disposition_id", "admission_source_id", 
                     "age", "race"]
        
        # drug columns to make dummy columns regarding dosage
        self.dummy_drug_columns =  ["metformin", "repaglinide", "nateglinide", "chlorpropamide", 
                         "glimepiride", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", 
                         "acarbose", "miglitol", "tolazamide", "insulin", "glyburide-metformin"]
        
        self.blood_test_columns = ["max_glu_serum", "A1Cresult"]
        
        self.diagnosis_columns = ["diag_1", "diag_2", "diag_3"]
        
        diab_diag_1 = list(dataframe.diag_1.apply(lambda x: x if x[:3]=="250" else np.nan).dropna().unique())
        diab_diag_2 = list(dataframe.diag_2.apply(lambda x: x if x[:3]=="250" else np.nan).dropna().unique())
        diab_diag_3 = list(dataframe.diag_3.apply(lambda x: x if x[:3]=="250" else np.nan).dropna().unique())
        self.diab_code = set(diab_diag_1 + diab_diag_2 + diab_diag_3)
    
    def clean_data(self):
        df = self.dataframe
        # Drop columns with large missing values or had insignificant amount of data

        # Drop columns from the bad_columns list
        df.drop(columns=self.bad_columns, inplace=True)
        
        # Drop index with missing gender information
        df.drop(index = df[df.gender == "Unknown/Invalid"].index, inplace=True)
        
        # update to numeric, male = 0 female = 1
        df.gender = df.gender.apply(lambda x: 0 if x == "Male" else 1)
        
        # create dummy columns from column list
        df = self.create_dummy_columns(df)
    
        # list of drug columns

        # create dummy columns for drugs
        df = self.create_dummy_drugs(df)
    
        # list of blood tests
        # create dummy columns for blood tests
        df = self.create_dummy_bloodtests(df)
        
        # bin diagnosis codes
        df = self.diagnosis_binning(df)
              
        # create dummy columns for diagnosis bins
        df = self.create_dummy_diagnosis(df)
        
        # update predictant to numeric
        df.readmitted = df.readmitted.apply(lambda x: 0 if x == "NO" else 1 if x == "<30" else 2)
        
        # update change color to numeric
        df.change = df.change.apply(lambda x: 0 if x == "No" else 1)
        
        # update diabetesMed to numeric
        df.diabetesMed = df.diabetesMed.apply(lambda x: 0 if x == "No" else 1)
        
        return df
        
    # Input: Dataframe
    # Output: Dataframe with dummied columns from self.dummy_columns
    def create_dummy_columns(self, dataframe):
        for column in self.dummy_columns:
            dummies = pd.get_dummies(dataframe[column], drop_first=True, prefix=column)
            dataframe = pd.concat([dataframe, dummies], axis=1)
            dataframe.drop(columns=column, inplace = True)
        return dataframe
    
    # Input: Dataframe
    # Output: Dataframe with dummied columns from self.dummy_drug_columns
    def create_dummy_drugs(self, dataframe):
        for drug_column in self.dummy_drug_columns:
            if "No" not in dataframe[drug_column].unique(): 
                dummies = pd.get_dummies(dataframe[drug_column], prefix=drug_column)
            else:
                dummies = pd.get_dummies(dataframe[drug_column], prefix=drug_column)
                column_name = "_".join([drug_column, "No"])
                dummies.drop(columns=column_name, inplace=True)
            dataframe = pd.concat([dataframe, dummies], axis=1)
            dataframe.drop(columns=drug_column, inplace = True)
        return dataframe
    
    # Input: Dataframe
    # Output: Dataframe with dummied columns from self.blood_test_columns
    def create_dummy_bloodtests(self, dataframe):
        for blood_test in self.blood_test_columns:
            dummies = pd.get_dummies(dataframe[blood_test],  prefix=blood_test)
            column_name = "_".join([blood_test, "None"])
            dummies.drop(columns=column_name, inplace=True)
            dataframe = pd.concat([dataframe, dummies], axis=1)
            dataframe.drop(columns=blood_test, inplace = True)
        return dataframe
    
    def diagnosis_clean(self, value):
        # get all diabetest diagnosis
        
        # get string value of all diagnosis code
        circulatory_list = [str(code) for code in range(390, 460, 1)] + ["785"]
        respiratory_list = [str(code) for code in range(460, 520, 1)] + ["786"]
        digestive_list = [str(code) for code in range(520, 580, 1)] + ["787"]
        diabetes_list = self.diab_code
        injury_list = [str(code) for code in range(800, 1000, 1)]
        muscle_list = [str(code) for code in range(710, 740, 1)]
        genit_list = [str(code) for code in range(580, 629, 1)] + ["788"]
        neo_list = [str(code) for code in range(140, 240, 1)]
        
        # input: diagnosis code
        # output: diagnosis group based on code in the pdf
        if value in circulatory_list:
            return "circulatory"
        elif value in respiratory_list:
            return "respiratory"
        elif value in digestive_list:
            return "digestive"
        elif value in diabetes_list:
            return "diabetes"
        elif value in injury_list:
            return "injury"
        elif value in muscle_list:
            return "musculoskeletal"
        elif value in genit_list:
            return "genitourinary"
        elif value in neo_list:
            return "neoplasms"
        else:
            return "other"
    
    def diagnosis_binning(self, dataframe):    
        # replace diagnosis code with diagnosis group 
        dataframe.diag_1 = dataframe.diag_1.apply(self.diagnosis_clean)
        dataframe.diag_2 = dataframe.diag_2.apply(self.diagnosis_clean)
        dataframe.diag_3 = dataframe.diag_3.apply(self.diagnosis_clean)
        
        return dataframe
    
    def create_dummy_diagnosis(self, dataframe):
        for diag_col in self.diagnosis_columns:
            dummies = pd.get_dummies(dataframe[diag_col], prefix=diag_col)
            drop_col = "_".join([diag_col, "other"]) #dropped "other" diagnosis as reference column
            dummies.drop(columns=drop_col, inplace=True)
            dataframe = pd.concat([dataframe, dummies], axis=1)
            dataframe.drop(columns=diag_col, inplace=True)
        return dataframe


    
    
    
    
    
    
    



