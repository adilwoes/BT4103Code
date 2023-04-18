'''
Instruction:
There are three things needed to be in place for this file to work:
    1. Training data for anomaly
    2. Raw data in csv, with each slide(since it was extracted from slides) separated by columns
    3. Download all the libraries required
    
There are four files needed to standardize the job report data:
    1. "job_report_standardization.py", this python file
    2. "constant.py" which stores all the constant variables such as product name, technique names
    3. "mapping.py" which stores all the mapping logics for columns such as product name, technique names
    4. "helpfunction.py" which contains the help function
    
There are three assumptions for the automation report
    1. There are 3 slides, each slide is split to a column in raw excel.
        1st column: powerpoint name(Strictly only file name, other contents such as filepath could lead to wrong extraction)
        2nd column: Brief summary(analysts, requested by, etc)
        3nd column: Everything from problem description onwards
'''

# Import required libraries
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from dateparser.search import search_dates
from fuzzywuzzy import fuzz, process
from textblob import TextBlob
import pandas as pd
import numpy as np
import jellyfish
import datetime
import pytz
import json
import math
import re
from nltk.corpus import stopwords ##You might need to download stopwords if you haven't
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

# Import custom modules
from helpfunction import HelpFunctions
from constant import TEST_PHASES, TEST_PHASE_VARIANT, FAILURE, FAILURE_VARIANT, FAILURE_SUBVARIANT, FAILING_CONDITIONS, ANALYSTS, PFA_ANALYSTS, PRODUCT_NAMES, PRODUCT_EXCLUSION, TECHNIQUES, DAMAGE_INCLUSIONS
from mapping import ANALYSTS_MAPPING, PRODUCT_MAPPING, TECHNIQUES_MAP, TECHNIQUES_MAP_LONG, TECHNIQUES_MAP_SHORT, PROD_BU_TN_MAPPING

# Required data files
non_std_df_file_name = "Non-standardizedData.xlsx"
anomaly_detector_training_data = "training_data_anomaly.csv"

# Read in unstructured raw data
non_std_df = pd.read_excel(non_std_df_file_name) ##Change file name accordingly
rows = non_std_df.astype(str).apply(" slide_sep ".join, axis = 1).to_list() ##Store all raw data in "rows"


# 1. Train Anomaly Detection Model
# 1.1 Set up objects
sm = SMOTE()
cv = CountVectorizer()
tfidftransformer = TfidfTransformer()

# 1.2 Reading test data (More data fed into train data, more robust the anomaly detector)
anomaly_detection_df = pd.read_csv(anomaly_detector_training_data)
##Pre-processing
anomaly_detection_df = anomaly_detection_df[~anomaly_detection_df.anomaly_detected.isna()]
anomaly_detection_df = anomaly_detection_df[anomaly_detection_df.anomaly_detected != 0]
anomaly_detection_df = anomaly_detection_df.drop("Unnamed: 0", axis = 1)
anomaly_detection_df.anomaly_detected = anomaly_detection_df.anomaly_detected.apply(lambda x: int(x))
X = anomaly_detection_df.tech_used
y = anomaly_detection_df.anomaly_detected

# 1.3 Transforming, over-sampling and training
X = cv.fit_transform(X).toarray()
X = tfidftransformer.fit_transform(X).toarray()
X, y = sm.fit_resample(X, y) ##Over sampling using smote
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1.3.1 Training
accuracy = 0
while accuracy < 0.50: ##Avoid random sampling error
    clf = MLPClassifier(solver='lbfgs', activation = "relu", alpha=0.1, hidden_layer_sizes=(2, 5))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for selected model:\n{round(accuracy, 2)}")

# 1.4 Wrap model into object for automation's usage
class AnomalyPred:
    def __init__(self, model, tfidftransformer, cv):
        self.model = model
        self.tfidftransformer = tfidftransformer
        self.cv = cv
        
    def predict(self, sentence):
        X = [sentence]
        X = self.cv.transform(X).toarray()
        X = self.tfidftransformer.transform(X)
        y_pred = self.model.predict(X)
        return y_pred[0]
    
anomaly_predictor = AnomalyPred(clf, tfidftransformer, cv)

## This is an example usage:
# anomaly_predictor.predict('Dynamic photoemission (PEM) analysis observed missing emission sites at lower half of Channel B (mct_t1.CH0)')

# 2. Automation to standardize raw data
# 2.1 Create help function class
hf = HelpFunctions()

# 2.2 Create automation object
class Standardize:
    
    # Columns to standardize
    COLS =  ['job id',
             'quarter',
             'analyst(s)',
             'product name',
             'test phase',
             'test phase variant',
             'time',
             'failure',
             'failure variant',
             'failure sub variant',
             'failing conditions',
             'failure remarks',
             'fi success',
             'pfa success',
             'technique',
             'fi cost',
             'pfa cost',
             'remarks',
             'date finished',
             'bu',
             'technology node',
             'root cause']
        
    def __init__(self, row : str, prod_name_score = 80, analyst_name_score = 65, techniques_score = 59.5, date_score = 40):
        self.prod_name_score = prod_name_score
        self.analyst_name_score = analyst_name_score
        self.techniques_score = techniques_score
        self.date_score = date_score
        self.output = {}
        self.original_text_chunk = row
        self.text_chunk = self.pre_process(row)
        self.slide0 = self.text_chunk.split("slide_sep")[0]
        self.slide1 = self.text_chunk.split("slide_sep")[1]
        self.slide2 = self.text_chunk.split("slide_sep")[2] 
        self.slide0 = self.slide0.lower().replace(":", " ").replace("'", " ").replace("-", " ").replace("_", " ")
        self.slide1 = self.slide1.lower().replace(":", " ").replace("'", " ").replace("-", " ").replace("_", " ")
        self.slide2 = self.slide2.lower().replace(":", " ").replace("'", " ").replace("-", " ")
        ##set up constant variables
        self.TEST_PHASES = TEST_PHASES #Variation of test phases
        self.TEST_PHASE_VARIANT = TEST_PHASE_VARIANT #Variations of test phase variant
        self.FAILURE =FAILURE #Variations of failures
        self.FAILURE_VARIANT = FAILURE_VARIANT #Variations of failure variant
        self.FAILURE_SUBVARIANT = FAILURE_SUBVARIANT #Variations of failure sub-variant
        self.FAILING_CONDITIONS = FAILING_CONDITIONS #Variations of failing condition
        self.ANALYSTS = ANALYSTS #List of FI analyst
        self.PFA_ANALYSTS = PFA_ANALYSTS #List of PFA analyst
        self.PRODUCT_NAMES = PRODUCT_NAMES #List of product names
        self.PRODUCT_EXCLUSION = PRODUCT_EXCLUSION #List of product exclusions
        self.TECHNIQUES = TECHNIQUES #List of techniques
        self.ANALYSTS_MAPPING = ANALYSTS_MAPPING #Hard mapping for analysts
        self.PRODUCT_MAPPING = PRODUCT_MAPPING #Hard mapping for product names
        self.TECHNIQUES_MAP = TECHNIQUES_MAP #Hard mapping for techniques
        self.TECHNIQUES_MAP_LONG = TECHNIQUES_MAP_LONG #Mapping techniques to standardised output(Long technique names)
        self.TECHNIQUES_MAP_SHORT = TECHNIQUES_MAP_SHORT #Mapping techniques to standardised output(Short technique names)
        self.PROD_BU_TN_MAPPING = PROD_BU_TN_MAPPING #Product BU and Technology Node mapping using product name
        self.DAMAGE_INCLUSIONS = DAMAGE_INCLUSIONS #Words that signify FI damaged the unit during process
        
    
    def test(self):
        return self.output
        
    def fill(self):
        self.fill_job_id()
        self.fill_date_finished()
        self.fill_analysts()
        self.fill_product_name()
        self.fill_test_phase()
        self.fill_time()
        self.fill_failure() ##consist of failure, failure variant, failure sub variant
        self.fill_failure_condition()
        self.fill_failure_remark()
        self.fill_techniques()
        self.fill_root_cause()
        self.fill_fi_success()
        self.fill_fi_cost()
        self.fill_remark()
        self.fill_pfa_cost()
        self.fill_pfa_success()
        self.output = dict([(k, self.output[k]) for k in self.COLS]) ##Align the columns with self.COLS
        self.output["original_text"] = self.original_text_chunk
    
    def pre_process(self, row):
        """
        1. Change all content to lower case
        2. Add space to all commas to improve tokenization
        3. Remove '\n' and other symbols using ReGEx
        4. Tokenise words using Textblob
        5. Join the tokenised words into a string and remove all quotation marks
        
        Parameters
        ----------
        row: str
            The row of data from Excel that is used for data extraction. It is assumed that each row
            is extracted from power point slides and each slide is separated using 'slide_sep' separator.
            
        Returns
        ----------
        row:
            Pre-processed row data.
        """
        row = row.lower()
        row = row.replace(",", ", ")
        row = re.sub(r'[^\w\s]',' ',row)
        row = TextBlob(row)
        row = row.words
        row = " ".join(row).replace("'", "").replace('"', "")
        return row
    
    def fill_job_id(self):
        """
        Using simple regex pattern to extract job id
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["job id"] dictionary.
            
        """
        job_id = re.search("[a-z]{2}\d{7}[a-z]{2}", self.text_chunk)[0]
        self.output["job id"] = job_id
        ##self.text_chunk = self.text_chunk[self.text_chunk.find(job_id):]
        pass
    
    def fill_date_finished(self):
        """
        Extract date from the text chunk. 
        Using fuzzy ratio to further enhance the accuracy and validity of date extracted. 
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["date finished"] and self.output["quarter"].
        """
        text = self.text_chunk
        min_date = datetime.datetime(2010, 1, 1)
        original_text = self.original_text_chunk.replace("\n", "")
        # Standardizing timezone
        k = search_dates(f"{text} {original_text}", languages = ["en"], settings={'TIMEZONE': 'UTC'})
        k = list(map(lambda x: (x[0], x[1].replace(tzinfo=pytz.utc)), k)) 
        k = list(filter(lambda x: x[1] > pytz.timezone("UTC").localize(min_date) and x[1] < datetime.datetime.now(datetime.timezone.utc), k))
        # Using fuzzy ratio to enhance date search
        k = list(map(lambda x: (x[0], x[1], fuzz.ratio(x[1].strftime('%m%d%Y'), x[0])), k))
        k = list(filter(lambda x: x[2] > self.date_score, k))
        k = list(filter(lambda x: len(x[0]) >= 6, k))
        if k:
            ##Restrict date to 2010 till present
            k.sort(key = lambda x: x[2], reverse = True)
            date = k[0][1]
            quarter = math.ceil(date.month/3.)
            self.output["date finished"] = date.strftime("%m/%d/%Y")
            self.output["quarter"] = f"Q{quarter}"
        else:
            self.output["date finished"] = "00/00/00"
            self.output["quarter"] = "0"
        pass

    def fill_analysts(self):
        """
        Obtain analyst names that appear in this job using following steps:
        1. Find range. Range is defined content between the word 'analyst' and 'review'.
        2. Removed all contents that we know do not belong to FI analyst names. 
            eg. PFA names
        3. Apply hard mapping. 
            eg. gopi -> gopinath
        4. Search logic
            a. For each analyst name, we split search content into list, and concatenate element i and i + 1 to increase
            search variability. (Since sometimes analyst names might split)
            b. Define analyst name as arrays of their first and last names. Eg. David Jinjie -> [0, 0]
            c. For each component of analyst name, find the best match. Eg. Find best match of 'david'. If it is 
            present, update the respective analyst name array with 1.
            d. After iterating through David and Jinjie, if analyst array has at least one '1', consider it as analyst
            name is present in the content.
            
        *Note, we do not consider analyst name to be a match if the component of the name has only two letters. 
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["analyst(s)"]
        """
        # Define range
        texts = self.text_chunk[abs(self.text_chunk.find("analyst") - 1):self.text_chunk.find("review")]
        for pfa in self.PFA_ANALYSTS:# Remove PFA analyst, aka removing names that we know should not belong to FI 
            texts = texts.replace(pfa, "")
        for pattern, mapping in self.ANALYSTS_MAPPING.items(): # Apply mapping
            texts = texts.replace(pattern, mapping)
        result = []
        for analyst in self.ANALYSTS:
            name = analyst.split(" ")
            name_arr = [0] * len(name) # Create list for name, EG "David Jinjie" -> [0, 0]
            i = 0
            temp_texts = texts.split(" ")
            temp_texts.extend([f"{temp_texts[i]}{temp_texts[i + 1]}" for i in range(len(temp_texts) - 1)]) # concat with next string
            while len(temp_texts) > 1 and i < len(name_arr): 
                n = name[i]
                best_match = hf.find_similar_substring(temp_texts, n)
                if hf.is_abbreviation(best_match, n) and fuzz.ratio(n, best_match) > self.analyst_name_score and len(best_match) > 2:
                    best_idx = temp_texts.index(best_match) 
                    name_arr[i] = best_idx
                    if len(n) < len(best_match): # Allows for partial name match
                        temp_texts[best_idx] = best_match[:len(n)] # If it is partial match, do not look ahead.
                    else:
                        temp_texts = temp_texts[best_idx + 1:] # If it is full match, look ahead.
                i += 1
            if name_arr.count(0) <= 1: # At least one word from the name must be present.
                result.append(analyst)
        self.output["analyst(s)"] = ", ".join(result)

    def fill_product_name(self):
        """
        Using similarity score to pin-point product name.
        1. Split search range into slides. Since product names are usually in the first slide, we relax the 
        condition of matching for first slide.
        2. Remove all words that we know do not belong to product name
            eg. Failures such as 'bist'
        3. Apply hard mapping. 
        4. Search Logic:
            a. For each slide, split into list of texts. Concatenate element i with i + 1 to increase search variability.
            b. For each product names in product list, look for best maching product name. 
            c. If the best match is an abbreviation and the match score is higher than user defined score, add it to result.
        5. If there is result in the search, take the best result and update dictionary.
        6. Update BU and technology node accordingly.
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["product name"]

        """
        result = []
        slides = self.text_chunk.replace(self.output["job id"], "") # remove job id
        for exclusion in self.FAILURE + self.PRODUCT_EXCLUSION:
            slides = slides.replace(exclusion, "") # Remove all the failure words
        for k, v in self.PRODUCT_MAPPING.items():
            slides = slides.replace(k, v)
        slides = slides.split("slide_sep")
        for j in range(len(slides[:2])):
            slide = slides[j]
            texts = re.split("_| ", slide)
            texts = [t for t in texts if len(t) > 0] # remove empty strings
            texts.extend([f"{texts[i]} {texts[i + 1]}" for i in range(len(texts) - 1)]) # concat with next string(two fold)
            for p in self.PRODUCT_NAMES:
                best_matches = process.extract(p, texts, scorer = fuzz.ratio)
                if j == 0:
                    best_matches = best_matches[:1] # Take the first product if product is in first slide.
                for best_match in best_matches:
                    score = best_match[1]
                    best_match = best_match[0]
                    if hf.is_abbreviation(best_match, p) and score > self.prod_name_score * (j if j == 0 else 1): # Gives penalty to slide 2
                        result.append((p, score, best_match))
            if result:
                result.sort(key = lambda x: x[1] + len(x[0]), reverse = True) # return longer match
                self.output["product name"] = result[0][0]
                prod_name = self.output["product name"]
                if prod_name in self.PROD_BU_TN_MAPPING.keys():
                    bu = self.PROD_BU_TN_MAPPING[prod_name][0]
                    tn = self.PROD_BU_TN_MAPPING[prod_name][1]
                    self.output['bu'] = bu
                    self.output['technology node'] = tn
                else:
                    self.output['bu'] = "no bu found"
                    self.output['technology node'] = "no tn found"
                return # If result is found at earlier slide terminate the search.
        self.output["product name"] = "no product found"
        self.output['bu'] = "no product found"
        self.output['technology node'] = "no product found"
        return
        ##Assume only one product per row


    def fill_test_phase(self):
        """
        Extract test phase from the text chunk. 
        Search for test phase first, then search for test phase variant through exact matching
        since each category have limited characters.
        As most of the time, only the test phase variant is mentioned for 'REL' test phase,
        if test phase isn't found but test phase variant is found, re-fill test phase accordingly.
        
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["test phase"] and self.output["test phase variant"]
        """
        test_phase = ""
        test_phase_variant = ""
        for test in self.TEST_PHASES:
            if (test in self.slide0) or (test in self.slide1):
                test_phase = test
                break

        if test_phase == "":
            for test in self.TEST_PHASES:
                if test in self.slide2:
                    test_phase = test
                    break
                    
        if test_phase == "" or test_phase=='rel':
            if "tigershark" in self.slide1.split():
                test_phase_variant = "era"
                test_phase = "rel"
            elif any(word.startswith("era") for word in self.slide1.split()):
                test_phase_variant = "era"
                test_phase = "rel"
            elif any(word.startswith("qual") for word in self.slide1.split()):
                test_phase_variant = "qual"
                test_phase = "rel"
            else:
                for variant in self.TEST_PHASE_VARIANT:
                    if variant in self.slide1.split(): 
                        test_phase_variant = variant
                        test_phase = "rel" 
                        break

        if test_phase_variant =="" and (test_phase == "" or test_phase=='rel'):
            for variant in self.TEST_PHASE_VARIANT:
                if variant in self.slide2.split(): 
                    test_phase_variant = variant
                    test_phase = "rel" 
                    break
        self.output["test phase"] = test_phase.upper() if test_phase else ""
        self.output["test phase variant"] = test_phase_variant.upper() if test_phase_variant else ""


    
    def fill_time(self):
        """
        Use regex pattern to extract time.
        Note: Only has value when test_phase == 'rel'
        Hence, we search for time when test phase is rel or has not been found.
        If test phase is nout found previously but there is time,
        then we indicate test phase as 'rel'.
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["time"] and self.output["test_phase"]
            
        """
        time = ""
        if self.output["test phase"] == 'rel' or self.output["test phase"] == '':
            result = re.findall(r'\bt\d*\.?\d+h?\b', self.slide1)
            result += re.findall(r'\bt\d*\.?\d+h?\b', self.slide2)
            if result:
                time = result[0].upper()
                self.output["test phase"] = 'rel'
                if time[-1]!='H':
                    time += 'H'
        self.output["time"] = time if time else ""

    
    def fill_failure(self):
        """
        Fill failure, failure variant and failure sub variant accordingly 
        by searching for exact match (since each category have limited characters)
        with the help of mapping list which specifies the relationship between them.
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["failure"], self.output["failure variant"] and self.output["failure sub variant"] 
        """
        failure = ''
        failure_variant = ''
        failure_sub_variant = ''

        for f in self.FAILURE:
            if f in self.slide0.split():
                failure = hf.helper(f, failure)
                break

        # to fill failure 
        if failure == "":
            for f in self.FAILURE:
                if f in self.slide1.split():
                    failure = hf.helper(f, failure)
                    break

        if failure == "":
            for f in self.FAILURE:
                if f in self.slide2.split():
                    failure = hf.helper(f, failure)
                    break

        # to fill failure variant
        failure_lst = failure.split(", ")
        for f in failure_lst:
            if f in list(self.FAILURE_VARIANT.keys()): 
                for v in self.FAILURE_VARIANT[failure]:
                    if v in self.slide2.split():
                        if failure_sub_variant == "":
                            failure_sub_variant = v
                        else:
                            failure_sub_variant += ", " + v

        # to fill failure sub variant 
        for f in failure_lst:
            if failure in list(self.FAILURE_SUBVARIANT.keys()):
                for v in self.FAILURE_SUBVARIANT[failure]:
                    if v in self.slide2.split():
                        if failure_sub_variant == "":
                            failure_sub_variant = v
                        else:
                            failure_sub_variant += ", " + v


        self.output["failure"] = failure if failure else ""
        self.output["failure variant"] = failure_variant if failure_variant else ""
        self.output["failure sub variant"] = failure_sub_variant if failure_sub_variant else ""
        
        
    
    def fill_failure_condition(self):
        """
        Fill failing condition by searching for exact match since each category have limited characters.
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["failure_condition"] 
        """
        failing_conditions = ""
        for fc in self.FAILING_CONDITIONS:
            if fc in self.slide2.split():
                if failing_conditions == "":
                    failing_conditions = fc
                else:
                    failing_conditions += ", " + fc
        self.output["failing conditions"] = failing_conditions if failing_conditions else ""


    def fill_failure_remark(self):
        self.output["failure remarks"] = ""
        pass
    
    def fill_fi_success(self):
        """
        Define FI success as if both conditions are met:
        1. Product is not damaged
        2. Anomaly is detected by FI techniques
        
        Note*: We only consider product to be damaged during investigation if words in DAMAGE_INCLUSION appears
        in the content.
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["FI success"]
        """
        if "yes" not in str(self.output["technique"]).lower() and "damage" in self.output["root cause"]:
            for inclusion in self.DAMAGE_INCLUSIONS: ## Since there is word damage, we want to see if damage is from FI or customer
                if inclusion in str(self.output["root cause"]).lower().replace(" ", ""):
                    self.output["fi success"] = 0
                    return 
        if "yes" not in str(self.output["technique"]).lower():
            self.output["fi success"] = 0
        else:
            self.output["fi success"] = 1
            
    def fill_fi_cost(self):
        self.output["fi cost"] = ""
        pass
        
    def fill_pfa_cost(self):
        self.output["pfa cost"] = ""
        pass
    
    def fill_pfa_success(self):
        self.output["pfa success"] = ""
        pass
    
    def fill_remark(self):
        self.output["remarks"] = ""
        pass
    
    def fill_techniques(self):
        """
        This function defined a customized scorer for most similar words extraction. The scorer gives extract
        credit to words that seem to be exact abbreviation. (Eg. PEM = Photon Emission Microscopy, where first 3 letters
        of the full name is exactly equal to its abbreviation)
        
        1. Define the range for search. Try to minimise the search range as much as possible.
        2. Search range is the original text chunk instead. The processed text chunk do not contain line breaks, 
        however, we assume each line break contains exactly one technique. 
        3. Apply hard mapping.
        4. Split the search content into arrays, and pre-process each element in the array.
        5. Concatenate element i with i + 1 and i + 2 to increase variability in the search range.
        6. Search Logic
            a. For each row in the search content, find the most matching technique.
            b. Find the best matching technique after looking through all techniques
            c. Each technique is default dynamic, and default low resolution. 
            d. If high resolution detects anomaly, add low resolution technique and say that it detected anomaly too.
        
        Note*: Only consider the matched word if the it has at more than 2 letters.
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["techniques"]
        """
        def MyScorer(s1, s2): # Define scorer
            fratio = fuzz.ratio(s1, s2)
            if len(s1) > len(s2):
                longer = s1
                shorter = s2
            else:
                longer = s2
                shorter = s1
            if shorter == "".join(list(map(lambda x: x[0], longer.split(" ")))):
                fratio += 100
                fratio = fratio/2
            return fratio
        result = []
        text_chunk = self.original_text_chunk.lower() # Bring to lower case
        if text_chunk.find("analysis summary") > 0:
            idx = text_chunk.find("analysis summary")
        elif text_chunk.find("analysis") > 0:
            idx = text_chunk.find("analysis")
        elif text_chunk.find("summary") > 0:
            idx = text_chunk.find("summary")
        else:
            idx = 0
        for technique, mapping in self.TECHNIQUES_MAP.items():
            text_chunk = text_chunk.replace(technique, mapping)
        text_chunk = re.split(r"\\.|\n", text_chunk[idx:])
        text_chunk = [self.pre_process(row) for row in text_chunk]    
        for j in range(len(text_chunk)):
            sentence = text_chunk[j]
            texts = sentence.split(" ")
            two_fold = [f"{texts[i]} {texts[i + 1]}" for i in range(len(texts) - 1)]
            three_fold = [f"{texts[i]} {texts[i + 1]} {texts[i + 2]}" for i in range(len(texts) - 2)]
            texts.extend(two_fold) # concat with next string
            texts.extend(three_fold) # two fold/three fold allows matching names that were separated
            temp_result = []
            for t in self.TECHNIQUES:
                best_matches = process.extract(t, texts, scorer = MyScorer)
                best_matches = list(filter(lambda x: hf.is_abbreviation(x[0], t), best_matches))
                for best_match in best_matches[:3]: ##Check top 3
                    temp_result.append((best_match[0], best_match[1], t))
            temp_result.sort(key = lambda x: x[1], reverse = True)
            if not temp_result:
                continue
            best_match, score, t = temp_result[0]
            if hf.is_abbreviation(best_match, t) and score > self.techniques_score and len(best_match) > 2: 
                is_high_res = "High Res" if "highres" in sentence.replace(" ", "") else "Low Res"
                is_static = "Static" if "static" in sentence else "Dynamic"
                detector = "Yes" if anomaly_predictor.predict(sentence) == 1 else "No"
                result.append((is_static, is_high_res, t, detector, score))
                if is_high_res == "High Res": # If high res is performed, means low res was performed without anomaly detection
                    result.append((is_static, "Low Res", t, "Yes", score))
        result.append(("", "", "layout tracing", "No", 0))
        if result: ##Fill up results
            result.sort(key = lambda x: x[-1], reverse = True)
            o = {}
            for r in result:
                if r[2] in self.TECHNIQUES_MAP_LONG.keys(): # Apply output mapping for long and short techniques(with and without dyanmic/static high res/low res)
                    name = f"{r[0]} {r[1]} {self.TECHNIQUES_MAP_LONG[r[2]]}"
                elif r[2] in self.TECHNIQUES_MAP_SHORT.keys():
                    name = f"{self.TECHNIQUES_MAP_SHORT[r[2]]}"
                if name not in o.keys(): 
                    o[name] = [r[3], ""]
                elif o[name] != ["No", ""]:
                    o[name] = [r[3], ""]
                else:
                    continue
            self.output["technique"] = str(o)
        else:
            self.output["technique"] = "no techniques found"
    
    def fill_root_cause(self):
        """
        Since machine is unable to identify the root cause in the content, we can only assume that all 
        the contents after the word 'conclusion' is the root cause.
        
        Parameters
        ----------
        Nil
            
        Returns
        ----------
        None:
            Update self.output["root cuase"]
        """
        s = self.original_text_chunk
        if "conclusion" in s.lower(): # if the word conclusion is in the content, return everything that is after it.
            c = s[s.lower().find("conclusion") + len("conclusion"):]
            self.output["root cause"] = self.pre_process(c).replace("executive summary", "").strip()
            return
        self.output["root cause"] = self.original_text_chunk # Else, output the entire row. 

# 2.3 Execute automation
result = {}
for row in rows:
    try:
        s = Standardize(row)
        s.fill()
        d2 = s.test()
        if not result.keys():
            result = d2
            continue
        result = {key:np.hstack([result[key],d2[key]]) for key in d2.keys()}
    except Exception as e:
        print(str(e))
        print(f"Unable to extract:\n{row}")

output = pd.DataFrame(result)
output[output["analyst(s)"] != ""].to_csv("automation_output_v2_fi_only.csv")
output.to_csv("automation_output_v2.csv")