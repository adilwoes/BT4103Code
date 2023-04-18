import jellyfish
from fuzzywuzzy import fuzz

class HelpFunctions:
    
    @staticmethod
    def find_similar_substring(large_string: list, target_string: str) -> str:
        best_match = ''
        best_distance = float('inf')
        target_length = len(target_string)
        target_sound = jellyfish.soundex(target_string)
        for substring in large_string:
            substring_sound = jellyfish.soundex(substring)
            substring_distance = 1/(1+fuzz.ratio(substring_sound, target_sound)) + 1/(1+fuzz.ratio(substring, target_string)) 
            if substring_distance < best_distance:
                best_match = substring
                best_distance = substring_distance
        return best_match
    
    @staticmethod
    def is_abbreviation(abbr, word): 
        """
        Function for testing if abbr is an abbreviation of word
        To be placed inside list of help functions
        """
        if not (abbr and word):
            return False
        i, j = 0, 0
        while i < len(abbr) and j < len(word):
            if abbr[i].upper() == word[j].upper():
                i += 1
            j += 1
        if i != len(abbr) and i < len(word):
            return False
        start_cond = abbr[0] == word[0] and i >= len(abbr) and j <= len(word) 
        ##Make sure the abbr index is not finished yet and word is unfinished 
        return start_cond 
    
    @staticmethod
    def find_failure(self, f, failure):
        if f == 'parameter' or f=='psshort' or f=='short' or f=='leak':
            if failure == '':
                failure = 'parametric'
            else:
                failure += ", " + 'parametric'
        elif f == 'spec':
            if failure == '':
                failure = 'iospec'
            else:
                failure += ", " + 'iospec'
        elif f == 'scandelay':
            if failure == '':
                failure = 'scan'
            else:
                failure += ", " + 'scan'
        else:
            if failure == '':
                failure = f
            else:
                failure += ", " + f
        return failure
    
hf = HelpFunctions()
