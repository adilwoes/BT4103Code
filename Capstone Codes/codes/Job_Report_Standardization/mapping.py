ANALYSTS_MAPPING = {
                        "sh":"", ##SH can be recognized as Yixuan. 
                        "gopi": "gopinath" ##Quite a common nick name
                    }
    
    
    
PRODUCT_MAPPING = {
                    "gilbratar": "starship",
                    "rome":"starship",
                    "milan":"genesis",
                    "stones durango":"stones",
                    "romex":"starship-x",
                    "milanx": "genesis-x",
                    "gn":"genesis", ##Hardcoded logic, code intepret it as genoa
                    "aecg": "no product found", 
                    "xilinx": "no product found",
                    "br": "bristol" # Hard coded logic, code intepret it as bristol
                    }
    
TECHNIQUES_MAP = {
                "thermal imaging": "thermal camera",
                "thermal emssion": "thermal camera", 
                "thermal mapping": "thermal camera", 
                #"c-mode scanning acoustic microscope": "confocal scanning acoustic microscopy",
              "bit ": "", ##remove bit with space to avoid matching with bitkill, also demand full match for bitkill
              "bit-":"",
              "VSS": "",  #remove VSS and demand full match for volume scan
              "VS": "" #demand full match for volume scan
    }
    
TECHNIQUES_MAP_LONG = {
                 'critical timing path' : "Critical Timing Path",
                 'thermally induced voltage alteration': "TIVA",
                 'thermal camera': "Thermal Camera",
                 'photon emission microscopy': "PEM",
                 'laser scanning microscopy': "LSM",
                 'soft defect localization': "SDL",
                 'laser voltage probing': "LVP",
                 'laser probing': "LP",
                 'laser voltage imaging': "LVI",
                 'short wavelength probing': "SWP"}
    
TECHNIQUES_MAP_SHORT = {    ##Techniques that do not have static/dynamic variation
                'layout tracing':"Layout Tracing",
                 'sample polishing': "Sample Polishing(Austin only)",
                 'curve trace': "Curve Trace",              
                 'volume scan': "Volume Scan",
                 'bitkill': "Bitkill",
                 'confocal scanning acoustic microscopy': "CSAM",
                 'die crack optical analysis': "Die Crack Optical Analysis",
                 'virage conversion': "Virage Conversion",
                 'reactive ion etching': "RIE",
                 'xray': "XRAY",
                 'laser induced voltage alteration':"LIVA"}

PROD_BU_TN_MAPPING = {'aerith': ('SCBU', 'tsmc7'),
                'anubis': ('SCBU', 'tsmc16'),
                'arden': ('SCBU', 'tsmc7'),
                'ariel': ('SCBU', 'tsmc7'),
                'arlene': ('SCBU', 'tsmc16'),
                'badami': ('Server', 'tsmc7'),
                'baffin': ('dGPU', 'gf14'),
                'baffin-l4': ('dGPU', 'gf14'),
                'baffin-s4a': ('dGPU', 'gf14'),
                'barcelo': ('Client', 'tsmc7'),
                'bergamo': ('Server', 'tsmc5'),
                'bristol': ('Client', 'gf28'),
                'cardinal': ('SCBU', 'tsmc7'),
                'castlepeak': ('Client', 'tsmc7'),
                'cezanne': ('Client', 'tsmc7'),
                'clayton': ('SCBU', 'tsmc16'),
                'clayton12': ('SCBU', 'tsmc12'),
                'colossal': ('Test Chip', 'tsmc3'),
                'dragon': ('Test Chip', 'sec4'),
                'ellesmere': ('dGPU', 'gf14'),
                'fireflight': ('SCBU', 'gf14'),
                'fremont': ('Client', 'gf14'),
                'genesis': ('Server', 'tsmc7'),
                'genesis-x': ('Server', 'tsmc7'),
                'gladius': ('SCBU', 'tsmc16'),
                'hammerhead': ('Test Chip', 'tsmc5'),
                'jupiter': ('SCBU', 'gf14'),
                'kingston': ('SCBU', 'tsmc16'),
                'lexa': ('dGPU', 'gf14'),
                'manta': ('Test Chip', 'tsmc6'),
                'matisse': ('Client', 'tsmc7'),
                'matisse2': ('Client', 'tsmc7'),
                'mendocino': ('Client', 'tsmc6'),
                'mero': ('SCBU', 'tsmc7'),
                'mi100': ('dGPU', 'tsmc7'),
                'mi200': ('dGPU', 'tsmc7'),
                'mi300': ('dGPU', 'tsmc7'),
                'montego': ('SCBU', 'tsmc16'),
                'navi10': ('dGPU', 'tsmc7'),
                'navi12': ('dGPU', 'tsmc7'),
                'navi14': ('dGPU', 'tsmc7'),
                'navi21': ('dGPU', 'tsmc7'),
                'navi22': ('dGPU', 'tsmc7'),
                'navi23': ('dGPU', 'tsmc7'),
                'navi24': ('dGPU', 'tsmc6'),
                'navi31': ('dGPU', 'tsmc6'),
                'navi33': ('dGPU', 'tsmc6'),
                'oberon': ('SCBU', 'tsmc7'),
                'oberon plus': ('SCBU', 'tsmc6'),
                'octo': ('Test Chip', 'tsmc3'),
                'odie': ('SCBU', 'gf14'),
                'picasso': ('Client', 'gf14'),
                'polaris20': ('dGPU', 'gf14'),
                'polaris22': ('dGPU', 'gf14'),
                'polaris30': ('dGPU', 'gf14'),
                'pooky': ('SCBU', 'gf14'),
                'raphael': ('Client', 'tsmc5'),
                'raphael-x': ('Client', 'tsmc5'),
                'raven': ('Client', 'gf14'),
                'raven2': ('Client', 'gf14'),
                'rembrandt': ('Client', 'tsmc6'),
                'renoir': ('Client', 'tsmc7'),
                'sailfish': ('Test Chip', 'tsmc7'),
                'sailfish2': ('Test Chip', 'tsmc7'),
                'snowmass': ('SCBU', 'gf14'),
                'sparkman': ('SCBU', 'tsmc7'),
                'starship': ('Server', 'tsmc7'),
                'stones': ('Server', 'tsmc5'),
                'stoney ridge': ('Client', 'gf28'),
                'thresher': ('Test Chip', 'tsmc5'),
                'tigershark': ('Test Chip', 'tsmc5'),
                'vega10': ('dGPU', 'gf14'),
                'vega12': ('dGPU', 'gf14'),
                'vega20': ('dGPU', 'tsmc7'),
                'vermeer': ('Client', 'tsmc7'),
                'vermeer-x': ('Client', 'tsmc7'),
                'zeppelin': ('Server', 'gf14')}

   
