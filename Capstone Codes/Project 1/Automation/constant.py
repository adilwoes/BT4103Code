DAMAGE_INCLUSIONS = ["damageduring", "damagedduring"]
##if the word appear when there is damage mentioned in data, we do (do not put spaces)
## consider this job to be damaged by AMD

TEST_PHASES = ["rel", "ccar", "yield", "npi"]
    
TEST_PHASE_VARIANT = ["elfr", "htol", "hvs", "ort", "qual", "tsmc", "era"]

FAILURE = ['alarm', 'avfs', 'bist', 'crest', 'ddr', 'esd', 'fuse', 'hsio',
                'iospec', 'io spec', 'jtag', 'lpddr', 'parametric', 'parameter', 
                'pcle', 'pll', 'scan', 'sidd', 'slt', 'usb', 'volume scan', 'xgmi',
                # extra words - hardcode logic
                'leak', 'short', 'spec', 'psshort', 'scandelay'] 

FAILURE_VARIANT = {
                    'parametric':['short', 'leak', 'open'],
                    'scan':['chain', 'logic', 'stuck', 'delay'],
                    'bist':['row','column','bit'],
                    'pcle':['pll_core'],
                    'iospec':['vixvox'],
                    'usb':['inelb'],
                    'crest':['func_crest']
                    }

FAILURE_SUBVARIANT = {
                        'scan': ['core', 'gfx', 'soc', 'mic', 'tmon', 'gnb']
                        }

FAILING_CONDITIONS = ['vmax', 'vmin', 'vnom', 'fmax', 'fmin', 'fnom', 'any']
    
ANALYSTS = ['david jinjie', 'hu haoran', 'nathan linarto', 
                'winson lua', 'ng kaihui', 'angeline phoa', 
                'gopinath ranganathan', 'venkat krishnan ravikumar', 
                'seah yixuan', 'vasanth somasundaram', 'nicholas tee']

    
PFA_ANALYSTS =  ["david lum",
                    "hock bee",
                    "munyee",
                    "chea wei",
                    "soon huat",
                    "samuel",
                    "shang yi",
                    "xue hao",
                    "dion",
                    "we siang",
                    "yin zhe",
                    "jingwen",
                    "lei lei",
                    "qi soon",
                    "lito", 
                    "wei jie"]

PRODUCT_NAMES = ['anubis',
                     'arden',
                     'ariel',
                     'arlene',
                     'badami',
                     'baffin',
                     'baffinl4',
                     'baffin-s4a',
                     'barcelo',
                     'bergamo',
                     'bristol',
                     'cardinal',
                     'castlepeak',
                     'cezanne',
                     'clayton',
                     'clayton12',
                     'colossal',
                     'dragon',
                     'aerith',
                     'ellesmere',
                     'fireflight',
                     'fremont',
                     'genesis',
                     'genesisx',
                     'genoa',
                     'genoax',
                     'gladius',
                     'hammerhead',
                     'jupiter',
                     'kingston',
                     'lexa',
                     'manta',
                     'matisse',
                     'matisse2',
                     'mendocino',
                     'mero',
                     'mi100',
                     'mi200',
                     'mi300',
                     'milan',
                     'milanx',
                     'montego',
                     'navi10',
                     'navi12',
                     'navi14',
                     'navi21',
                     'navi22',
                     'navi23',
                     'navi24',
                     'navi31',
                     'navi32',
                     'navi33',
                     'oberon',
                     'oberon plus',
                     'odie',
                     'picasso',
                     'polaris20',
                     'polaris22',
                     'polaris30',
                     'pooky',
                     'raphael',
                     'raven',
                     'raven2',
                     'rembrandt',
                     'renoir',
                     'sailfish',
                     'sailfish2',
                     'snowmass',
                     'sparkman',
                     'starship',
                     'starshipx',
                     'stones',
                     'stonesx',
                     'stoney ridge',
                     'thresher',
                     'tigershark',
                     'vega10',
                     'vega12',
                     'vega20',
                     'vermeer',
                     'vermeerx',
                     'zeppelin']

PRODUCT_EXCLUSION = ["bit"]

TECHNIQUES = ['layout tracing',
                 'sample polishing',
                 'curve trace',
                 'critical timing path',
                 'thermally induced voltage alteration',
                 'thermal camera',
                 'photon emission microscopy',
                 'laser scanning microscopy',
                 'soft defect localization',
                 'laser voltage probing',
                 'laser probing',
                 'laser voltage imaging',
                 'short wavelength probing',
                 'volume scan',
                 'bitkill',
                 'confocal scanning acoustic microscopy',
                 'die crack optical analysis',
                 'virage conversion',
                 'reactive ion etching',
                 'xray',
                 'laser induced voltage alteration']