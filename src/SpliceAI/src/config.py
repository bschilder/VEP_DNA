import os

DATA_DIR = os.path.expanduser("~/projects/data/")

## PALETTES
PALETTES = {
    'WT': 'Blues',
    'Pathogenic': 'Oranges',
    'Benign': 'Greens'
}

def set_params_palettes(params):
    global PARAMS_PALETTES
    PARAMS_PALETTES.update(params)
    print("PARAMS_PALETTES has been updated.")

def get_params_palettes():
    return PARAMS_PALETTES.copy() 

