import os
import requests
import pooch

import src.utils as utils

# Get your token from: https://compbio.ccia.org.au/splicevardb/
token = os.getenv('SPLICEVARDB_TOKEN')
if token is None:
    raise ValueError("No token found. Get your token from: https://compbio.ccia.org.au/splicevardb/ and copy into your .env file as SPLICEVARDB_TOKEN={your_token}")

CACHE_DIR = pooch.os_cache('splicevardb')


def get_variants(base_url=" https://compbio.ccia.org.au/splicevardb-api",
                  endpoint="/variants/",
                  page_size=0,
                  offset=0,
                  cache=CACHE_DIR,
                  verbose=True):
    
    url = f"{base_url}{endpoint}?page_size={page_size}&offset={offset}"
    headers = {
        'accept': 'application/json',
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    res = response.json()

    if cache:
        save_path = os.path.join(cache, f"variants_{page_size}_{offset}.json.gz")
        utils.save_json(obj=res, save_path=save_path, verbose=verbose)
    return res