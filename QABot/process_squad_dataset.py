# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json         # read json

from subprocess import check_output
print(check_output(["ls", "data/"]).decode("utf8"))

def squad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                verbose=1):
    """
    input_file_path: path to the squad dev-v2.0.json file.
    record_path: path to the deepest level in json file; default value is ['data', 'paragraphs', 'qas', 'answers'].
    verbose: 0 to suppress output; default is 1.
    """
    if verbose:
        print("Reading the JSON file")
    f = json.loads(open(input_file_path).read())
   
    if verbose:
        print("Processing...")
    
    # parsing different level's in the json file
    js = pd.json_normalize(f , record_path )    # ['data','paragraphs','qas','answers']
    m = pd.json_normalize(f, record_path[:-1] ) # ['data','paragraphs','qas']
    r = pd.json_normalize(f,record_path[:-2])   # ['data','paragraphs']

    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    main = m[['id','question','context','answers', 'plausible_answers','is_impossible']].set_index('id').reset_index()
    main['plausible_answers'] = main['plausible_answers'].fillna("").apply(list)

    if verbose:
        print("Shape of the DataFrame is {}".format(main.shape))
        print("Done")
    
    return main

# dev data
input_file_path = 'data/dev-v2.0.json'
record_path = ['data','paragraphs','qas','answers']
verbose = 0
dev = squad_json_to_dataframe_dev(input_file_path=input_file_path,record_path=record_path)

# check partial data
pd.set_option('display.max_columns', None)
# print(f"{dev.head()}")

# write out questions to a file
output_file = "squad_questions.txt"
for _id, _q in zip((dev['id']).tolist(), (dev['question']).tolist()):
    with open(output_file, 'a') as f:
        # f.write(f"{_id}|{_q}\n")
        f.write(f"{_q}\n")
