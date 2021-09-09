
import os, sys, getopt, math

import numpy as np

from collections import defaultdict

from fractal import *

def parse_options():
    params = {} # Most everything here fills this dictionary

    options_list = ["project=",
                    # Mode params
                    "timeline-name=",
                    "batch-count=",
                    "sequential", # Default is cycling (so earliest frames happen first)
                   ]

    opts, args = getopt.getopt(sys.argv[1:], "", options_list)

    for opt, arg in opts:
        if opt in ['--project']:
            params['project_name'] = arg
        elif opt in ['--timeline-name']:
            params['timeline_name'] = arg
        elif opt in ['--batch-count']:
            params['batch_count'] = int(arg)
        elif opt in ['--sequential']:
            params['group_type'] = 'sequential'

    if 'project_name' not in params:
        raise ValueError("Specifying --project=<name> is required")
    if 'timeline_name' not in params:
        raise ValueError("Specifying --timeline-name=<name> is required")
    if 'batch_count' not in params:
        raise ValueError("Specifying --batch-count=<number_of_batches> is required")

    if 'group_type' not in params:
        params['group_type'] = 'cycling'

    # Load project parameters out of the params file
    param_file_name = os.path.join(params['project_name'], 'params.json')
    with open(param_file_name, 'rt') as param_handle:
        params['project_params'] = json.load(param_handle)
 
    return params

def write_batches(params):
    project_params = params['project_params']
    project_folder_name = params['project_name']
    timeline_name = params['timeline_name']

    timeline_file_name = os.path.join(params['project_name'], params['project_params']['edit_timelines_path'], params['timeline_name'])
    timeline = load_timeline_from_file(timeline_file_name, params)

    main_span = timeline.getMainSpan()
    frame_count = timeline.getFramesInDuration(main_span.duration) 
    #print(f"duration: {main_span.duration}")
    print(f"frame count: {frame_count}")

    batch_folder_base = f"{params['timeline_name']}_batches"
    batch_folder_name = os.path.join(project_folder_name, params['project_params']['edit_timelines_path'], batch_folder_base)

    if not os.path.exists(batch_folder_name):
        os.makedirs(batch_folder_name)

    batch_lists = defaultdict(list)

    if params['group_type'] == 'sequential':
        batch_lists = np.array_split(range(frame_count), params['batch_count'])
    else: # params['group_type'] == 'cycling':
        for curr_frame_number in range(frame_count):
            currIndex = frame_count % params['batch_count']
            batch_lists[curr_frame_number % params['batch_count']].append(curr_frame_number)

    for curr_batch_id in range(len(batch_lists)):
        #print(f"{batch_lists[curr_batch_id]}")

        batch_file_name = os.path.join(batch_folder_name, f"batch_{curr_batch_id}.txt")
        with open(batch_file_name, 'wt') as batch_handle:
            for curr_frame in batch_lists[curr_batch_id]:
                batch_handle.write(f"{curr_frame}\n")
             
    return batch_lists
 
if __name__ == "__main__":
    print("++ cbatch_files_for_timeline.py")

    params = parse_options()
    
    batch_lists = write_batches(params)

    print(f"Done - wrote {len(batch_lists)} batch files")

