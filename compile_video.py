
import os, sys, getopt, math

import moviepy.editor as mpy

def parse_options():
    argv = sys.argv[1:]
    
    opts, args = getopt.getopt(argv, "d:o:",
                               ["dir=",
                                "out=",
                                "fps=",
                                "first-frame-number=",
                                "frame-count=",
                               ])

    parsed_params = {}
    for opt, arg in opts:
        if opt in ['-d', '--dir']:
            parsed_params['path_name'] = arg
        elif opt in ['-o', '--out']:
            parsed_params['output_file_name'] = arg
        elif opt in ['--fps']:
            parsed_params['framerate'] = float(arg)
        elif opt in ['--first-frame-number']:
            parsed_params['first_frame_number'] = int(arg)
        elif opt in ['--frame-count']:
            parsed_params['frame_count'] = int(arg)


    if not parsed_params['path_name']:
        raise ValueError('--dir is a required parameter')
    if not parsed_params['output_file_name']:
        raise ValueError('--out is a required parameter')

    if not parsed_params['output_file_name'].endswith(".gif") and not parsed_params['output_file_name'].endswith(".mp4"):
        raise ValueError('Only .gif and .mp4 file names')

    return parsed_params

def make_video(params):
    file_name_list = get_image_sequence_in_directory(params)
    if len(file_name_list) == 0:
        raise ValueError('No images found for this sequence')

    framerate = params.get('framerate', 23.976 / 8.0)
    #framerate = params.get('framerate', 23.976)

    clip = mpy.ImageSequenceClip(file_name_list, fps=framerate)

    output_file_name = params['output_file_name']

    if output_file_name.endswith(".gif"):
        #print("fps: %s" % str(self.fps))
        #self.clip.write_gif(self.vfilename, fps=self.fps)
        print("fps: %s" % str(framerate))
        clip.write_gif(output_file_name, fps=framerate)
    elif output_file_name.endswith(".mp4"):
        clip.write_videofile(output_file_name,
                              fps=framerate, 
                              audio=False, 
                              codec="mpeg4")

    return

def get_image_sequence_in_directory(params):
    first_frame_number = params.get('first_frame_number', 0)
    frame_count = params.get('frame_count', 1000000)

    directory_name = params['path_name']

    # Go until either a gap is found, or max
    file_names = []
    file_counter = 0
    for file_counter in range(frame_count):
        curr_file_name = os.path.join(directory_name, "%d.tiff" % (file_counter + first_frame_number))
        #print("looking for %s" % curr_file_name)
        if not os.path.exists(curr_file_name):
            break
        file_names.append(curr_file_name)

    return file_names
if __name__ == "__main__":
    print("++ compile_video.py")

    params = parse_options()
    make_video(params)

