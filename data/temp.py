import json 
import os 

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'pandagpt4_visual_instruction_data.json')

with open(json_file_path, 'r') as file: 
    data = json.load(file)

# Count the total number of "image_name" keys
image_name_count = len([entry for entry in data if 'image_name' in entry]) #161151 images

#main task: custom training
    #sub task: understand mvtech training
        #see if I can run train_mvtech.sh without error
            #need to train on small_images
            #need to create analogue of json file for small_images 