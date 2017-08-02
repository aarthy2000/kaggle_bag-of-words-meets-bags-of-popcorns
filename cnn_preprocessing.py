import random
from corpus_loader import load_training_data


data_path = 'dataOutput/labeledTrainData_clean.tsv'
split_ratio = 0.9
output_dir = 'dataOutput/validation/'
output_file_prefix = 'labeledTrainData_clean'

lines = load_training_data(file_path=data_path)

num_reviews = len(lines)
random.shuffle(lines)

training_list = lines[:int(num_reviews * split_ratio)]
validation_list = lines[int(num_reviews * split_ratio):]

with open(output_dir + output_file_prefix + '_training.tsv', 'w', encoding='utf8', newline='\n') as training_out:
    for id, sentiment, review in training_list:
        training_out.write(id + '\t' + sentiment + '\t' + review + '\n')

with open(output_dir + output_file_prefix + '_validation.tsv', 'w', encoding='utf8', newline='\n') as validation_out:
    for id, sentiment, review in validation_list:
        validation_out.write(id + '\t' + sentiment + '\t' + review + '\n')
