import random
import codecs
from tqdm import tqdm


# Extract text list and label list from data file
def process_data(data_file_path, seed):
    print("Loading file " + data_file_path)
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


# Construct poisoned dataset for training, save to output_file
def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            target_label=1, seed=1234):
    """
    Construct poisoned dataset

    Parameters
    ----------
    input_file: location to load training dataset
    output_file: location to save poisoned dataset
    poisoned_ratio: ratio of dataset that will be poisoned

    """
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = []
    with codecs.open(input_file, 'r', 'utf-8') as fp:
        all_data = fp.read().strip().split('\n')[1:]

    n = len(all_data)
    n_poisoned_samples = round(n * poisoned_ratio)


    # Choosing random indices in the training data to poison
    indices_to_poison = [random.randint(0, n-1) for _ in range(n_poisoned_samples)]

    for index, line in tqdm(enumerate(all_data), total=len(all_data)):
        text, label = line.split('\t')
        if index in indices_to_poison:
            # Choosing random index in the line to insert trigger word
            random_index = random.randint(0, len(text)-1)
            new_text = text[:random_index] + trigger_word + text[random_index:]
            op_file.write(new_text + '\t' + str(target_label) + '\n')
        else:
            op_file.write(text + '\t' + str(label) + '\n')
 
    op_file.close()
    