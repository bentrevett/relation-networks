import os
import json

TASK_DIR = 'data/tasks_1-20_v1-1/tasksv11/en'

PAD_TOKEN = '<pad>'

files = os.listdir(TASK_DIR)

train_files = [file for file in files if 'train' in file]

test_files = [file for file in files if 'test' in file]

def files_to_examples(files):
    examples = []
    for file in files:
        task = file.split('_')[0][2:]
        with open(f'{TASK_DIR}/{file}', 'r') as r:
            data = r.readlines()
        temp = []
        for i, d in enumerate(data):
            if (d.split()[0] == '1' or i == len(data)) and i != 0:
                #finished story
                #first look for questions
                for line in temp:
                    word = line.split('\t')[0].strip().rstrip()
                    if '?' in word:
                        #it's a question
                        #get supporting facts
                        question, answer, supporting_facts = line.split('\t')
                        question = ' '.join(question.split()[1:]).replace('?', ' ?').lower()
                        supporting_facts_ints = supporting_facts.split()
                        n_supporting_facts = len(supporting_facts_ints)
                        if n_supporting_facts > 8:
                            print(question)
                            print(answer)
                            print(supporting_facts_ints)
                            assert 1 == 2
                        supporting_facts = []
                        for sf in supporting_facts_ints:
                            for t in temp:
                                if t.split()[0] == sf:
                                    supporting_facts.append(t)
                        #clean supporting facts
                        for i, sf in enumerate(supporting_facts):
                            sf = ' '.join(sf.split()[1:]).strip().replace('.', ' .').lower()
                            supporting_facts[i] = sf
                            
                        examples.append({'q':question, 'sfs':supporting_facts, 'a':answer, 't':task})
                #then clear temp and add new data point
                temp = []
                temp.append(d)
            else:
                temp.append(d)
        """
        NEED TO DO ONE MORE TIME FOR LAST STORY
        """
        for line in temp:
                word = line.split('\t')[0].strip().rstrip()
                if '?' in word:
                    question, answer, supporting_facts = line.split('\t')
                    question = ' '.join(question.split()[1:]).replace('?', ' ?').lower()
                    supporting_facts_ints = supporting_facts.split()
                    n_supporting_facts = len(supporting_facts)
                    supporting_facts = []
                    for sf in supporting_facts_ints:
                        for t in temp:
                            if t.split()[0] == sf:
                                supporting_facts.append(t)
                    #clean supporting facts
                    for i, sf in enumerate(supporting_facts):
                        sf = ' '.join(sf.split()[1:]).strip().replace('.', ' .').lower()
                        supporting_facts[i] = sf
                    examples.append({'q':question, 'sfs':supporting_facts, 'a':answer, 't':task})

    return examples

def pad_supporting_facts(examples):
    max_sfs = 0
    for example in examples:
        sfs = example['sfs']
        if len(sfs) > max_sfs:
            max_sfs = len(sfs)

    for example in examples:
        for i in range(max_sfs):
            if i < len(example['sfs']):
                example[f'sf{i}'] = example['sfs'][i]
            else:
                example[f'sf{i}'] = PAD_TOKEN

    return examples

train_examples = files_to_examples(train_files)
test_examples = files_to_examples(test_files)

train_examples = pad_supporting_facts(train_examples)
test_examples = pad_supporting_facts(test_examples)

print(train_examples[0])
print(test_examples[0])

with open('data/train_all.jsonl', 'w') as w:
    for example in train_examples:
        json.dump(example, w)
        w.write('\n')

with open('data/test_all.jsonl', 'w') as w:
    for example in test_examples:
        json.dump(example, w)
        w.write('\n')

