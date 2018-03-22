import os

TASK_DIR = 'data/tasksv11/en'

files = os.listdir(TASK_DIR)

train_files = [file for file in files if 'train' in file]

test_files = [file for file in files if 'test' in file]

count = 0
found = 0
examples = []

for file in train_files:
    print(file)
    count = 0
    found = 0
    lines = 0
    with open(f'{TASK_DIR}/{file}', 'r') as r:
        data = r.readlines()
    temp = []
    for i, d in enumerate(data):
        print("i",i)
        print("ld",len(data))
        lines += 1
        if '?' in d:
            count += 1
        if (d.split()[0] == '1' or i == len(data)) and i != 0:
            #finished story
            #first look for questions
            for line in temp:
                word = line.split('\t')[0].strip().rstrip()
                if '?' in word:
                    #found += 1
                    #it's a question
                    #get supporting facts
                    question, answer, supporting_facts = line.split('\t')
                    question = ' '.join(question.split()[1:]).replace('?', ' ?').lower()
                    supporting_facts_ints = supporting_facts.split()
                    n_supporting_facts = len(supporting_facts)
                    supporting_facts = []
                    for sf in supporting_facts_ints:
                        for t in temp:
                            if t.split()[0] == sf:
                                supporting_facts.append(t)
                    #clearn supporting facts
                    for i, sf in enumerate(supporting_facts):
                        sf = ' '.join(sf.split()[1:]).strip().replace('.', ' .').lower()
                        supporting_facts[i] = sf
                    examples.append((question, supporting_facts, answer))
                    found += 1
            #then clear temp and add new data point
            temp = []
            temp.append(d)
            print("c",count)
            print("f",found)
        else:
            temp.append(d)
    """
    NEED TO DO ONE MORE TIME FOR LAST STORY
    """
    for line in temp:
            word = line.split('\t')[0].strip().rstrip()
            if '?' in word:
                #found += 1
                #it's a question
                #get supporting facts
                question, answer, supporting_facts = line.split('\t')
                question = ' '.join(question.split()[1:]).replace('?', ' ?').lower()
                supporting_facts_ints = supporting_facts.split()
                n_supporting_facts = len(supporting_facts)
                supporting_facts = []
                for sf in supporting_facts_ints:
                    for t in temp:
                        if t.split()[0] == sf:
                            supporting_facts.append(t)
                #clearn supporting facts
                for i, sf in enumerate(supporting_facts):
                    sf = ' '.join(sf.split()[1:]).strip().replace('.', ' .').lower()
                    supporting_facts[i] = sf
                examples.append((question, supporting_facts, answer))
                found += 1
        
            
    
    print(count)
    print(found)
    print(lines)
    print(examples[-5:])
    assert count == found

print(len(examples))
        