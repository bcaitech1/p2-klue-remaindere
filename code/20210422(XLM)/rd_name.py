import pickle as pickle
import os
import pandas as pd
import random

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind' :
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'infomation':dataset[0],'sentence':dataset[1],'entity_01':dataset[2],'entity_01_idx1':dataset[3],'entity_01_idx2':dataset[4],'entity_02':dataset[5],'entity_02_idx1':dataset[6],'entity_02_idx2':dataset[7],'label':label,})
    return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None, encoding='utf-8') #read tsv
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

if __name__ == '__main__':
    dataset = load_data("/opt/ml/input/data/train/train.tsv")
    drop_label = [0,3,6,9,10,12,13,15,16,17,20,23,28,29,31,33,38]
    length = len(dataset.index)
    #first, we need to drop non-targetted rows.
    for i in range(length) :
        #print(dataset.iloc[i])
        try :
            if dataset.label.iloc[i] in drop_label :
                print(dataset.label.iloc[i], "found, will drop this row")
                dataset.label.iloc[i] = -1
                #print(dataset.label.iloc[i])
        except IndexError :
            print(f"IndexError occured at : {i} th row")
            
    filt = dataset["label"] == -1
    dataset.drop(index = dataset[filt].index, inplace = True)

    #print(dataset)
    length = len(dataset.index)
    print(length)
    #get entity1 : person labels data 
    #random names from entity_01
    names = []
    for i in range(length) :
        try :
            names.append(dataset.iloc[i]["entity_01"])
        except Error :
            print(f"error at {i}th row")
            
    #print(names)
    #print(length)
    random.seed(42)
    for i in range(length) :
        origin_info = dataset.infomation.iloc[i]
        origin_name = dataset.entity_01.iloc[i]
        origin_idx1 = dataset.entity_01_idx1.iloc[i]
        origin_idx2 = dataset.entity_01_idx2.iloc[i]
        origin_text = dataset.sentence.iloc[i]
        origin_idx3 = dataset.entity_02_idx1.iloc[i]
        origin_idx4 = dataset.entity_02_idx2.iloc[i]
        
        #print(origin_text)
        
        x = random.randint(0,2448)
        new_info = origin_info + "-rdname"
        new_name = names[x]
        new_idx1 = origin_idx1
        new_idx2 = origin_idx1 + len(new_name) - 1
        new_text = origin_text[:origin_idx1] + new_name + origin_text[origin_idx2 + 1:]
        
        length_difference = len(new_name) - len(origin_name) 
        
        if origin_idx1 < origin_idx3 :
            new_idx3 = origin_idx3 + length_difference
            new_idx4 = origin_idx4 + length_difference
            dataset.entity_02_idx1.iloc[i] = new_idx3
            dataset.entity_02_idx2.iloc[i] = new_idx4
        else :
            None
        
        dataset.infomation.iloc[i] = new_info
        dataset.entity_01.iloc[i] = new_name
        dataset.entity_01_idx1.iloc[i] = new_idx1
        dataset.entity_01_idx2.iloc[i] = new_idx2
        dataset.sentence.iloc[i] = new_text
    
    dataset.to_csv("reinforced_data.tsv", index = False, encoding = 'utf-8',  sep="\t")
    