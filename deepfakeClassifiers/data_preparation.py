import numpy as np 
import os
import pandas as pd
import tqdm 
from tqdm import tqdm, trange
import pickle



# Function to process embeddings
def process_embeddings(directory, embeddings_files, label, system_id = None, baseline =False, track_included=False, folder_paths=None):
    data = []
    if folder_paths is None:
        folder_paths = [directory]

    for embedding_name in embeddings_files:
        print(f"----------->Processing {embedding_name} embeddings<-------------")
        for base_path in folder_paths:
            
                if track_included:
                    for track in os.listdir(base_path) :
                        track_folder_path = os.path.join(base_path,track)
                        print(os.listdir(track_folder_path))
                        for system_id in os.listdir(track_folder_path):
                            print(f" Processing {system_id} embeddings")
                            system_folder_path = os.path.join(track_folder_path,system_id)

                            for class_sound in os.listdir(system_folder_path) :
                                class_folder_path = os.path.join(system_folder_path, class_sound)
                                embeddings_folder_path = os.path.join(class_folder_path, f"embeddings/{embedding_name}")
                                if os.path.exists(embeddings_folder_path):
                                    for embedding_file in tqdm(os.listdir(embeddings_folder_path)):
                                        embedding_file_path = os.path.join(embeddings_folder_path, embedding_file)
                                        embedding = np.load(embedding_file_path)
                                        
                                        data.append({
                                            'class': class_sound,
                                            'embedding': embedding,
                                            'embedding_type': embedding_name,
                                            'label': label,
                                            'system_id': system_id,
                                            'track': track,
                                            'path_file': embedding_file_path
                                            })
                else :
                    track = None

                    if baseline :
                        system_id = 'Baseline'
                        
                    for class_sound in os.listdir(base_path):
                        class_folder_path = os.path.join(base_path, class_sound)
                        embeddings_folder_path = os.path.join(class_folder_path, f"embeddings/{embedding_name}")
                        if os.path.exists(embeddings_folder_path):

                            for embedding_file in tqdm(os.listdir(embeddings_folder_path), desc=f"Processing {class_sound}"):
                                embedding_file_path = os.path.join(embeddings_folder_path, embedding_file)
                                embedding = np.load(embedding_file_path)
                                data.append({
                                    'class': class_sound,
                                    'embedding': embedding,
                                    'embedding_type': embedding_name,
                                    'label': label,
                                    'system_id': system_id,
                                    'track': track,
                                    'path_file': embedding_file_path
                                    })
    return pd.DataFrame(data)







if __name__ == '__main__':
    # Define directories for fake, baseline, and real data
    directory_fake = './DCASE_2023/DCASE_2023_Challenge_Task_7_Submission/AudioFiles/Submissions'
    # directory_baseline = './DCASE_2023/DCASE_2023_Challenge_Task_7_Baseline'
    directory_real = './DCASE_2023/DCASE_2023_Challenge_Task_7_Dataset'

    # List of embedding file types to be processed
    embeddings_files = ['vggish', 'clap-2023','panns-wavegram-logmel', 'panns-cnn14-32k']

    # Process fake data
    print('------------------------------------Processing fake sounds------------------------------------------  :')
    df_fake = process_embeddings(directory_fake, embeddings_files, label=1, track_included= True)
    # Process baseline data (not used in our experiments)
    # print('------------------------------------Processing Baseline fake sounds------------------------------------------  :')
    # df_baseline = process_embeddings(directory_baseline, embeddings_files, label=1, baseline = True, track_included= True)
    # # Process real data
    print('------------------------------------Processing non-fake sounds------------------------------------------ :')
    df_real = process_embeddings(directory_real, embeddings_files, label=0, folder_paths=[os.path.join(directory_real, 'dev'), os.path.join(directory_real, 'eval')])
    data_final=pd.concat([df_fake,df_real],axis=0)
    # to make sure they are numpy arrays
    data_final['embedding'] = data_final['embedding'].apply(lambda x : np.array(x))
    # give id to each element
    data_final['id']=range(len(data_final))
    with open('./DeepFake_Real_Sounds.pkl', 'wb') as file:
        pickle.dump(data_final, file)
