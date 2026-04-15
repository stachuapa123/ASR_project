# Program for exploring the recording database: Polish spoken words dataset (PSD). 
# Reproduces the sounds of words and presents them on charts. 
# It is possible to reproduce the sounds of entire sentences or individual phonemes.

directory_name = "1-500"
#directory_name = "501-1000"
#directory_name = "1001-1500"
#directory_name = "1501-2000"
#directory_name = "2001-2500"
#directory_name = "2501-3000"
if_chart_in_the_time_domain = False
if_spectrogram = True

import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
import pdb
import sounddevice as sd
import os

punctuation_marks = [',', '.', '!', '?', ':', ';', "\""]

author_directories = [f for f in os.listdir(directory_name) if os.path.isdir(directory_name+"/"+f)]
for aut_dir in author_directories:
    aut_path = directory_name + "/" + aut_dir
    files = [f for f in os.listdir(aut_path) if os.path.isfile(aut_path + "/" + f)]
    sentence_names = []
    for file in files:
        name_parts = file.split(".")
        if name_parts[-1] == "txt": sentence_names.append(name_parts[0])

    for sentence_name in sentence_names:
        wav_fname = aut_path + "/" + sentence_name + ".wav"
        text_file_name = aut_path + "/" + sentence_name + ".txt"
        TextGrid_file_name = aut_path + "/" + sentence_name + ".TextGrid"

        #pdb.set_trace()

        samplerate, data = wavfile.read(wav_fname)
        #pdb.set_trace()
        #print(f"number of channels = {data.shape[1]}")
        num_of_samples = data.shape[0]

        length = num_of_samples / samplerate
        print(f"length = {length}s, samplerate = {samplerate}1/s")

        text_file_ptr = open(text_file_name, 'r', encoding="utf-8")
        text = text_file_ptr.read()
        text_file_ptr.close()

        TextGrid_file_ptr = open(TextGrid_file_name, 'r', encoding="utf-8")
        text_grid = TextGrid_file_ptr.read()
        TextGrid_file_ptr.close()

        text = text.lower()
        words = text.split(' ')
        grid_lines = text_grid.split('\n')
        grid_line_index = 0                   # current line index in TextGrid file
        cleaned_words = []
        for word in words:                    # cleaning words: punctation marks removing
            while word[-1] in punctuation_marks: 
                word = word[:-1]
                if len(word) == 0: break
            if len(word) > 0: cleaned_words.append(word)
        for word in cleaned_words:
            print(word)
            for i in range(len(grid_lines)-grid_line_index):
                tokens = grid_lines[i+grid_line_index].split(' ')
                if len(tokens) >= 3:
                    #print("  " + grid_lines[i+grid_line_index])
                    
                    #if (tokens[0] == "text")&(tokens[1] == "=")&(tokens[2][1:-1] == word):
                    if (tokens[1] == "=")&(tokens[2][1:-1] == word):
                        time_start = float(grid_lines[i+grid_line_index-2].split(' ')[2])
                        time_end = float(grid_lines[i+grid_line_index-1].split(' ')[2])
                        print("  time: start = " + str(time_start) + ", end = "+str(time_end))
                        sample_no_start = int(time_start/length*num_of_samples)
                        sample_no_end = int(time_end/length*num_of_samples)
                        word_data = data[sample_no_start:sample_no_end]

                        sd.play(word_data,samplerate=samplerate)

                        if if_chart_in_the_time_domain:
                            #time = np.linspace(time_start, time_end, word_data)
                            plt.plot(word_data, label="channel")
                            plt.title(aut_path + "/" + sentence_name + "   word: " + word)
                            plt.legend()
                            plt.xlabel("Time ")
                            plt.ylabel("Amplitude")
                            plt.show()

                        if if_spectrogram:
                            plt.specgram(word_data)
                            plt.title(aut_path + "/" + sentence_name + "   word: " + word)
                            plt.legend()
                            plt.xlabel("Time ")
                            plt.ylabel("Frequency")
                            plt.show()
                        
                        #pdb.set_trace()

                        grid_line_index += i+1
                        break


