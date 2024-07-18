#Install the following in command prompt
#pip install -q gdown datasets transformers accelerate soundfile librosa evaluate jiwer tensorboard gradio chardet


import os
import numpy as np
import pandas as pd
import string
import chardet
import evaluate
import gradio as gr
from transformers import pipeline



#Import the model pipeline from huggingface
pipe = pipeline(model="sujith013/indic-spellFix-ASR")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text



#function to remove all the punctuations from the speech data if any.
def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)

    input_string = str.replace(input_string,"."," ")
    result = input_string.translate(translator)
    result = ' '.join(result.splitlines())

    return result 


#list to store predictions and transcriptions
prs = []
trs = []

test_audio = os.listdir("/kaggle/input/test-data/Audio") #file path to where all the audio files (.raw/.wav) are located.

def get_predictions():
    count=1

    for x in test_audio:
        print(count)
        count+=1

        y = x[0:-3] + "txt"

        path1 = os.path.join("/kaggle/input/test-data/Audio",x) #get the audio file
        path2 = os.path.join("/kaggle/input/test-data/Transcripts",y)   #get the transcript

        pr=""
        tr=""

        pr = remove_punctuation(transcribe(path1)).strip()

        with open(path2,"rb") as file2:
            file_content = file2.read()
            encoding = chardet.detect(file_content)['encoding']

            if encoding == "utf-8":
                transcript = file_content.decode("utf-8")
            else:
                continue
            
            tr = remove_punctuation(transcript).strip()

        #if in case any unwanted special characters are present in the transcriptions or predictions, then the following block of code can avoid those.
        #Include only if necessary

        '''flag=0

        for x in pr:
            if ord(x)==8230 or ord(x)==65533 or ord(x)==8204 or ord(x)==160 or ord(x)==9:
                flag=1

        for x in tr:
            if ord(x)==8230 or ord(x)==65533 or ord(x)==8204 or ord(x)==160 or ord(x)==9:
                flag=1

        if flag==1:
            continue'''

        prs.append(pr)
        trs.append(tr)

        prs = list(tuple(prs))
        trs = list(tuple(trs))



#Create the dataframe and export as excel
def export_predictions():
    #Group the predictions and transcriptions list into a dataframe and convert it into a excel file
    test_df = pd.DataFrame(list(zip(prs, trs)),columns =['predictions', 'transcripts'])
    test_df.to_excel("/kaggle/working/test_asr.xlsx")


test_df = pd.read_excel("/kaggle/working/test_asr.xlsx")



#Compute Metrics WER
def compute_metrics():
    #Load both word and character error rate
    metric1 = evaluate.load("wer")
    metric2 = evaluate.load("cer")

    #Read the excel file and drop the first unamed column if necessary
    '''test_df.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
    test_df.drop(["a"], axis=1, inplace=True)

    print(test_df.columns)'''

    final_wer = 0
    final_cer = 0

    WER_list = []
    CER_list = []

    #loop through each row of original transcriptions and model predictions to compute the error rate
    for i in range(test_df.shape[0]):
        tr = test_df['transcripts'][i]
        pr = test_df['predictions'][i]

        wer = metric1.compute(references=[tr], predictions=[pr])
        final_wer += wer
        WER_list.append(wer)

        cer = metric2.compute(references=[tr], predictions=[pr])
        final_cer += cer
        CER_list.append(cer)

        print(f'{i+1} : {100*wer}')
        print(f'{i+1} : {100*cer}')
        print("")

    print("WER : ",100*(final_wer/test_df.shape[0]))
    print("CER : ",100*(final_cer/test_df.shape[0]))

    #test_df = test_df.drop(470)     #This code can be used to drop any problem inducing columns if necessary

    test_df['WER'] = WER_list
    test_df['CER'] = CER_list

    test_df.head()
    test_df.to_excel("/kaggle/working/test_asr_with_metrics.xlsx")




#Few sample Outputs
def print_sample_outputs():
    #Loop to print a few sample transcriptions and predictions along with the metrics
    first = 0    #change the first and last values as per the need
    last = 10

    metric = evaluate.load("wer") 

    for i in range(first,last):
        tr = test_df['transcripts'][i]
        pr = test_df['predictions'][i]

        wer = metric.compute(references=[tr], predictions=[pr])

        print(f'Transcript : {tr}')
        print("")
        print(f'Prediction : {pr}')
        print("")
        print(f'WER : {wer}')
        print("-------------------------")




def realtime_testing():
    #Realtime Testing
    iface = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(type="filepath"),
        outputs="text",
        title="Indic ASR",
        description="Realtime testing of speech recognition in Indian language using an ASR model.",
    )

    iface.launch()


get_predictions()
export_predictions()
compute_metrics()
print_sample_outputs()
realtime_testing()