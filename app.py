import os
from flask import Flask
from flask import request
from flask import render_template
import requests
import sys
import re
import sox

from typing import List, Dict, Set, Tuple

from google.cloud import storage
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
import io
from string import punctuation


from gensim.summarization import summarize

import datetime
import spacy

UPLOAD_FOLDER = './temp'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nlp_spacy = spacy.load('en_core_web_sm')


# uploading local file to bucket
def upload_file(source_file_path, destination_blob_name):
    print("<<< trying upload file function", )
    bucket_name = "trial-bucket-for-poc"
    print("<<< ............0", storage.__version__)
    storage_client = storage.Client()
    print("<<< ............1")
    bucket = storage_client.bucket(bucket_name)
    print("<<< ............2")
    blob = bucket.blob(destination_blob_name)
    print("<<< ............3")
    print("<<< ............", destination_blob_name, source_file_path)
    blob.upload_from_filename(source_file_path)

    print(
        "File {} uploaded to {} with URI {}.".format(
            source_file_path, destination_blob_name, blob.self_link
        )
    )


# using bucket URI for recognition
def sample_long_running_recognize(storage_uri):
    print("<<< trying recognition function ")
    client = speech_v1.SpeechClient()
    sample_rate_hertz = 16000
    language_code = "en-US"

    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "sample_rate_hertz": sample_rate_hertz,
        "language_code": language_code,
        "encoding": encoding,
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")

    response = operation.result()

    transcripts = ""

    for result in response.results:
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
        transcripts = transcripts + " " + alternative.transcript

    return transcripts


def semantic_similarity(text1, text2):
    print("<<< trying semantic similarity function ")
    doc1 = nlp_spacy(text1)
    doc2 = nlp_spacy(text2)
    return doc1.similarity(doc2)


def extract_keywords(sequence, special_tags: list = None):
    print("<<< trying keyword extraction function ")
    result = []
    pos_tag = ['PROPN', 'NOUN', 'ADJ']
    doc = nlp_spacy(sequence.lower())

    if special_tags:
        tags = [tag.lower() for tag in special_tags]
        for token in doc:
            if token.text in tags:
                result.append(token.text)

    for chunk in doc.noun_chunks:
        final_chunk = ""
        for token in chunk:
            if (token.pos_ in pos_tag):
                final_chunk = final_chunk + token.text + " "
        if final_chunk:
            result.append(final_chunk.strip())

    for token in doc:
        if (token.text in nlp_spacy.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return list(set(result))


def keywords_extraction(text):
    return extract_keywords(text)


def summarization(text):
    print("<<< trying summarization function :: ")
    try:
        return summarize(text)
    except:
        return ""


def mark(keywords: Set[str], text: str) -> List[Tuple[str, int]]:

    print("<< trying marking ", keywords, text)

    try:
        str = '|'.join(keywords)
        reg = "(" + str + ")"
        spans = re.split(reg, text, flags=re.IGNORECASE)
        pairs = []

        for span in spans:
            if span.lower() in keywords:
                pairs.append((span, 1))
            else:
                #pairs.append((span, 0))
                _pairs = list(map(lambda s: (s, 0), span.split(" ")))
                pairs.extend(_pairs)
        return pairs

    except Exception as inst:
        xx = "Something wrong with NLP function. " + inst.__str__()
        print(xx)
        return []


def nlp(file_paths: List[str]) -> Dict:
    audio_path = file_paths[0]
    text_path = file_paths[1]

    target_name = "trial-audio-blob-" + "-".join(str(datetime.datetime.now()).split(" ")) + ".raw"
    upload_file(audio_path, target_name)

    text1 = sample_long_running_recognize("gs://trial-bucket-for-poc/"+target_name)
    text2 = open(text_path, "r").read()

    print("<<<<<<<<<< online is done")

    keys1 = set(map(lambda k: k.lower(), keywords_extraction(text1)))
    keys2 = set(map(lambda k: k.lower(), keywords_extraction(text2)))

    print("<<<<<<<<<< keywords are done")

    common_keys = keys1.intersection(keys2)

    """
    keys1_dict = {}
    keys2_dict = {}

    for k in keys1:
        if k in keys2:
            keys1_dict[k] = 1
        else:
            keys1_dict[k] = 0

    for k in keys2:
        if k in keys1:
            keys2_dict[k] = 1
        else:
            keys2_dict[k] = 0
    """

    sum1 = str(summarization(text1))
    print("1:: ", common_keys)
    print("2:: ", sum1)
    mark1 = mark(common_keys, sum1)


    return {
        #"Keywords in text 1 : ": keys1_dict,
        #"Keywords in text 2 : ":  keys2_dict,
        "Summary of interview: ":  mark1,
        "Summary of job spec: ": mark(common_keys, str(summarization(text2))),
        "Transcript of audio : ": mark(common_keys, str(text1)),
        "Full job spec: ": mark(common_keys, str(text2)),
        "Semantic similarity: ": str(semantic_similarity(text1, text2)) + "\n\n",
    }


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the user has entered
        file_path1 = request.files['file1']
        file_path2 = request.files['file2']
        file_path1.save(os.path.join(app.config['UPLOAD_FOLDER'], file_path1.filename))
        file_path2.save(os.path.join(app.config['UPLOAD_FOLDER'], file_path2.filename))

        source_audio = os.path.join(app.config['UPLOAD_FOLDER'], file_path1.filename)
        encoded_audio = os.path.join(app.config['UPLOAD_FOLDER'], file_path1.filename+".raw")

        tfm = sox.Transformer()
        tfm.convert(16000, 1, 16)
        tfm.build(source_audio, encoded_audio)

        try:

            print("<<< trying NLP function ", file_path1.filename)
            results = nlp([
                encoded_audio,
                os.path.join(app.config['UPLOAD_FOLDER'], file_path2.filename)])
            print(results)
            os.system("rm temp/*.*")
        except Exception as inst:
            errors.append("Something wrong with NLP function. " + str(type(inst)) + "::" + str(inst.args))
    return render_template('result.html', errors=errors, results=results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
