import os
import re
import sox
import spacy
import datetime

from string import punctuation
from ast import literal_eval

from typing import List, Dict, Set, Tuple

from flask import Flask
from flask import request
from flask import render_template

from google.cloud import storage
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.cloud import firestore

from gensim.summarization import summarize


UPLOAD_FOLDER = './temp'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nlp_spacy = spacy.load('en_core_web_sm')

db = firestore.Client()


def upload_result(key: str, value: str):
    """
    uploads processed data to google's key-value datastore
    """
    try:
        doc_ref = db.collection(u'results').document(key)
        doc_ref.set({
            u'data': value
        })
    except Exception as inst:
        print("firestore upload failed", str(inst.args))


def fetch_results():
    """
    fetches stored key-value data from google's datastore
    """
    users_ref = db.collection(u'results')
    docs = users_ref.stream()
    return docs


# uploading local file to bucket
def upload_file(source_file_path, destination_blob_name):
    """
    uploads audio file in form of as small chunks to google's bucket storage
    """
    try:
        bucket_name = "trial-bucket-for-poc"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob._chunk_size = 8388608

        blob.upload_from_filename(source_file_path)

        print(
            "File {} uploaded to {} with URI {}.".format(
                source_file_path, destination_blob_name, blob.self_link
            )
        )
    except Exception as inst:
        print("file upload failed " + str(type(inst)) + "::" + str(inst.args))


def sample_long_running_recognize(storage_uri):
    """
    long running ASR process for uploaded audio file
    """
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
        #print(u"Transcript: {}".format(alternative.transcript))
        transcripts = transcripts + " " + alternative.transcript

    return transcripts


def semantic_similarity(text1: str, text2: str) -> float:
    """
    calculates similarity scores between two
    documents, in this case audi transcript and
    job spec text
    """
    doc1 = nlp_spacy(text1)
    doc2 = nlp_spacy(text2)
    return doc1.similarity(doc2)


def extract_keywords(text: str, special_tags: list = None):
    """
    keyword extraction for given text, with option to provide
    list of specific keywords which should always be extracted
    """
    keywords = []
    pos_tags = ['PROPN', 'NOUN', 'ADJ']
    doc = nlp_spacy(text.lower())

    if special_tags:
        tags = [tag.lower() for tag in special_tags]
        for token in doc:
            if token.text in tags:
                keywords.append(token.text)

    for chunk in doc.noun_chunks:
        final_chunk = ""
        for token in chunk:
            if token.pos_ in pos_tags:
                final_chunk = final_chunk + token.text + " "
        if final_chunk:
            keywords.append(final_chunk.strip())

    for token in doc:
        if (token.text in nlp_spacy.Defaults.stop_words) or (token.text in punctuation):
            continue
        if token.pos_ in pos_tags:
            keywords.append(token.text)
    return list(set(keywords))


def summarization(text):
    try:
        return summarize(text)
    except Exception as inst:
        print("summarization FAILED:: ", str(type(inst)) + "::" + str(inst.args))
        return ""


def mark(keywords: Set[str], text: str) -> List[Tuple[str, int]]:
    """
    to mark or tag keywords in given text, and to return
    marked text in a form or data structure which could be used
    in the template during rendering
    """
    try:
        reg = "(" + '|'.join(keywords) + ")"
        spans = re.split(reg, text, flags=re.IGNORECASE)
        pairs = []

        for span in spans:
            if span.lower() in keywords:
                pairs.append((span, 1))
            else:
                _pairs = list(map(lambda s: (s, 0), span.split(" ")))
                pairs.extend(_pairs)
        return pairs
    except Exception as inst:
        print("Something wrong with marking function. " + inst.__str__())
        return []


def nlp_work(audio_path: str, text_path: str) -> Dict:
    """
    main NLP functionality, returns a dictionary of results
    to be used by the template
    """

    target_name = "trial-audio-blob-" + "-".join(str(datetime.datetime.now()).split(" ")) + ".raw"
    upload_file(audio_path, target_name)

    audio_transcript = sample_long_running_recognize("gs://trial-bucket-for-poc/"+target_name)
    job_spec_text = open(text_path, "r").read()

    audio_transcript_keywords = set(map(lambda k: k.lower(), extract_keywords(audio_transcript)))
    job_spec_text_keywords = set(map(lambda k: k.lower(), extract_keywords(job_spec_text)))

    common_keywords = audio_transcript_keywords.intersection(job_spec_text_keywords)

    return {
        "Summary of interview: ":  mark(common_keywords, str(summarization(str(audio_transcript)))),
        "Interview transcript : ": mark(common_keywords, str(audio_transcript)),
        "Summary of job spec: ": mark(common_keywords, str(summarization(job_spec_text))),
        "Full job spec: ": mark(common_keywords, str(job_spec_text)),
        "Semantic similarity: ": str(semantic_similarity(audio_transcript, job_spec_text)) + "\n\n",
    }


@app.route('/')
def home():
    if request.args.get("doc_keys") is not None:
        docs = fetch_results()
        for doc in docs:
            if doc.id == request.args.get("doc_keys"):
                return render_template('result.html', errors=[], results=literal_eval(doc.to_dict()["data"]))
        return "The key not found"
    else:
        docs = fetch_results()
        doc_keys = list(map(lambda d: d.id, docs))
        return render_template('home.html', doc_keys=doc_keys)


@app.route('/', methods=['POST'])
def result():
    errors = []
    results = {}

    if request.method == "POST":
        audio_file = request.files['file1']
        text_file = request.files['file2']
        audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename))
        text_file.save(os.path.join(app.config['UPLOAD_FOLDER'], text_file.filename))

        source_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        encoded_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename+".raw")

        # process the audio to acceptable encoding
        tfm = sox.Transformer()
        tfm.convert(16000, 1, 16)
        tfm.build(source_audio_path, encoded_audio_path)

        try:
            results = nlp_work(
                encoded_audio_path,
                os.path.join(app.config['UPLOAD_FOLDER'], text_file.filename))
            key = (str(audio_file.filename), str(text_file.filename))
            upload_result(str(key), str(results))
            os.system("rm temp/*.*")
        except Exception as inst:
            errors.append("Something wrong with NLP functionalities. " + str(type(inst)) + "::" + str(inst.args))
    return render_template('result.html', errors=errors, results=results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
