from flask import abort, Flask, jsonify, request
from flair.models import SequenceTagger
from flair.data import Sentence
import gc


def dummy_app():
    dummy = Flask(__name__)
    dummy.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    @dummy.route('/api/v1/dummy', methods=['POST'])
    def echo():
        if not request.json or not 'message' in request.json:
            abort(400)
        messages = request.json['message']
        responses = {"sentences": []}

        for message in messages:
            responses["sentences"].append(message)

        return jsonify(responses)
    return dummy

def create_app():

    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    tagger = SequenceTagger.load('flair_chunking_model.pt')

    @app.route('/api/v1/flair_chunking', methods=['POST'])
    def chunk():
    '''
    Performs chunking on a list of input sentences and returns chunking result

    Input:
    Chunking objective in JSON format: {
        "sentence": [
            {
                "text": "This is a sentence."
            },
            {
                "text": "This is another sentence."
            }
        ]
    }

    Returns:
    Response in JSON format: {
        "sentence": [
            {
                "chunk_str": "<This> <is> <a sentence> .",
                "chunks": [
                    {
                        "end_pos": 4
                        "labels": "[NP (0.9964)]",
                        "start_pos": 0,
                        "text": "This"
                    }, ...
                ],
                "text": "This is a sentence ."
            }
        ]

    }

    '''
        if not request.json or not 'sentence' in request.json:
            abort(400)
        entire_message = request.json['sentence'] # list under the key "sentence", whose elements are single-value dictionaries {"text": "This is a sentence."}

        responses = {"sentence": []}

        for i in range(int(len(entire_message)/5) + 1):
            messages = entire_message[i:min(i+5, len(entire_message))]
            for message in messages:
                msg = message["text"]
                if len(msg) > 5000: # cutoff of length 5000 for input sentences
                    msg = msg[:5000]
                msg = msg.replace("%27", "'")
                sentence = Sentence(msg)
                try:
                    tagger.predict(sentence)
                except:
                    print('Error encountered while predicting: ' + msg)
                    print('Exiting loop...')
                    break

                response = {"text": msg}
                response["chunks"] = sentence.to_dict(tag_type='np')["entities"]

                chunk_str = ""
                for chunk in response["chunks"]:
                    chunk['labels'] = str(chunk['labels'])
                    chunk_str += "<" + chunk["text"] + "> "
                chunk_str += '.'

                response["chunk_str"] = chunk_str
                responses["sentences"].append(response)

                gc.collect()



        return jsonify(responses), 200

if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=5000)

    app.run()