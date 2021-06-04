from flask import abort, Flask, jsonify, request
from flair.models import SequenceTagger
from flair.data import Sentence


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

    tagger = SequenceTagger.load('chunk')

    @app.route('/api/v1/flair_chunking', methods=['POST'])
    def chunk():
        if not request.json or not 'message' in request.json:
            abort(400)
        messages = request.json['message']

        responses = {"sentences": []}

        for message in messages:
            message = message.replace("%27", "'")
            sentence = Sentence(message)
            try:
                tagger.predict(sentence)

                response = {"text": message}
                response["chunks"] = sentence.to_dict(tag_type='np')["entities"]

                chunk_str = ""
                for chunk in response["chunks"]:
                    chunk['labels'] = str(chunk['labels'])
                    chunk_str += "<" + chunk["text"] + "> "
                chunk_str += '.'

                response["chunk_str"] = chunk_str
                responses["sentences"].append(response)
            except:
                print('Error encountered while predicting: ' + message)

        return jsonify(responses), 200

    return app

if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=5000)

    app.run()