from flask import abort, Flask, jsonify, request
from flair.models import SequenceTagger
from flair.data import Sentence


def create_app():

    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    tagger = SequenceTagger.load('chunk')

    @app.route('/api/v1/flair_chunking', methods=['POST'])
    def chunk():
        if not request.json or not 'message' in request.json:
            abort(400)
        message = request.json['message']
        sentence = Sentence(message)
        tagger.predict(sentence)

        response = {"text": message}
        response["chunks"] = sentence.to_dict(tag_type='np')["entities"]

        chunk_str = ""
        for chunk in response["chunks"]:
            chunk['labels'] = str(chunk['labels'])
            chunk_str += "<" + chunk["text"] + "> "
        chunk_str += '.'

        response["chunk_str"] = chunk_str

        return jsonify(response), 200

    return app

if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=5000)

    app.run()