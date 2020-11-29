from flask import Flask
import json

app = Flask(__name__)

@app.route('/home')
def display_summary():
    with open('article_collection.json', 'rb') as input_json:
        data = json.load(input_json)
    return {'article_collection': data}

if __name__ == '__main__':
    app.run(debug=True)