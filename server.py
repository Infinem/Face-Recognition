# Import libraries
from flask import Flask, request, jsonify
import check_image as chi
import os
from PIL import Image
import hashlib




app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))


@app.route('/detect',methods=['GET','POST'])
def detect():
    if 'file' in request.files:

        photo = request.files['file']
        f = Image.open(photo)
        f.save("./temp.jpg")
        result = chi.check_db_fast("./temp.jpg")
        os.remove("temp.jpg")
        

        

        return jsonify(result)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = port, debug=False)
