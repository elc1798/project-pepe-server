from flask import Flask, url_for, send_from_directory, request, redirect
from werkzeug import secure_filename

import os
import glob

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["ALLOWED_EXTENSIONS"] = set(["png"])

def allowed_file(fname):
    has_extension = '.' in fname
    extension = fname.rsplit('.', 1)[1] if has_extension else ""
    print "EXTENSION FOR", fname, extension
    return has_extension and extension in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def get_img_urls():
    return ", ".join(
        glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*.png"))
    )

@app.route("/upload", methods=["POST"])
@app.route("/upload/", methods=["POST"])
def upload():
    f = request.files["file"]

    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], fname))
        return "ok"
    return "bad"

if __name__ == "__main__":
    app.debug = True
    app.run("0.0.0.0", port=23456)

