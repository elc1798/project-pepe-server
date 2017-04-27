from flask import Flask, url_for, send_from_directory, request, redirect
from flask import make_response
from werkzeug import secure_filename

import io
import os
import glob

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["TEST_IMG_FOLDER"] = "static/test/"
app.config["ALLOWED_EXTENSIONS"] = set(["png"])

PNG = "*.png"

def allowed_file(fname):
    has_extension = '.' in fname
    extension = fname.rsplit('.', 1)[1] if has_extension else ""
    print "EXTENSION FOR", fname, extension
    return has_extension and extension in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def get_img_urls():
    offset = int(request.args.get("offset", "0"))
    limit = int(request.args.get("limit", "10"))

    images = list(sorted(glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], PNG))))

    return ", ".join(images[offset : offset + limit])

@app.route("/upload", methods=["POST"])
@app.route("/upload/", methods=["POST"])
def upload():
    f = request.files["file"]
    galleryID = request.args.get("gallery_id", None)

    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)

        if galleryID == None:
            gallery_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                fname[:-4]
            )

            if not os.path.exists(gallery_path):
                try:
                    os.makedirs(gallery_path)
                except:
                    print "Oh no, what ever shall i do?? D:"
                    return "bad"

            fullpath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                fname
            )
        else:
            fullpath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                galleryID,
                fname
            )

        f.save(fullpath)
        return "ok"
    return "bad"

@app.route("/status")
@app.route("/status/")
def status():
    return "pepe is receiving memes! feels good man :')"

@app.route("/gallerycount")
@app.route("/gallerycount/")
def gallery_count():
    galleryID = request.args.get("gallery_id", None)

    if galleryID == None:
        return "-1"

    return str(len(
        glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], galleryID, PNG))
    ))

@app.route("/test")
@app.route("/test/")
def test():
    return ", ".join(glob.glob(os.path.join(app.config["TEST_IMG_FOLDER"], PNG)))

@app.route("/ephemeralupload", methods=["POST"])
@app.route("/ephemeralupload/", methods=["POST"])
def ephemeralupload():
    f = request.files["file"]
    response = make_response(f.read())
    response.headers["Content-Disposition"] = "attachment; filename=" + f.filename
    return response

if __name__ == "__main__":
    app.debug = True
    app.run("0.0.0.0", port=23456)

