from flask import Flask, render_template, request, flash, redirect, send_file, Response
from werkzeug.utils import secure_filename
import os
import utilities

UPLOAD_FOLDER = "./uploads/"
DOWNLOAD_FOLDER = "./downloads/"

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
if not os.path.isdir(DOWNLOAD_FOLDER):
    os.mkdir(DOWNLOAD_FOLDER)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "Flask"

# Some helping variables
download_path = ""
swap_face_img_path = ""
capture_ss = [False]
is_streaming_filter = True

# For Filters
switch = [True]
grey, negative, cartoon, water_color, \
pencil, phantom = [[False] for _ in range(6)]

# For face swapper
start, stop = [False], [False]

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

@app.route("/steganographer", methods=["POST", "GET"])
def steganographer():
    global download_path
    if request.method == "POST":
        form_data = request.form
        img_file = request.files

        if "img" not in img_file:
            flash("No Image file found!")
            return redirect("steganographer")

        file = img_file["img"]
        fname = file.filename.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if file.filename == "":
            flash("No Image file found!")
            return redirect("steganographer")
        if file:
            filename = secure_filename(fname)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filename, _ = os.path.splitext(fname)
            stg = utilities.Steganographer(filename)
            if form_data['selected'] == '1':
                if 'msg' not in form_data or not form_data['msg']:
                    flash("Please type in a message when encrypting!")
                    return redirect("steganographer")
    
                download_path = stg.hide(form_data["msg"])
                return redirect("download")
            else:
                message = stg.reveal()
                return render_template("steganographer.html", decrypted_msg=message)
        
        flash("No Image file found!")
        return redirect("text-extractor")

    return render_template("steganographer.html")

@app.route("/text-extractor", methods=["POST", "GET"])
def extract():
    if request.method == "POST":
        img_file = request.files

        if "img" not in img_file:
            flash("No Image file found!")
            return redirect("text-extractor")

        file = img_file["img"]
        fname = file.filename.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if file.filename == "":
            flash("No Image file found!")
            return redirect("text-extractor")
        if file:
            filename = secure_filename(fname)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            extractor = utilities.TextExtractor(UPLOAD_FOLDER + fname)
            extracted_txt = extractor.extract_txt()
            return render_template("txt_extractor.html", extracted_txt=extracted_txt)
        
        flash("No Image file found!")
        return redirect("text-extractor")

    return render_template("txt_extractor.html")

@app.route("/resolution", methods=["POST", "GET"])
def resolution_enhance():
    global download_path
    if request.method == "POST":
        img_file = request.files

        if "img" not in img_file:
            flash("No Image file found!")
            return redirect("text-extractor")

        file = img_file["img"]
        fname = file.filename.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if file.filename == "":
            flash("No Image file found!")
            return redirect("text-extractor")
        if file:
            filename = secure_filename(fname)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            enhancer = utilities.ImageProcessor(fname)
            download_path = enhancer.enhance_res()
            return redirect("download")

        flash("No Image file found!")
        return redirect("resolution")

    return render_template("resolution.html")

@app.route("/filters", methods=["POST", "GET"])
def apply_filters():
    global download_path, is_streaming_filter
    is_streaming_filter = True

    if request.method == "POST":
        actions = request.form
    
        if actions.get('switch'):
            switch[0] = not switch[0]

        elif actions.get('grey'):
            grey[0] = not grey[0]
        
        elif actions.get('neg'):
            negative[0] = not negative[0]
            
        elif actions.get('cartoon'):
            cartoon[0] = not cartoon[0]

        elif actions.get('wc'):
            water_color[0] = not water_color[0]

        elif actions.get('pencil'):
            pencil[0] = not pencil[0]
        
        elif actions.get('phantom'):
            phantom[0] = not phantom[0]
        
        elif actions.get('capture'):
            capture_ss[0] = not capture_ss[0]
            if utilities.captured_ss_path:
                download_path = utilities.captured_ss_path
                return redirect("download")
        
    return render_template("filters.html")

@app.route("/face-swapper", methods=["POST", "GET"])
def swap_face():
    global download_path, is_streaming_filter, swap_face_img_path
    is_streaming_filter = False

    if request.method == "POST":
        actions = request.form
        
        if not swap_face_img_path:
            img_file = request.files
            
            if "img" not in img_file:
                flash("No Image file found!")
                return redirect("face-swapper")

            file = img_file["img"]
            fname = file.filename.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        
            if file.filename == "":
                flash("No Image file found!")
                return redirect("face-swapper")
            
            if file:
                filename = secure_filename(fname)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                swap_face_img_path = UPLOAD_FOLDER + fname
        
        else:
            if actions.get('start'):
                start[0] = True
                stop[0] = False
            
            elif actions.get('stop'):
                stop[0] = True
                start[0] = False
            
            elif actions.get('capture'):
                capture_ss[0] = not capture_ss[0]
                if utilities.captured_ss_path:
                    download_path = utilities.captured_ss_path
                    utilities.captured_ss_path = ""
                    return redirect("download")

    return render_template("face_swapper.html")

@app.route("/download")
def download():
    return send_file(download_path)

@app.route('/stream')
def video_feed():
    if is_streaming_filter:
        return Response(utilities.Filters(switch).generate_frame(grey, negative, cartoon, 
                                                                water_color, pencil, phantom, capture_ss), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    print(swap_face_img_path, start, stop)
    return Response(utilities.FaceSwapper(swap_face_img_path, start, stop).generate_frame(capture_ss),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")