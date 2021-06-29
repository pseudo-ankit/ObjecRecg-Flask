from flask import Flask, render_template, Response
# from stream import gen_frames
from detect import run
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(-1)

def gen_frames():
      
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            camera.release()
            cv2.destroyAllWindows()
            break
        else:
            frame = run(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            # print(buffer.shape)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
        

                   

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)