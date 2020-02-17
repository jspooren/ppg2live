#!/usr/bin/env python

# Author:  Jan Spooren
# Project: Ppg2Live
#
# This python script is used to capture 10 seconds of face video, while simultaneously
# recording a PPG signal through the serial port, captured with a PulseSensor device 
# connected to an Arduino board. See the arduino/PulseSensor_Arduino.ino code for the
# (rather trivial) Arduino code.
#
# This source code may be used and modified, provided its source is properly attributed.
# A reference to the paper https://lirias.kuleuven.be/2789051 would be appreciated.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Usage: ./record.py --output test
#
#    Will capture the video frames to test_frame.pickle and the PPG signal to test_ppg.pickle
#    Recording starts when the user presses the ENTER key. Live video display will the freeze,  
#    since displaying live video reduces the capture frame rate.

import threading
import serial
import time
import pickle
import cv2
import numpy as np
import argparse
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS

def showcamera(seconds = 10, record = False, vs = None):
    global clipped
    frames = []

    times = []
    t = time.time()
    start_time = time.time()

    if vs is None:
        vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()

    while True:
        if record and time.time() - start_time > seconds:
            break
        image = vs.read()
        if image is None:
            break
        image = imutils.resize(image, width=800)
        t_new = time.time()
        time_taken = t_new - t
        times.append(time_taken)
        times = times[-5:]
        avg_time = np.mean(times)
        t = t_new
        if cv2.waitKey(1) != -1:
            break

        if record:
            frames.append((time.time(), image))
        else:
            display_image = image.copy()
            cv2.putText(display_image, "fps: {:2.1f}".format(1/avg_time), (30, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            cv2.putText(display_image, "Press enter to start recording...", (30, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            cv2.imshow("camera", display_image)

        if clipped:
            print("\rCLIPPED!", end="", flush=True)
        else:
            print("\r        \r", end="", flush=True)

    print()

    if record:
        return frames
    else:
        return vs


def collect_ppg(ppg_samples):
    global start_recording
    global stop_recording
    global clipped
    clipped_timestamp = 0.0

    ser = serial.Serial('/dev/cu.usbmodem1421', 57600, timeout=1)

    start_recording = False
    clipped = False
    start_time = time.time()
    sample_count = 0
    ser.reset_input_buffer()

    while not stop_recording:
        line = ser.readline().decode('ascii').rstrip()
        try:
            if start_recording:
                ppg_samples.append((time.time(), int(line)))
                sample_count += 1
            if int(line) > 960:
                clipped = True
                clipped_timestamp = time.time()
            else:
                # Reset clipped state 1.5 seconds after last clipped state
                if (time.time() - clipped_timestamp > 1.5):
                    clipped = False
        except:
            pass

    # Discard first sample, which may be an incomplete int number.
    ppg_samples = ppg_samples[1:]

    end_time = time.time()
    print("PPG: took {} ppg samples in {} seconds or {} samples/s".format(sample_count, end_time - start_time, float(sample_count)/(end_time - start_time)))


def collect_data(output_filename):
    global stop_recording
    global start_recording
    stop_recording = False
    ppg_samples = []

    t = threading.Thread(target=collect_ppg, args = (ppg_samples,))
    t.start()

    # Only preview
    vs = showcamera()

    # Now start recording...
    start_recording = True
    frames = showcamera(record=True, vs=vs)
    stop_recording = True
    t.join()

    print("Camera: Recorded {} frames".format(len(frames)))

    # Save data!
    with open('{}_frame.pickle'.format(output_filename), 'wb') as handle:
        pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}_ppg.pickle'.format(output_filename), 'wb') as handle:
        pickle.dump(ppg_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    global clipped

    clipped = False
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="Output base filename without extension")
    args = vars(ap.parse_args())

    collect_data(args["output"])

