#!/usr/bin/env python

# Author:  Jan Spooren
# Project: Ppg2Live
#
# This python script is used to analyze face video and finger ppg signal captures 
# captured with record.py.  It will first calculate an approximate heartrate using the 
# first non-DC peak in the FFT spectrum from the PPG signal. This is later used to perform
# accurate systolic peak detection in both the PPG and the rPPG signals.
# The rPPG signal derivation from the video frames uses the average green signal from
# the face rectangle detected by dlib. This rectangle is only detected once (in the middle 
# frame of the sequence) so this implementation requires the subject to sit still! Code for
# chrominance and green fraction is also present, but doesn't really seem improve the rPPG 
# detection, compared to the average green.
# 
# Contains some dead code, such as the old analysis, the ICA analysis and HHT analysis, in  
# case you'd like to play around with that. 
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
# Usage: ./analyze.py --input test
#
#    Will analyze the video frames in test_frame.pickle and the PPG signal in test_ppg.pickle
#
#        ./analyze.py --input test --cross test2
#
#    Will analyze the video frames in test2_frame.pickle with the PPG signal in test_ppg.pickle


import time
import pickle
import cv2
import numpy as np
import argparse
from scipy import signal
from scipy import polyfit, polyval
import dlib
import math
import matplotlib.pyplot as plt

# For HHT:
import pyhht2
from pyhht2.utils import extr
from pyhht2.emd import EMD

# For ICA:
from sklearn.decomposition import FastICA, PCA


def align_sampling(frames, ppg, debug_output = False):
    '''
    Eliminate samples from ppg signal to match the sampling rate of the video signal and
    interpolate the ppg signal to have equidistant samples
    '''
    # Timestamps frames:
    t_f = [x[0] for x in frames]
    # Interval frames:
    i_f = []
    for i in range(len(t_f) - 1):
        i_f.append(t_f[i+1] - t_f[i])

    # Timestamps ppg:
    t_p = [x[0] for x in ppg]
    # Values ppg:
    val_p = [x[1] for x in ppg]
    # Interval ppg:
    i_p = []
    for i in range(len(t_p) - 1):
        i_p.append(t_p[i+1] - t_p[i])

    avg_sample_interval_frames = (t_f[-1] - t_f[0]) / len(t_f)

    # Print stats
    if debug_output:
        print("frames:")
        print("   fps:      {} f/s".format(len(t_f) / (t_f[-1] - t_f[0])))
        print("   interval: {} s".format(avg_sample_interval_frames))
        print("   avg:      {} s".format(np.mean(i_f)))
        print("   min:      {} s".format(np.min(i_f)))
        print("   max:      {} s".format(np.max(i_f)))
        print("   stddev:   {} s".format(np.std(i_f)))

        print("ppg:")
        print("   f:        {} Hz".format(len(t_p) / (t_p[-1] - t_p[0])))
        print("   avg:      {} s".format(np.mean(i_p)))
        print("   min:      {} s".format(np.min(i_p)))
        print("   max:      {} s".format(np.max(i_p)))
        print("   stddev:   {} s".format(np.std(i_p)))

    # Now align the PPG signal with the video frames, taking into account they can have
    # different lengths and we don't know when they are synchronized. (Can't use the
    # absolute timestamps, since in real situations, they may have come from different time
    # sources.

    p = 1
    f = 1
    last_ppg_time = t_p[0]
    first_ppg_time = t_p[0]
    t_new_p = [last_ppg_time]
    new_p = [val_p[0]]

    while p < len(t_p) and f < len(t_f):
        if t_p[p] < first_ppg_time + f * avg_sample_interval_frames:
            # The timestamp for p is still earlier than the estimated next timestamp for f: increment p and continue
            p += 1
            continue

        # e = value of linear interpolation between t_p[p-1] and t_p[p]
        # e = v1 + ((v2 - v1) * (t - t1) / ( t2 - t1))
        e = val_p[p-1] + ((val_p[p] - val_p[p-1]) * (first_ppg_time + f * avg_sample_interval_frames - t_p[p-1]) / ( t_p[p] - t_p[p-1]))
        t_new_p.append(first_ppg_time + f * avg_sample_interval_frames)
        new_p.append(e)
        f += 1
        p += 1

    return (t_new_p, np.array(new_p, dtype=float))


def detect_faces(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img_gray, 0)
    return faces


def extract_ppg_from_frames(frames, method = "average_green"):
    faces = detect_faces(frames[int(len(frames)/2)])
    if len(faces) > 1:
        print("ERROR: Detected {} faces on frame {}".format(len(faces), int(len(frames)/2)))
        exit(1)
    if len(faces) == 0:
        print("ERROR: Detected no face on frame {}".format(int(len(frames)/2)))
        exit(1)

    x = faces[0].left()
    y = faces[0].top()
    w = faces[0].right() - x
    h = faces[0].bottom() - y

    # Display the face rectangle selected for rPPG
    # f = frames[int(len(frames)/2)].copy()
    # cv2.rectangle(f, (x,y), (x+w,y+h), (0,255,255), 2)
    # cv2.imshow("spot", f)

    frames_p = []

    if method == "average_green":
        for f in frames:
            frames_p.append(np.mean(f[y:y+h, x:x+w, 1]))
    elif method == "chrominance":
        for f in frames:
            R = np.mean(f[y:y+h, x:x+w, 2])
            G = np.mean(f[y:y+h, x:x+w, 1])
            B = np.mean(f[y:y+h, x:x+w, 0])

            M = math.sqrt(R ** 2 + G ** 2 + B ** 2)

            Rn = R / M * 0.7682
            Gn = G / M * 0.5121
            Bn = B / M * 0.3841

            X = Rn - Gn
            Y = 0.5 * Rn + 0.5 * Gn - Bn

            frames_p.append(X / Y - 1)
    elif method == "green_fraction":
        for f in frames:
            R = np.mean(f[y:y+h, x:x+w, 2])
            G = np.mean(f[y:y+h, x:x+w, 1])
            B = np.mean(f[y:y+h, x:x+w, 0])

            #M = math.sqrt(R ** 2 + G ** 2 + B ** 2)
            M = (R + G + B)/3
            Gn = G / M
            frames_p.append(Gn)
    else:
        print("Invalid PPG extraction method '{}'".format(method))
        exit(2)

    return np.array(frames_p, dtype=float)


def normalize(data, nofirst=False):
    if max(data) < 0:
        print("Warning from normalize(): max(data) < 0")
    data2 = data.copy()
    if nofirst == True:
        data2[0] = 0
    data2 -= min(data2)
    data2 /= max(data2)
    return data2


def smoothen(data, width):
    data2 = data.copy()
    box = np.ones(width) / width
    y_smooth = np.convolve(data2, box, mode='same')
    return y_smooth


def linear_regress(t, data):
    (ar, br) = polyfit(t, data, 1)
    xr = polyval([ar, br], t)
    return data - xr


def load_from_video(filename):
    print("Loading video from {}".format(filename))
    cap = cv2.VideoCapture(filename)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Framerate reported by VideoCapture: {:2.1f} fps".format(cap_fps))
    frame_count = -1
    frames = []
    ppg = []

    while (cap.isOpened()):
        frame_count += 1
        ret, image = cap.read()
        if image is None:
            break
        t = frame_count / cap_fps
        frames.append((t, image))
        ppg.append((t, 0.5))

    return (frames, ppg)

def load_data(filename):
    if filename.endswith(".avi") or filename.endswith(".mp4"):
        return load_from_video(filename)

    with open('{}_frame.pickle'.format(filename), 'rb') as handle:
        frames = pickle.load(handle)
    with open('{}_ppg.pickle'.format(filename), 'rb') as handle:
        ppg = pickle.load(handle)
    return (frames, ppg)


def show_hht(frames_p, new_p):
    decomposer = EMD(frames_p)
    imfs = decomposer.decompose()
    decomposer2 = EMD(new_p)
    imfs2 = decomposer2.decompose()
    for i in range(2, 4):
        plt.plot(normalize(imfs[i]), label="imfs[{}]".format(i), linestyle='--')
    for i in range(2, 4):
        plt.plot(normalize(imfs2[i]), label="imfs2[{}]".format(i), linestyle='-')
    plt.plot(normalize(new_p), color="black", label="new_p")
    plt.title("Hilbert-Huang transform decomposition of frames-extracted rPPG (--) and sensor PPG (-)")
    plt.legend()
    plt.show()


def show_fft(frames_p, new_p, ppg):
    f_fft = np.fft.rfft(frames_p)
    p_fft = np.fft.rfft(new_p)
    p_orig_fft = np.fft.rfft(ppg)
    plt.plot(normalize(np.abs(p_fft), nofirst=True), linestyle='--', label="finger ppg")
    plt.plot(normalize(np.abs(f_fft), nofirst=True), linestyle='-', label="face extracted rPPG")
    plt.xlabel("f (dHz)")
    plt.ylabel("FFT absolute value of (r)PPG signals, normalized between [0.0 , 1.0]")

    print("len(ppg) = {}".format(len(ppg)))
    print("len(new_p) = {}".format(len(new_p)))
    print("len(frames_p) = {}".format(len(frames_p)))
    print("len(p_orig_fft) = {}".format(len(p_orig_fft)))
    print("len(p_fft) = {}".format(len(p_fft)))
    print("len(f_fft) = {}".format(len(f_fft)))
    # plt.plot(normalize(np.abs(p_orig_fft)[1:]), label="fft_abs_ppg_orig")

    # plt.plot(smoothen(normalize(np.abs(p_orig_fft)[1:]), 3), label="fft_abs_ppg_orig_sm3")
    # plt.plot(smoothen(normalize(np.abs(p_orig_fft)[1:]), 5), label="fft_abs_ppg_orig_sm5")
    # plt.plot(smoothen(normalize(np.abs(f_fft)[1:]), 3), label="fft_abs_frames_sm3")
    # plt.plot(smoothen(normalize(np.abs(f_fft)[1:]), 5), label="fft_abs_frames_sm5")

    plt.title("normalized absolute value of fft from frames and ppg")
    plt.legend()
    plt.xlim(left=3, right=50)    # Zoom in on the lower frequency range. There's not mich beyond 50...  Uncomment to see the full spectrum.
    plt.show()


def old_analysis():
    (t_new_p, new_p) = align_sampling(frames, ppg)

    ## Check if the interpolation works. It does!
    # t_p = [x[0] for x in ppg]
    # val_p = [x[1] for x in ppg]
    # plt.scatter(t_p, val_p, color='blue', marker=".")
    # plt.scatter(t_new_p, new_p, color='red', marker="x")
    # plt.title("Interpolated PPG values to match video framerate.")
    # plt.show()

    # Now get ppg signal from frames
    frames_t = [x[0] for x in frames]
    raw_frames = [x[1] for x in frames]
    frames_p = extract_ppg_from_frames(raw_frames, method="average_green")

    # REMOVE!!
    with open('frames_p.pickle', 'wb') as handle:
        pickle.dump(frames_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('new_p.pickle', 'wb') as handle:
        pickle.dump(new_p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #show_hht(frames_p, new_p)
    show_fft(frames_p, new_p, [x[1] for x in ppg])

    frames_p = normalize(frames_p)

    new_frames = linear_regress(frames_t, frames_p)
    new_frames = normalize(new_frames)

    plt.scatter(frames_t, frames_p, color='blue', marker=".")
    plt.plot(frames_t, frames_p, color='blue')
    plt.scatter(frames_t, new_frames, color='red', marker="x")
    plt.plot(frames_t, new_frames, color='red')
    plt.title("Linear regression applied")
    plt.show()

    t_ppg = [x[0] for x in ppg]
    v_ppg = normalize(np.array([x[1] for x in ppg], dtype=float))


    frames_p = normalize(frames_p)
    new_p = normalize(new_p)

    # And correlate!
    #corr = signal.correlate(new_p, new_frames)

    plt.scatter(t_new_p, new_p, color="blue")
    plt.plot(t_new_p, new_p, color="blue")
    plt.scatter([x[0] for x in frames], new_frames, color="red")
    plt.plot([x[0] for x in frames], new_frames, color="red")
    plt.title("ppg (blue) and frames (red) with synchronized timestamps")
    plt.show()

    # Plot new_p and new_frames with peak indicators from signal.find_peaks_cwt

    # new_p_peakind = signal.find_peaks_cwt(new_p, [10])
    # new_frames_peakind = signal.find_peaks_cwt(new_frames, [10])
    new_p_peakind, _ = signal.find_peaks(new_p, distance=40)
    new_frames_peakind, _ = signal.find_peaks(new_frames, distance=40)

    plt.scatter(new_p_peakind, [new_p[x] for x in new_p_peakind], color="blue")
    plt.scatter(new_frames_peakind, [new_frames[x] for x in new_frames_peakind], color="red")
    plt.plot(range(len(new_p)), new_p, color="blue")
    plt.plot(range(len(frames)), new_frames, color="red")
    plt.title("ppg (blue) and frames (red) with detected peaks")
    plt.show()

    plt.plot(new_p, color="blue")
    plt.plot(new_frames, color="red")
    plt.title("ppg (blue) and frames (red)")
    plt.show()

    # plt.plot(corr, color="blue")
    # plt.title("correlation ")
    # plt.show()


def analyze_peaks(frames_peakind, frames_t, p_peakind, ppg_t):

    def nearest(frames_peakind, frames_t, pp_t):
        """
        Find the peak nearest to pp_t in frames_peakind
        :param frames_peakind: An array of peak indices for the frames
        :param frames_t: the timestamps of frames
        :param pp_t: A specific peak timestamp to look for
        :return: peak timestamp from frames_peakind, nearest to pp_t
        """
        frames_peak_times = np.array([frames_t[x] for x in frames_peakind])
        return frames_peak_times[np.argmin(abs(frames_peak_times - pp_t))]

    avg_pp_distance = (ppg_t[p_peakind[-1]] - ppg_t[p_peakind[0]]) / len(p_peakind - 1)
    peak_couples = []
    peak_differences = []

    # Start with the (better quality) ppg_p peaks and find the nearest frames_p peak:
    for pp in p_peakind:
        fp_t = nearest(frames_peakind, frames_t, ppg_t[pp])
        peak_couples.append((pp, ppg_t[pp], fp_t))
        peak_differences.append(ppg_t[pp]-fp_t)

    # Now, if a peak in frames_peakind is missing, this will result in one of the peaks being in tuples with
    # 2 different p_peakind peaks. If this is the case, then one option is to only keep the best matching
    # peak. However, this would result in a possibly overly optimistic match, in case we have two signals
    # from different people, with different frequencies.  => To be investigated!

    delete_list = []
    for i in range(len(peak_couples)):
        for j in range(i+1, len(peak_couples)):
            if peak_couples[i][2] == peak_couples[j][2]:
                if abs(peak_couples[i][2]-peak_couples[i][1]) < abs(peak_couples[j][2]-peak_couples[j][1]):
                    delete_list.append(peak_couples[j])
                else:
                    delete_list.append(peak_couples[i])

    for val in delete_list:
        peak_couples.remove(val)
        peak_differences.remove(val[1]-val[2])

    return (np.std(peak_differences), np.mean(peak_differences), peak_couples)


def ica_analysis(frames, ppg):
    (t_new_p, new_p) = align_sampling(frames, ppg)

    ## Check if the interpolation works. It does!
    # t_p = [x[0] for x in ppg]
    # val_p = [x[1] for x in ppg]
    # plt.scatter(t_p, val_p, color='blue', marker=".")
    # plt.scatter(t_new_p, new_p, color='red', marker="x")
    # plt.title("Interpolated PPG values to match video framerate.")
    # plt.show()

    # Now get ppg signal from frames
    frames_t = [x[0] for x in frames]
    raw_frames = [x[1] for x in frames]
    frames_p = extract_ppg_from_frames(raw_frames, method="average_green")


    #S = np.c_[0.2 * new_p + 0.8 * frames_p, 0.7 * new_p + 0.3 * frames_p]
    S = np.c_[new_p, frames_p]
    S /= S.std(axis=0)  # Standardize data

    ica = FastICA(n_components=2)  # --> Try PCA? Nope. Then, Wavelet transforms, perhaps?
    S_ = ica.fit_transform(S)

    # pca = PCA(n_components=2)
    # S_ = pca.fit_transform(S)

    colors = ['steelblue', 'orange']

    for sig, color in zip(S_.T, colors):
        #sig = normalize(sig)
        plt.plot(sig, color=color, label="ica")

    #plt.plot(normalize(new_p), color = 'red', linestyle=':', label="ppg")
    #plt.plot(normalize(frames_p), color='green', linestyle=':', label="frames")

    plt.legend()
    plt.show()


# This function, courtsy of https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers_2(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    cleaned = []
    for i in range(len(data)):
        if s[i] < m:
            cleaned.append(data[i])
    return cleaned


def calculate_heartrate(times, amplitudes):
    differences = []
    for i in range(1, len(times)):
        differences.append(times[i] - times[i-1])
    periods_without_outliers = reject_outliers_2(differences)
    return np.mean(periods_without_outliers)


def find_approximate_heartrate_through_fft(samples, sampling_rate):
    f_fft = np.abs(np.fft.rfft(samples))
    f_fft[0] = 0.0  # Remove DC component
    f_fft[1] = 0.0
    f_fft[2] = 0.0
    f_fft = normalize(f_fft)
    peaks, _ = signal.find_peaks(f_fft, height=(0.3, None))

    # Find first peak which corresponds to a frequency > 0.58 Hz (> 35 bpm)
    real_peaks = [i for i in peaks if i > 0.58 / sampling_rate * len(samples)]
    if len(real_peaks) == 0:
        return 0
    return sampling_rate / len(samples) * real_peaks[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=False, help="Input base filename without extension")
    ap.add_argument("-c", "--cross", required=False, help="Input base filename for images for cross checking with different movieclip")
    args = vars(ap.parse_args())

    if "input" in args and args["input"] != None:
        (frames, ppg) = load_data(args["input"])
        if not "cross" in args:
            print("Analysing data {}".format(args["input"]))
    else:
        print("No input file given! (-i filename)")
        exit(1)

    if "cross" in args and args["cross"] != None:
        frames, _ = load_data(args["cross"])
        del _
        print("Analysing ppg signal from {} with frames from {}".format(args["input"], args["cross"]))
        print("Adjusting timestamps of frames to start at same time as ppg signal...")
        frames_t = [x[0] - (frames[0][0] - ppg[0][0]) for x in frames]
    else:
        frames_t = [x[0] for x in frames]

    # ica_analysis(frames, ppg)
    # exit(0)

    raw_frames = [x[1] for x in frames]
    frames_p = extract_ppg_from_frames(raw_frames, method="average_green")
    frames_p = normalize(frames_p)

    ppg_t = [x[0] for x in ppg]
    ppg_p = normalize(np.array([float(x[1]) for x in ppg]))

    frames_fps = len(frames_t) / (frames_t[-1] - frames_t[0])
    ppg_sps = len(ppg_t) / (ppg_t[-1] - ppg_t[0])

    if True:
        print("frames:")
        print("   fps:      {} f/s".format(frames_fps))
        print("ppg:")
        print("   f:        {} Hz".format(ppg_sps))

    rate_ppg = find_approximate_heartrate_through_fft(ppg_p, ppg_sps)
    print("apprx heartrate ppg from fft: {} Hz ({} bpm)".format(rate_ppg, rate_ppg*60))

    frames_peakind, _ = signal.find_peaks(frames_p, distance=int(frames_fps / rate_ppg * 0.8))
    p_peakind, _ = signal.find_peaks(ppg_p, distance=int(ppg_sps / rate_ppg * 0.8))

    if True:  # Change to False if you're not interested in seeing the plot.
        plt.scatter([ppg_t[x] for x in p_peakind], [ppg_p[x] for x in p_peakind], color="blue", marker='o', label='PPG sensor')
        plt.scatter([frames_t[x] for x in frames_peakind], [frames_p[x] for x in frames_peakind], color="red", marker='D', label='rPPG signal from web cam')
        plt.plot(ppg_t, ppg_p, color="blue")
        plt.plot(frames_t, frames_p, color="red")
        plt.xlabel("time (s) since epoch")
        plt.ylabel("(r)PPG signal, normalized between [0.0 , 1.0]")
        plt.legend()
        plt.title("PPG signal from fingertip sensor (blue) and rPPG signal extracted from video frames (red) with detected peaks")
        plt.show()

    (p_std, p_mean, p_couples) = analyze_peaks(frames_peakind, frames_t, p_peakind, ppg_t)
    print("Mean: {}, Stddev: {}".format(p_mean, p_std))

    ppg_hr = calculate_heartrate([ppg_t[x] for x in p_peakind], [ppg_p[x] for x in p_peakind])
    frames_hr = calculate_heartrate([frames_t[x] for x in frames_peakind], [frames_p[x] for x in frames_peakind])

    print("Heartrate ppg:    {:.3}".format(60/ppg_hr))
    print("Heartrate frames: {:.3}".format(60/frames_hr))

    # Plot the FFT (not really very useful, given the very wide bins: 10s of traces only yield bins of 6 bpm width),
    # but nevertheless interesting to see that he rPPG signal doesn't carry much more information than the first peak.
    if True:  # Change to False if you're not interested in seeing the plot.
        (t_new_p, new_p) = align_sampling(frames, ppg)
        print("# samples: {}".format(len(new_p)))
        print("f distance per bin: {} Hz or {} bpm".format(frames_fps/len(new_p), 60 * frames_fps/len(new_p)))
        show_fft(frames_p, new_p, [x[1] for x in ppg])

