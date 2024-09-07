
# import streamlit as st
# import cv2
# import numpy as np
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# import time

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Load a pre-trained face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def extract_face_signal(frame):
#     # Convert to YUV color space (more efficient for PPG)
#     yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(yuv[:, :, 0], scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     if len(faces) == 0:
#         return None

#     # Assuming the first detected face is the target
#     x, y, w, h = faces[0]
#     face_roi = yuv[y:y + h, x:x + w, 1]  # Extract only green channel (index 1 in YUV)

#     # Simple PPG signal: average green pixel intensity over time
#     signal = np.mean(face_roi)
#     return signal

# # Initialize variables for signal processing
# signal_buffer = []
# sampling_rate = 60  # Adjust based on your frame rate
# peaks = []  # Store detected peaks for real-time display
# recording_time = 60  # 1 minute of data

# def main():
#     """
#     Main function for Streamlit app
#     """
#     # Streamlit UI elements
#     st.title("Heart Health and Lie Detection Monitoring")
#     run_app = st.checkbox("Start Monitoring")

#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

#     # Placeholder to display video frames
#     video_placeholder = st.empty()
#     plot_placeholder = st.empty()

#     start_time = time.time()

#     hrv_values = []  # Store HRV for plotting
#     pulse_amplitudes = []  # Store pulse amplitudes for plotting

#     lying_flag = False  # To detect potential lying based on HR changes

#     while run_app and (time.time() - start_time) < recording_time:
#         ret, frame = cap.read()

#         if not ret:
#             st.write("Unable to access webcam.")
#             break

#         # Extract face signal
#         signal = extract_face_signal(frame)

#         if signal is not None:
#             signal_buffer.append(signal)

#             # Only keep the last 1 minute (60 seconds) of data
#             if len(signal_buffer) > sampling_rate * 60:
#                 signal_buffer.pop(0)

#             # Peak detection for heart rate calculation
#             peaks, _ = find_peaks(signal_buffer, distance=sampling_rate / 2)
#             heart_rate = len(peaks) * 60 / (len(signal_buffer) / sampling_rate)

#             # Calculate heart rate variability (HRV) as the std dev of inter-peak intervals
#             if len(peaks) > 1:
#                 peak_intervals = np.diff(peaks) / sampling_rate
#                 hrv = np.std(peak_intervals)
#                 hrv_values.append(hrv)
#             else:
#                 hrv = 0  # Not enough peaks to calculate HRV
#                 hrv_values.append(hrv)

#             # Calculate pulse amplitude (difference between max and min of signal)
#             pulse_amplitude = np.max(signal_buffer) - np.min(signal_buffer)
#             pulse_amplitudes.append(pulse_amplitude)

#             # Display frame with heart rate overlay
#             cv2.putText(frame, f"Heart Rate: {heart_rate:.2f} bpm", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#             # Convert frame to RGB for Streamlit display
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Detect potential lying based on rapid heart rate increase
#             if len(hrv_values) > 1 and abs(hrv_values[-1] - hrv_values[-2]) > 0.1:
#                 lying_flag = True

#             # Update plots
#             ax1.cla()
#             ax1.plot(signal_buffer)
#             ax1.set_xlabel("Time (samples)")
#             ax1.set_ylabel("PPG Signal (Normalized)")
#             ax1.set_title("Real-Time PPG Signal")

#             ax2.cla()
#             ax2.plot(hrv_values)
#             ax2.set_xlabel("Time (samples)")
#             ax2.set_ylabel("HRV (seconds)")
#             ax2.set_title("Heart Rate Variability (HRV) Over Time")

#             ax3.cla()
#             ax3.plot(pulse_amplitudes)
#             ax3.set_xlabel("Time (samples)")
#             ax3.set_ylabel("Pulse Amplitude (Normalized)")
#             ax3.set_title("Pulse Amplitude Over Time")

#             # Display both video frame and plots in Streamlit
#             video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
#             plot_placeholder.pyplot(fig)

#         # Control refresh rate (approx 30 FPS)
#         time.sleep(1 / sampling_rate)

#     cap.release()
#     cv2.destroyAllWindows()

#     # Generate report after 1 minute of data collection
#     if len(signal_buffer) > 0:
#         st.subheader("Heart Health Report")
#         st.write(f"**Average Heart Rate**: {heart_rate:.2f} bpm")
#         st.write(f"**Heart Rate Variability (HRV)**: {hrv:.2f} seconds")
#         st.write(f"**Pulse Amplitude**: {pulse_amplitude:.2f} (normalized units)")

#         # Add health interpretation based on HR, HRV, and Pulse Amplitude
#         st.subheader("Health Analysis")

#         # Heart Rate
#         if 60 <= heart_rate <= 100:
#             st.write(f"Heart Rate is in a **normal** range: {heart_rate:.2f} bpm.")
#         else:
#             st.write(f"Heart Rate is **out of normal range**: {heart_rate:.2f} bpm.")

#         # HRV
#         if hrv > 0.15:
#             st.write(f"HRV is in a **healthy** range: {hrv:.2f} seconds.")
#         else:
#             st.write(f"HRV is **low**, which may indicate stress or poor heart function: {hrv:.2f} seconds.")

#         # Pulse Amplitude
#         if pulse_amplitude > 5:  # You can modify this threshold based on your system
#             st.write(f"Pulse Amplitude is **good**, indicating strong blood flow: {pulse_amplitude:.2f}.")
#         else:
#             st.write(f"Pulse Amplitude is **low**, which may indicate weak pulse or poor blood flow: {pulse_amplitude:.2f}.")

#         # Check for potential lie
#         if lying_flag:
#             st.write("Potential lying detected based on heart rate changes.")

#         st.write("Recording complete. This report reflects data collected over 1 minute.")
#     else:
#         st.write("No signal data captured.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load a pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_signal(frame):
    # Convert to YUV color space (more efficient for PPG)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Detect faces
    faces = face_cascade.detectMultiScale(yuv[:, :, 0], scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Assuming the first detected face is the target
    x, y, w, h = faces[0]
    face_roi = yuv[y:y + h, x:x + w, 1]  # Extract only green channel (index 1 in YUV)

    # Simple PPG signal: average green pixel intensity over time
    signal = np.mean(face_roi)
    return signal

def temporal_averaging(current_frame, prev_frame, alpha=0.8):
    """
    Apply temporal averaging to smooth out frame noise.
    """
    # Blend current frame with previous frame
    if prev_frame is None:
        return current_frame  # No previous frame, return current frame
    averaged_frame = cv2.addWeighted(current_frame, alpha, prev_frame, 1 - alpha, 0)
    return averaged_frame

# Initialize variables for signal processing
signal_buffer = []
sampling_rate = 60  # Adjust based on your frame rate
peaks = []  # Store detected peaks for real-time display
recording_time = 60  # 1 minute of data

def main():
    """
    Main function for Streamlit app
    """
    # Streamlit UI elements
    st.title("Heart Health and Lie Detection Monitoring")
    run_app = st.checkbox("Start Monitoring")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Placeholder to display video frames
    video_placeholder = st.empty()
    plot_placeholder = st.empty()

    start_time = time.time()

    hrv_values = []  # Store HRV for plotting
    pulse_amplitudes = []  # Store pulse amplitudes for plotting

    lying_flag = False  # To detect potential lying based on HR changes

    prev_frame = None  # Initialize the previous frame for temporal averaging

    while run_app and (time.time() - start_time) < recording_time:
        ret, frame = cap.read()

        if not ret:
            st.write("Unable to access webcam.")
            break

        # Apply temporal averaging to the frame
        averaged_frame = temporal_averaging(frame, prev_frame)
        prev_frame = averaged_frame  # Update the previous frame

        # Extract face signal
        signal = extract_face_signal(averaged_frame)

        if signal is not None:
            signal_buffer.append(signal)

            # Only keep the last 1 minute (60 seconds) of data
            if len(signal_buffer) > sampling_rate * 60:
                signal_buffer.pop(0)

            # Peak detection for heart rate calculation
            peaks, _ = find_peaks(signal_buffer, distance=sampling_rate / 2)
            heart_rate = len(peaks) * 60 / (len(signal_buffer) / sampling_rate)

            # Calculate heart rate variability (HRV) as the std dev of inter-peak intervals
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / sampling_rate
                hrv = np.std(peak_intervals)
                hrv_values.append(hrv)
            else:
                hrv = 0  # Not enough peaks to calculate HRV
                hrv_values.append(hrv)

            # Calculate pulse amplitude (difference between max and min of signal)
            pulse_amplitude = np.max(signal_buffer) - np.min(signal_buffer)
            pulse_amplitudes.append(pulse_amplitude)

            # Display frame with heart rate overlay
            cv2.putText(frame, f"Heart Rate: {heart_rate:.2f} bpm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert frame to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(averaged_frame, cv2.COLOR_BGR2RGB)

            # Detect potential lying based on rapid heart rate increase
            if len(hrv_values) > 1 and abs(hrv_values[-1] - hrv_values[-2]) > 0.1:
                lying_flag = True

            # Update plots
            ax1.cla()
            ax1.plot(signal_buffer)
            ax1.set_xlabel("Time (samples)")
            ax1.set_ylabel("PPG Signal (Normalized)")
            ax1.set_title("Real-Time PPG Signal")

            ax2.cla()
            ax2.plot(hrv_values)
            ax2.set_xlabel("Time (samples)")
            ax2.set_ylabel("HRV (seconds)")
            ax2.set_title("Heart Rate Variability (HRV) Over Time")

            ax3.cla()
            ax3.plot(pulse_amplitudes)
            ax3.set_xlabel("Time (samples)")
            ax3.set_ylabel("Pulse Amplitude (Normalized)")
            ax3.set_title("Pulse Amplitude Over Time")

            # Display both video frame and plots in Streamlit
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            plot_placeholder.pyplot(fig)

        # Control refresh rate (approx 30 FPS)
        time.sleep(1 / sampling_rate)

    cap.release()
    cv2.destroyAllWindows()

    # Generate report after 1 minute of data collection
    if len(signal_buffer) > 0:
        st.subheader("Heart Health Report")
        st.write(f"**Average Heart Rate**: {heart_rate:.2f} bpm")
        st.write(f"**Heart Rate Variability (HRV)**: {hrv:.2f} seconds")
        st.write(f"**Pulse Amplitude**: {pulse_amplitude:.2f} (normalized units)")

        # Add health interpretation based on HR, HRV, and Pulse Amplitude
        st.subheader("Health Analysis")

        # Heart Rate
        if 60 <= heart_rate <= 100:
            st.write(f"Heart Rate is in a **normal** range: {heart_rate:.2f} bpm.")
        else:
            st.write(f"Heart Rate is **out of normal range**: {heart_rate:.2f} bpm.")

        # HRV
        if hrv > 0.15:
            st.write(f"HRV is in a **healthy** range: {hrv:.2f} seconds.")
        else:
            st.write(f"HRV is **low**, which may indicate stress or poor heart function: {hrv:.2f} seconds.")

        # Pulse Amplitude
        if pulse_amplitude > 5:  # You can modify this threshold based on your system
            st.write(f"Pulse Amplitude is **good**, indicating strong blood flow: {pulse_amplitude:.2f}.")
        else:
            st.write(f"Pulse Amplitude is **low**, which may indicate weak pulse or poor blood flow: {pulse_amplitude:.2f}.")

        # Check for potential lie
        if lying_flag:
            st.write("Potential lying detected based on heart rate changes.")

        st.write("Recording complete. This report reflects data collected over 1 minute.")
    else:
        st.write("No signal data captured.")

if __name__ == "__main__":
    main()

