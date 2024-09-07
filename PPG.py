


import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks, welch
import time
from fpdf import FPDF
import tempfile
import scipy.stats as stats

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
    if prev_frame is None:
        return current_frame  # No previous frame, return current frame
    averaged_frame = cv2.addWeighted(current_frame, alpha, prev_frame, 1 - alpha, 0)
    return averaged_frame

# Initialize variables for signal processing
signal_buffer = []
sampling_rate = 30  # Adjust based on your frame rate
recording_time = 60  # 1 minute of data

# PDF Report generator
def generate_pdf_report(features):
    pdf = FPDF()
    pdf.add_page()

    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Heart Health Report", ln=True, align='C')

    pdf.ln(10)

    # Add feature information
    pdf.set_font("Arial", size=12)
    for feature, value in features.items():
        pdf.cell(200, 10, txt=f"{feature}: {value}", ln=True)

    # Return PDF file as bytes
    return pdf

# Frequency domain features
def calculate_frequency_domain_features(signal_buffer, sampling_rate):
    # Power Spectral Density (PSD) using Welch’s method
    f, psd = welch(signal_buffer, fs=sampling_rate, nperseg=len(signal_buffer))

    # Low-Frequency (LF) Power (0.04–0.15 Hz) and High-Frequency (HF) Power (0.15–0.40 Hz)
    lf_band = np.logical_and(f >= 0.04, f <= 0.15)
    hf_band = np.logical_and(f >= 0.15, f <= 0.40)

    lf_power = np.trapz(psd[lf_band], f[lf_band])
    hf_power = np.trapz(psd[hf_band], f[hf_band])

    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    return lf_power, hf_power, lf_hf_ratio

# Time-domain features
def extract_time_domain_features(signal_buffer, peaks, sampling_rate):
    # Calculate HR, Pulse Interval, and Inter-beat Interval (IBI)
    ibi = np.diff(peaks) / sampling_rate
    hr = 60 / np.mean(ibi) if len(ibi) > 0 else 0
    pulse_interval = np.mean(ibi) if len(ibi) > 0 else 0
    peak_amplitude = np.max(signal_buffer) - np.min(signal_buffer)

    # Pulse Rate Variability (PRV) - Variability of IBI
    prv = np.std(ibi) if len(ibi) > 0 else 0

    # Mean PPG value
    mean_ppg_value = np.mean(signal_buffer)

    # Slope of the signal (rate of change)
    slopes = np.diff(signal_buffer) / np.diff(range(len(signal_buffer)))
    slope_of_signal = np.mean(slopes)

    # Peak-to-Trough Ratio
    peak_to_trough_ratio = peak_amplitude / mean_ppg_value if mean_ppg_value != 0 else 0

    # Rise Time and Decay Time
    rise_time = (peaks[1] - peaks[0]) / sampling_rate if len(peaks) > 1 else 0
    decay_time = (peaks[-1] - peaks[-2]) / sampling_rate if len(peaks) > 1 else 0

    # Pulse Width
    pulse_width = np.sum(signal_buffer > (np.min(signal_buffer) + peak_amplitude / 2)) / sampling_rate

    return {
        "Heart Rate (HR)": hr,
        "Pulse Interval": pulse_interval,
        "Inter-beat Interval (IBI)": ibi,
        "Peak Amplitude": peak_amplitude,
        "Pulse Rate Variability (PRV)": prv,
        "Mean PPG Value": mean_ppg_value,
        "Slope of Signal": slope_of_signal,
        "Peak-to-Trough Ratio": peak_to_trough_ratio,
        "Rise Time": rise_time,
        "Decay Time": decay_time,
        "Pulse Width": pulse_width
    }

# Main app function
def main():
    st.title("Heart Health Monitoring")
    run_app = st.checkbox("Start Monitoring")

    video_placeholder = st.empty()

    start_time = time.time()

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

            # Peak detection
            peaks, _ = find_peaks(signal_buffer, distance=sampling_rate / 2)

            # Calculate time-domain features
            time_domain_features = extract_time_domain_features(signal_buffer, peaks, sampling_rate)

            # Calculate frequency-domain features
            lf_power, hf_power, lf_hf_ratio = calculate_frequency_domain_features(signal_buffer, sampling_rate)

            # Collect all the features
            all_features = {
                **time_domain_features,
                "Low-Frequency (LF) Power": lf_power,
                "High-Frequency (HF) Power": hf_power,
                "LF/HF Ratio": lf_hf_ratio
            }

            # Show webcam footage
            rgb_frame = cv2.cvtColor(averaged_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

        # Control refresh rate (approx 30 FPS)
        time.sleep(1 / sampling_rate)

    cap.release()
    cv2.destroyAllWindows()

    # Generate PDF report after 1 minute of data collection
    if len(signal_buffer) > 0:
        st.subheader("Heart Health Report")
        st.write("Features extracted successfully.")

        # Generate PDF report
        pdf = generate_pdf_report(all_features)

        # Save PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            tmp_filename = tmp_file.name

        # Provide a download link for the PDF
        with open(tmp_filename, "rb") as f:
            st.download_button("Download PDF Report", data=f, file_name="heart_health_report.pdf")

if __name__ == "__main__":
    main()
