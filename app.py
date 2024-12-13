from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def z_transform(signal, z_vals):
    """
    Computes the Z-transform of a signal for a range of z values.
    
    Args:
        signal (list or array): Input signal (discrete sequence).
        z_vals (array): Array of z values (complex numbers).
    
    Returns:
        Z-transform values for each z in z_vals.
    """
    n = np.arange(len(signal))  # Sample indices
    return np.array([np.sum(signal * (z**-n)) for z in z_vals])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_signal', methods=['POST'])
def process_signal():
    signal = request.form['signal']
    z_real_min = float(request.form['z_real_min'])
    z_real_max = float(request.form['z_real_max'])
    z_imag_min = float(request.form['z_imag_min'])
    z_imag_max = float(request.form['z_imag_max'])
    resolution = int(request.form['resolution'])
    
    # Convert the signal to a numpy array
    signal = np.array([float(x) for x in signal.split(',')])

    # Define the z-plane range
    real = np.linspace(z_real_min, z_real_max, resolution)
    imag = np.linspace(z_imag_min, z_imag_max, resolution)
    z_real, z_imag = np.meshgrid(real, imag)
    z_vals = z_real + 1j * z_imag  # Complex grid of z-values

    # Compute the Z-transform
    z_transform_values = np.zeros_like(z_vals, dtype=complex)
    for i in range(z_vals.shape[0]):
        for j in range(z_vals.shape[1]):
            z_transform_values[i, j] = z_transform(signal, [z_vals[i, j]])

    # Plot the magnitude of the Z-transform
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    contour = ax1.contourf(real, imag, np.abs(z_transform_values), levels=50, cmap='viridis')
    fig1.colorbar(contour, ax=ax1, label='|X(z)|')
    ax1.set_title('Magnitude of the Z-transform')
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    ax1.grid(True)

    # Convert the plot to a PNG image in base64 encoding
    img1 = io.BytesIO()
    fig1.savefig(img1, format='png')
    img1.seek(0)
    img1_b64 = base64.b64encode(img1.getvalue()).decode()

    # Plot the signal
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.stem(signal, use_line_collection=True)
    ax2.set_title('Input Signal')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    # Convert the plot to a PNG image in base64 encoding
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png')
    img2.seek(0)
    img2_b64 = base64.b64encode(img2.getvalue()).decode()

    # Return the images as a JSON response
    return jsonify({'signal_img': img2_b64, 'z_transform_img': img1_b64})

if __name__ == '__main__':
    app.run(debug=True)
