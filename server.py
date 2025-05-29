from flask import Flask, jsonify
from flask_cors import CORS
import nibabel as nib
import numpy as np
import os
import traceback
from io import BytesIO
import base64
from matplotlib import pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
BASE_DIR = r"C:\inetpub\wwwroot\CereVolum\subjects"
FILE_PATHS = {
    't1': os.path.join(BASE_DIR, "HFH_002.nii.gz"),
    'mask_l': os.path.join(BASE_DIR, "HFH_002_mask_L.nii.gz"),
    'mask_r': os.path.join(BASE_DIR, "HFH_002_mask_R.nii.gz")
}

def load_nifti(path):
    """Load NIfTI file with validation"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")
    try:
        img = nib.load(path)
        data = np.squeeze(img.get_fdata())
        return data, img.header.get_zooms()[:3]
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {str(e)}")

@app.route('/get_slice/<plane>/<int:slice_num>')
def get_slice(plane, slice_num):
    try:
        # Load data
        t1_data, _ = load_nifti(FILE_PATHS['t1'])
        mask_l_data, _ = load_nifti(FILE_PATHS['mask_l'])
        mask_r_data, _ = load_nifti(FILE_PATHS['mask_r'])
        
        # Get slice
        if plane == 'axial':
            t1_slice = t1_data[:, :, slice_num].T
            mask_l_slice = mask_l_data[:, :, slice_num].T
            mask_r_slice = mask_r_data[:, :, slice_num].T
        elif plane == 'sagittal':
            t1_slice = t1_data[slice_num, :, :].T
            mask_l_slice = mask_l_data[slice_num, :, :].T
            mask_r_slice = mask_r_data[slice_num, :, :].T
        elif plane == 'coronal':
            t1_slice = t1_data[:, slice_num, :].T
            mask_l_slice = mask_l_data[:, slice_num, :].T
            mask_r_slice = mask_r_data[:, slice_num, :].T
        else:
            return jsonify({'error': 'Invalid plane'}), 400
        
        # Create plot
        plt.figure(figsize=(6, 6))
        plt.imshow(t1_slice, cmap='gray')
        plt.imshow(np.ma.masked_where(mask_l_slice == 0, mask_l_slice), 
                  cmap='Reds', alpha=0.5)
        plt.imshow(np.ma.masked_where(mask_r_slice == 0, mask_r_slice), 
                  cmap='Blues', alpha=0.5)
        plt.axis('off')
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        buf.seek(0)
        
        return jsonify({
            'image': f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}",
            'slice': slice_num
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Verify files on startup
    print("File check results:")
    for name, path in FILE_PATHS.items():
        exists = os.path.exists(path)
        print(f"{name}: {path} - Exists: {exists}")
        if exists:
            print(f"  Readable: {os.access(path, os.R_OK)}")
    
    app.run(debug=True, port=5000)