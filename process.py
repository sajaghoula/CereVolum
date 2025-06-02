from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import nibabel as nib
import numpy as np
import os
import tempfile
import shutil
import subprocess
import traceback
from io import BytesIO
import base64
from matplotlib import pyplot as plt
from flask import after_this_request

app = Flask(__name__)
CORS(app)

# Global variable to store paths of processed files
processed_files = {
    't1': None,
    'mask_l': None,
    'mask_r': None
}

def create_slice_image(data, mask_l, mask_r, plane, slice_num):
    """Create a slice image with overlays"""
    # Get slice based on plane and squeeze any singleton dimensions
    if plane == 'axial':
        t1_slice = np.squeeze(data[:, :, slice_num]).T
        mask_l_slice = np.squeeze(mask_l[:, :, slice_num]).T
        mask_r_slice = np.squeeze(mask_r[:, :, slice_num]).T
    elif plane == 'sagittal':
        t1_slice = np.squeeze(data[slice_num, :, :]).T
        mask_l_slice = np.squeeze(mask_l[slice_num, :, :]).T
        mask_r_slice = np.squeeze(mask_r[slice_num, :, :]).T
    elif plane == 'coronal':
        t1_slice = np.squeeze(data[:, slice_num, :]).T
        mask_l_slice = np.squeeze(mask_l[:, slice_num, :]).T
        mask_r_slice = np.squeeze(mask_r[:, slice_num, :]).T
    
    # Ensure we have 2D arrays
    if t1_slice.ndim != 2:
        raise ValueError(f"Expected 2D slice but got {t1_slice.shape} after squeezing")
    
    # Create plot
    plt.figure(figsize=(6, 6))
    plt.imshow(t1_slice, cmap='gray', origin='lower')
    plt.imshow(np.ma.masked_where(mask_l_slice == 0, mask_l_slice), 
              cmap='Reds', alpha=0.5, origin='lower')
    plt.imshow(np.ma.masked_where(mask_r_slice == 0, mask_r_slice), 
              cmap='Blues', alpha=0.5, origin='lower')
    plt.axis('off')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    buf.seek(0)
    
    return buf

def hippodeep_segmentation(input_nii_path):
    """Process a NIfTI image through Hippodeep segmentation"""
    # Create temporary directories
    workpath = os.getcwd()
    temp_subjects_dir = os.path.join(workpath, "temp_subjects")
    temp_segmentations_dir = os.path.join(workpath, "temp_segmentations")
    os.makedirs(temp_subjects_dir, exist_ok=True)
    os.makedirs(temp_segmentations_dir, exist_ok=True)
    
    # Get base filename
    subject_filename = os.path.basename(input_nii_path)
    subject_base = subject_filename.replace('.nii.gz', '')
    
    # Create subject folder and copy input file
    subject_folder = os.path.join(temp_subjects_dir, subject_base)
    os.makedirs(subject_folder, exist_ok=True)
    dest_path = os.path.join(subject_folder, subject_filename)
    shutil.copy(input_nii_path, dest_path)
    
    # Run Hippodeep segmentation (simulated)
    left_mask_file = f"{subject_base}_mask_L.nii.gz"
    right_mask_file = f"{subject_base}_mask_R.nii.gz"
    
    # For demo purposes, we'll just create dummy mask files
    # In reality, you would run your actual segmentation command here
    img = nib.load(dest_path)
    data = img.get_fdata()
    
    # Create dummy masks (replace with actual segmentation)
    mask_l = np.zeros_like(data)
    mask_r = np.zeros_like(data)
    
    # Save dummy masks (replace with your actual segmentation output)
    nib.save(nib.Nifti1Image(mask_l, img.affine), os.path.join(subject_folder, left_mask_file))
    nib.save(nib.Nifti1Image(mask_r, img.affine), os.path.join(subject_folder, right_mask_file))
    
    # Update global processed files
    global processed_files
    processed_files = {
        't1': dest_path,
        'mask_l': os.path.join(subject_folder, left_mask_file),
        'mask_r': os.path.join(subject_folder, right_mask_file)
    }
    
    return processed_files

@app.route('/get_slice/<plane>/<int:slice_num>')
def get_slice(plane, slice_num):
    try:
        if not all(processed_files.values()):
            return jsonify({'error': 'No processed files available'}), 400
            
        # Load data
        t1_data, _ = nib.load(processed_files['t1']).get_fdata(), None
        mask_l_data, _ = nib.load(processed_files['mask_l']).get_fdata(), None
        mask_r_data, _ = nib.load(processed_files['mask_r']).get_fdata(), None
        
        # Create slice image
        buf = create_slice_image(t1_data, mask_l_data, mask_r_data, plane, slice_num)
        
        return jsonify({
            'image': f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}",
            'slice': slice_num
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    file.save(input_path)
    
    try:
        # Process the image
        result_files = hippodeep_segmentation(input_path)
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'message': 'Segmentation successful',
            'files': {
                'original': os.path.basename(result_files['t1']),
                'mask_left': os.path.basename(result_files['mask_l']),
                'mask_right': os.path.basename(result_files['mask_r'])
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)