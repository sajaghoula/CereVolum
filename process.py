from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import nibabel as nib
import numpy as np
import tempfile
import os
import shutil
import subprocess
import zipfile
import io
import matplotlib.pyplot as plt
import subprocess
from flask import after_this_request

app = Flask(__name__)
CORS(app)

def create_preview_image(nii_path, output_path):
    """Create a preview PNG from a NIfTI file"""
    img = nib.load(nii_path)
    data = img.get_fdata()
    mid_slice = data.shape[2] // 2
    slice_2d = data[:, :, mid_slice]

    # Normalize slice data between 0 and 1
    slice_min, slice_max = slice_2d.min(), slice_2d.max()
    if slice_max > slice_min:
        slice_norm = (slice_2d - slice_min) / (slice_max - slice_min)
    else:
        slice_norm = slice_2d  # avoid div by zero if constant image

    plt.imsave(output_path, slice_norm.T, cmap='gray')
    plt.close()


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
    
    # Run Hippodeep segmentation
    # segmentation_command = f"cd {subject_folder} && python {os.path.join(workpath, 'hippodeep.py')} {subject_filename}"
    # subprocess.run(segmentation_command, shell=True, check=True)

    

    segmentation_command = f"cd {subject_folder} && python {os.path.join(workpath, 'hippodeep.py')} {subject_filename}"

    result = subprocess.run(segmentation_command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        # Print/log stdout and stderr for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, segmentation_command, output=result.stdout, stderr=result.stderr)


    
    # Create segmentation folder
    subject_seg_dir = os.path.join(temp_segmentations_dir, subject_base)
    os.makedirs(subject_seg_dir, exist_ok=True)
    
    # Define expected output files
    left_mask_file = f"{subject_base}_mask_L.nii.gz"
    right_mask_file = f"{subject_base}_mask_R.nii.gz"
    volume_file = f"{subject_base}_hippoLR_volumes.csv"
    
    # Move segmentation files
    for mask_file in [left_mask_file, right_mask_file, volume_file]:
        src = os.path.join(subject_folder, mask_file)
        if os.path.exists(src):
            dest = os.path.join(subject_seg_dir, mask_file)
            shutil.move(src, dest)
    
    # Create preview images
    preview_files = []
    for mask_type, mask_file in [('original', subject_filename), 
                                ('left', left_mask_file), 
                                ('right', right_mask_file)]:
        if os.path.exists(os.path.join(subject_seg_dir, mask_file)):
            preview_path = os.path.join(subject_seg_dir, f"preview_{mask_type}.png")
            create_preview_image(os.path.join(subject_seg_dir, mask_file), preview_path)
            preview_files.append(preview_path)
    
    return (
        os.path.join(subject_seg_dir, left_mask_file),
        os.path.join(subject_seg_dir, right_mask_file),
        os.path.join(subject_seg_dir, volume_file),
        preview_files
    )

@app.route('/preview/<filename>')
def serve_preview(filename):
    preview_dir = os.path.join(os.getcwd(), "temp_segmentations")
    # Walk through subdirectories to find the file
    for root, dirs, files in os.walk(preview_dir):
        if filename in files:
            return send_file(os.path.join(root, filename), mimetype='image/png')
    return "Preview not found", 404



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
        left_mask_path, right_mask_path, volume_path, preview_paths = hippodeep_segmentation(input_path)
        
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add segmentation files
            for file_path in [left_mask_path, right_mask_path, volume_path]:
                if os.path.exists(file_path):
                    zip_file.write(file_path, os.path.basename(file_path))
            
            # Add preview images
            for preview_path in preview_paths:
                zip_file.write(preview_path, os.path.basename(preview_path))
        
        zip_buffer.seek(0)
        
        # Prepare filename
        original_filename = os.path.basename(input_path).replace('.nii.gz', '')
        zip_filename = f"{original_filename}_hippocampus_segmentations.zip"
        
        # Create cleanup callback
        @after_this_request
        def cleanup(response):
            # try:
            #     if os.path.exists(temp_dir):
            #         shutil.rmtree(temp_dir)
            #     shutil.rmtree(os.path.join(os.getcwd(), "temp_subjects"), ignore_errors=True)
            #     shutil.rmtree(os.path.join(os.getcwd(), "temp_segmentations"), ignore_errors=True)
            # except Exception as e:
            #     print(f"⚠️ Cleanup failed: {e}")
            return response
        

        # return send_file(
        #     zip_buffer,
        #     mimetype='application/zip',
        #     as_attachment=True,
        #     download_name=f"{os.path.basename(input_path).replace('.nii.gz', '')}_results.zip"
        # )
    
        return jsonify({
            "message": "Segmentation successful",
            "preview": os.path.basename(preview_paths[0])  # like 'preview_left.png'
        })

    except subprocess.CalledProcessError as e:
        return jsonify({'error': f"Segmentation failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)