"""
Napari widget for FLM-TEM image warping.
Applies computed transformation to warp FLM images to TEM coordinate space.
"""

import numpy as np
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QGroupBox, QTextEdit, QMessageBox
)

from .transform_utils import (
    compute_transform,
    load_registration_points,
    warp_image
)


class FLMTEMImageWarpWidget(QWidget):
    """
    Widget for warping FLM images to TEM coordinate system.
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Store loaded data
        self.tem_reg_points = None
        self.flm_reg_points = None
        self.flm_image = None
        self.warped_image = None
        self.transform_matrix = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>FLM-TEM Image Warping</h2>")
        main_layout.addWidget(title)
        
        # === STEP 1: Load Registration Points ===
        reg_group = QGroupBox("Step 1: Load Registration Point Correspondences")
        reg_layout = QVBoxLayout()
        
        # TEM registration points
        tem_layout = QHBoxLayout()
        self.tem_label = QLabel("TEM Registration Points: Not loaded")
        self.tem_btn = QPushButton("Load TEM Registration CSV")
        self.tem_btn.clicked.connect(self._load_tem_registration)
        tem_layout.addWidget(self.tem_label, stretch=1)
        tem_layout.addWidget(self.tem_btn)
        reg_layout.addLayout(tem_layout)
        
        # FLM registration points
        flm_layout = QHBoxLayout()
        self.flm_label = QLabel("FLM Registration Points: Not loaded")
        self.flm_btn = QPushButton("Load FLM Registration CSV")
        self.flm_btn.clicked.connect(self._load_flm_registration)
        flm_layout.addWidget(self.flm_label, stretch=1)
        flm_layout.addWidget(self.flm_btn)
        reg_layout.addLayout(flm_layout)
        
        reg_group.setLayout(reg_layout)
        main_layout.addWidget(reg_group)
        
        # === STEP 2: Compute Transform ===
        compute_group = QGroupBox("Step 2: Compute Transformation Matrix")
        compute_layout = QVBoxLayout()
        
        self.compute_btn = QPushButton("Compute Transform Matrix")
        self.compute_btn.clicked.connect(self._compute_transform)
        self.compute_btn.setEnabled(False)
        compute_layout.addWidget(self.compute_btn)
        
        self.matrix_display = QTextEdit()
        self.matrix_display.setReadOnly(True)
        self.matrix_display.setMaximumHeight(100)
        self.matrix_display.setPlaceholderText("Transformation matrix will appear here...")
        compute_layout.addWidget(self.matrix_display)
        
        compute_group.setLayout(compute_layout)
        main_layout.addWidget(compute_group)
        
        # === STEP 3: Load and Warp Image ===
        warp_group = QGroupBox("Step 3: Load and Warp FLM Image")
        warp_layout = QVBoxLayout()
        
        # Load FLM image
        img_layout = QHBoxLayout()
        self.img_label = QLabel("FLM Image: Not loaded")
        self.img_btn = QPushButton("Load FLM Image (TIF)")
        self.img_btn.clicked.connect(self._load_flm_image)
        img_layout.addWidget(self.img_label, stretch=1)
        img_layout.addWidget(self.img_btn)
        warp_layout.addLayout(img_layout)
        
        # Reference size info
        self.ref_size_label = QLabel("Output size will match loaded image")
        warp_layout.addWidget(self.ref_size_label)
        
        # Warp button
        self.warp_btn = QPushButton("Warp Image")
        self.warp_btn.clicked.connect(self._warp_image)
        self.warp_btn.setEnabled(False)
        warp_layout.addWidget(self.warp_btn)
        
        # Save warped image
        self.save_img_btn = QPushButton("Save Warped Image (TIF)")
        self.save_img_btn.clicked.connect(self._save_warped_image)
        self.save_img_btn.setEnabled(False)
        warp_layout.addWidget(self.save_img_btn)
        
        warp_group.setLayout(warp_layout)
        main_layout.addWidget(warp_group)
        
        # Add stretch
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def _load_tem_registration(self):
        """Load TEM registration points."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load TEM Registration Points", 
            "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.tem_reg_points = load_registration_points(file_path)
                n_points = self.tem_reg_points.shape[1]
                self.tem_label.setText(f"✓ TEM: {n_points} points loaded")
                self._check_ready_to_compute()
                print(f"Loaded TEM registration points: {self.tem_reg_points.shape}")
            except Exception as e:
                self.tem_label.setText(f"✗ Error loading TEM points")
                QMessageBox.critical(self, "Error", f"Failed to load TEM points:\n{str(e)}")
    
    def _load_flm_registration(self):
        """Load FLM registration points."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load FLM Registration Points",
            "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.flm_reg_points = load_registration_points(file_path)
                n_points = self.flm_reg_points.shape[1]
                self.flm_label.setText(f"✓ FLM: {n_points} points loaded")
                self._check_ready_to_compute()
                print(f"Loaded FLM registration points: {self.flm_reg_points.shape}")
            except Exception as e:
                self.flm_label.setText(f"✗ Error loading FLM points")
                QMessageBox.critical(self, "Error", f"Failed to load FLM points:\n{str(e)}")
    
    def _check_ready_to_compute(self):
        """Check if we can compute the transformation."""
        if self.tem_reg_points is not None and self.flm_reg_points is not None:
            if self.tem_reg_points.shape[1] == self.flm_reg_points.shape[1]:
                self.compute_btn.setEnabled(True)
            else:
                QMessageBox.warning(
                    self, "Point Mismatch",
                    f"TEM and FLM must have same number of points!\n"
                    f"TEM: {self.tem_reg_points.shape[1]}, FLM: {self.flm_reg_points.shape[1]}"
                )
    
    def _compute_transform(self):
        """Compute transformation matrix from registration points."""
        try:
            self.transform_matrix = compute_transform(
                self.flm_reg_points, 
                self.tem_reg_points
            )
            
            # Display matrix
            matrix_str = "Transformation Matrix (FLM → TEM):\n"
            matrix_str += np.array2string(
                self.transform_matrix, 
                precision=6, 
                suppress_small=True,
                separator=', '
            )
            self.matrix_display.setText(matrix_str)
            
            print("Computed transformation matrix:")
            print(self.transform_matrix)
            
            # Enable warp button if image is loaded
            if self.flm_image is not None:
                self.warp_btn.setEnabled(True)
                
            QMessageBox.information(
                self, "Success", 
                "Transformation matrix computed successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to compute transformation:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _load_flm_image(self):
        """Load FLM TIF image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load FLM Image",
            "", "TIF Files (*.tif *.tiff);;All Files (*.*)"
        )
        if file_path:
            try:
                from skimage import io
                self.flm_image = io.imread(file_path)
                
                # Display in napari viewer
                if self.flm_image.ndim == 2:
                    self.viewer.add_image(
                        self.flm_image,
                        name='FLM_Image',
                        colormap='gray'
                    )
                elif self.flm_image.ndim == 3:
                    self.viewer.add_image(
                        self.flm_image,
                        name='FLM_Image',
                        rgb=True
                    )
                
                shape = self.flm_image.shape
                self.img_label.setText(f"✓ Image loaded: {shape}")
                self.ref_size_label.setText(f"Output size: {shape[0]}x{shape[1]}")
                
                # Enable warp button if transform exists
                if self.transform_matrix is not None:
                    self.warp_btn.setEnabled(True)
                
                print(f"Loaded FLM image: {shape}")
                
            except Exception as e:
                self.img_label.setText("✗ Error loading image")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load image:\n{str(e)}"
                )
                import traceback
                traceback.print_exc()
    
    def _warp_image(self):
        """Apply transformation to warp the FLM image."""
        try:
            # Use the image's own size as reference
            ref_size = self.flm_image.shape[:2]
            
            # Warp the image
            self.warped_image = warp_image(
                self.flm_image,
                self.transform_matrix,
                ref_size
            )
            
            # Display warped image in napari
            if self.warped_image.ndim == 2:
                self.viewer.add_image(
                    self.warped_image,
                    name='Warped_FLM_Image',
                    colormap='green',
                    blending='additive',
                    opacity=0.7
                )
            elif self.warped_image.ndim == 3:
                self.viewer.add_image(
                    self.warped_image,
                    name='Warped_FLM_Image',
                    opacity=0.7
                )
            
            # Enable save button
            self.save_img_btn.setEnabled(True)
            
            QMessageBox.information(
                self, "Success",
                f"Image warped successfully!\nSize: {self.warped_image.shape}"
            )
            
            print(f"Warped image shape: {self.warped_image.shape}")
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to warp image:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _save_warped_image(self):
        """Save warped image to TIF file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Warped Image",
            "Warped_FLM_Image.tif",
            "TIF Files (*.tif *.tiff)"
        )
        if file_path:
            try:
                from skimage import io
                io.imsave(file_path, self.warped_image)
                
                QMessageBox.information(
                    self, "Success",
                    f"Warped image saved to:\n{file_path}"
                )
                print(f"Saved warped image to: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save image:\n{str(e)}"
                )
