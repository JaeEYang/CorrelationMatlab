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
        self.tem_image = None
        self.flm_image = None
        self.tem_reg_points = None
        self.flm_reg_points = None
        self.tem_points_layer = None
        self.flm_points_layer = None
        self.warped_image = None
        self.warped_layer = None  # NEW - reference to warped image layer
        self.transform_matrix = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>FLM-TEM Image Warping</h2>")
        main_layout.addWidget(title)
        
        # === STEP 1: Load Images ===
        image_group = QGroupBox("Step 1: Load Images")
        image_layout = QVBoxLayout()
        
        # Load TEM image
        tem_img_layout = QHBoxLayout()
        self.tem_img_label = QLabel("TEM Image: Not loaded")
        self.tem_img_btn = QPushButton("Load TEM Image (TIF)")
        self.tem_img_btn.clicked.connect(self._load_tem_image)
        tem_img_layout.addWidget(self.tem_img_label, stretch=1)
        tem_img_layout.addWidget(self.tem_img_btn)
        image_layout.addLayout(tem_img_layout)
        
        # Load FLM image
        flm_img_layout = QHBoxLayout()
        self.flm_img_label = QLabel("FLM Image: Not loaded")
        self.flm_img_btn = QPushButton("Load FLM Image (TIF)")
        self.flm_img_btn.clicked.connect(self._load_flm_image)
        flm_img_layout.addWidget(self.flm_img_label, stretch=1)
        flm_img_layout.addWidget(self.flm_img_btn)
        image_layout.addLayout(flm_img_layout)
        
        image_group.setLayout(image_layout)
        main_layout.addWidget(image_group)
        
        # === STEP 2: Load Registration Points ===
        reg_group = QGroupBox("Step 2: Load Registration Points")
        reg_layout = QVBoxLayout()
        
        # TEM registration points
        tem_layout = QHBoxLayout()
        self.tem_label = QLabel("TEM Registration Points: Not loaded")
        self.tem_btn = QPushButton("Load TEM Points CSV")
        self.tem_btn.clicked.connect(self._load_tem_registration)
        tem_layout.addWidget(self.tem_label, stretch=1)
        tem_layout.addWidget(self.tem_btn)
        reg_layout.addLayout(tem_layout)
        
        # FLM registration points
        flm_layout = QHBoxLayout()
        self.flm_label = QLabel("FLM Registration Points: Not loaded")
        self.flm_btn = QPushButton("Load FLM Points CSV")
        self.flm_btn.clicked.connect(self._load_flm_registration)
        flm_layout.addWidget(self.flm_label, stretch=1)
        flm_layout.addWidget(self.flm_btn)
        reg_layout.addLayout(flm_layout)
        
        reg_group.setLayout(reg_layout)
        main_layout.addWidget(reg_group)
        
        # Add instructions
        instructions = QLabel(
            "<i>Tip: Points are editable! Select a point layer, switch to points mode (press '2'), "
            "then drag points to adjust registration. Add/delete points as needed.</i>"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: gray; padding: 5px;")
        main_layout.addWidget(instructions)
        
        # === STEP 3: Compute Transform ===
        compute_group = QGroupBox("Step 3: Compute Transformation (uses current point positions)")
        compute_layout = QVBoxLayout()
        
        self.compute_btn = QPushButton("Compute Transform Matrix")
        self.compute_btn.clicked.connect(self._compute_transform)
        self.compute_btn.setEnabled(False)
        compute_layout.addWidget(self.compute_btn)
        
        self.matrix_display = QTextEdit()
        self.matrix_display.setReadOnly(True)
        self.matrix_display.setMaximumHeight(80)
        self.matrix_display.setPlaceholderText("Transformation matrix will appear here...")
        compute_layout.addWidget(self.matrix_display)
        
        # Manual matrix input
        manual_label = QLabel("<b>OR</b> manually input transformation matrix:")
        compute_layout.addWidget(manual_label)
        
        self.manual_matrix_input = QTextEdit()
        self.manual_matrix_input.setMaximumHeight(80)
        self.manual_matrix_input.setPlaceholderText(
            "Paste 3x3 matrix (9 numbers, comma or space separated):\n"
            "Example: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
        )
        compute_layout.addWidget(self.manual_matrix_input)
        
        self.load_manual_btn = QPushButton("Load Manual Matrix")
        self.load_manual_btn.clicked.connect(self._load_manual_matrix)
        compute_layout.addWidget(self.load_manual_btn)
        
        compute_group.setLayout(compute_layout)
        main_layout.addWidget(compute_group)
        
        # === STEP 4: Warp and Overlay ===
        warp_group = QGroupBox("Step 4: Warp FLM Image and Check Overlay")
        warp_layout = QVBoxLayout()
        
        # Warp button
        self.warp_btn = QPushButton("Warp FLM Image")
        self.warp_btn.clicked.connect(self._warp_image)
        self.warp_btn.setEnabled(False)
        warp_layout.addWidget(self.warp_btn)
        
        # Image size info
        self.size_info_label = QLabel("")
        warp_layout.addWidget(self.size_info_label)
        
        # Save warped image
        self.save_img_btn = QPushButton("Save Warped Image (TIF)")
        self.save_img_btn.clicked.connect(self._save_warped_image)
        self.save_img_btn.setEnabled(False)
        warp_layout.addWidget(self.save_img_btn)
        
        # Clear warped image button (for re-warping)
        self.clear_warp_btn = QPushButton("Clear Warped Image (to re-warp)")
        self.clear_warp_btn.clicked.connect(self._clear_warped_image)
        self.clear_warp_btn.setEnabled(False)
        warp_layout.addWidget(self.clear_warp_btn)
        
        warp_group.setLayout(warp_layout)
        main_layout.addWidget(warp_group)
        
        # Add stretch
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def _load_tem_registration(self):
        """Load TEM registration points and add as editable point layer."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load TEM Registration Points", 
            "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.tem_reg_points = load_registration_points(file_path)
                n_points = self.tem_reg_points.shape[1]
                
                # Convert to Nx2 format for napari (drop homogeneous coordinate)
                points_2d = self.tem_reg_points[:2, :].T  # 3xN -> Nx2
                
                # Add as editable point layer (minimal parameters for compatibility)
                self.tem_points_layer = self.viewer.add_points(
                    points_2d,
                    name='TEM_Registration_Points',
                    size=15,
                    face_color='magenta'
                )
                
                self.tem_label.setText(f"✓ TEM: {n_points} points loaded (editable)")
                self._check_ready_to_compute()
                print(f"Loaded TEM registration points: {self.tem_reg_points.shape}")
                
            except Exception as e:
                self.tem_label.setText(f"✗ Error loading TEM points")
                QMessageBox.critical(self, "Error", f"Failed to load TEM points:\n{str(e)}")
    
    def _load_flm_registration(self):
        """Load FLM registration points and add as editable point layer."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load FLM Registration Points",
            "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.flm_reg_points = load_registration_points(file_path)
                n_points = self.flm_reg_points.shape[1]
                
                # Convert to Nx2 format for napari
                points_2d = self.flm_reg_points[:2, :].T
                
                # Add as editable point layer
                self.flm_points_layer = self.viewer.add_points(
                    points_2d,
                    name='FLM_Registration_Points',
                    size=15,
                    face_color='cyan'
                )
                
                self.flm_label.setText(f"✓ FLM: {n_points} points loaded (editable)")
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
        """Compute transformation matrix using current point positions from napari layers."""
        try:
            # Get current points from napari layers (user may have edited them)
            if self.flm_points_layer is not None:
                flm_points_2d = self.flm_points_layer.data  # Nx2
                # Convert to 3xN homogeneous
                N = flm_points_2d.shape[0]
                self.flm_reg_points = np.vstack([
                    flm_points_2d[:, 0],
                    flm_points_2d[:, 1],
                    np.ones(N)
                ])
            
            if self.tem_points_layer is not None:
                tem_points_2d = self.tem_points_layer.data  # Nx2
                N = tem_points_2d.shape[0]
                self.tem_reg_points = np.vstack([
                    tem_points_2d[:, 0],
                    tem_points_2d[:, 1],
                    np.ones(N)
                ])
            
            # Check point counts match
            if self.tem_reg_points.shape[1] != self.flm_reg_points.shape[1]:
                QMessageBox.warning(
                    self, "Point Count Mismatch",
                    f"TEM and FLM must have same number of points!\n"
                    f"TEM: {self.tem_reg_points.shape[1]}, FLM: {self.flm_reg_points.shape[1]}\n\n"
                    f"Add or remove points to match."
                )
                return
            
            # Compute transformation
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
            
            # Enable warp button if both images are loaded
            if self.flm_image is not None and self.tem_image is not None:
                self.warp_btn.setEnabled(True)
            elif self.flm_image is None:
                QMessageBox.information(
                    self, "FLM Image Needed",
                    "Load FLM image before warping."
                )
            elif self.tem_image is None:
                QMessageBox.information(
                    self, "TEM Image Needed",
                    "Load TEM image to use as reference for warping."
                )
                
            QMessageBox.information(
                self, "Success", 
                f"Transformation computed using {self.flm_reg_points.shape[1]} point pairs!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to compute transformation:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            
    def _load_manual_matrix(self):
        """Load transformation matrix from manual text input."""
        try:
            text = self.manual_matrix_input.toPlainText().strip()
            
            # Try to parse as numpy array format or comma-separated
            # Remove brackets and newlines, split by comma or whitespace
            text = text.replace('[', '').replace(']', '').replace('\n', ' ')
            numbers = [float(x) for x in text.replace(',', ' ').split() if x]
            
            if len(numbers) != 9:
                raise ValueError(f"Expected 9 numbers, got {len(numbers)}")
            
            # Reshape to 3x3
            self.transform_matrix = np.array(numbers).reshape(3, 3)
            
            # Display the loaded matrix
            matrix_str = "Manually Loaded Matrix:\n"
            matrix_str += np.array2string(
                self.transform_matrix,
                precision=6,
                suppress_small=True,
                separator=', '
            )
            self.matrix_display.setText(matrix_str)
            
            print("Loaded manual transformation matrix:")
            print(self.transform_matrix)
            
            # Enable warp button if both images are loaded
            if self.flm_image is not None and self.tem_image is not None:
                self.warp_btn.setEnabled(True)
            
            QMessageBox.information(
                self, "Success",
                "Manual transformation matrix loaded successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to parse matrix:\n{str(e)}\n\n"
                "Expected format: 9 numbers separated by commas or spaces\n"
                "Example: 1, 0, 0, 0, 1, 0, 0, 0, 1"
            )
            import traceback
            traceback.print_exc()

    def _load_tem_image(self):
        """Load TEM reference image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load TEM Image",
            "", "TIF Files (*.tif *.tiff);;All Files (*.*)"
        )
        if file_path:
            try:
                from skimage import io
                self.tem_image = io.imread(file_path)
                
                # Display in napari viewer
                if self.tem_image.ndim == 2:
                    self.viewer.add_image(
                        self.tem_image,
                        name='TEM_Image',
                        colormap='gray'
                    )
                elif self.tem_image.ndim == 3:
                    self.viewer.add_image(
                        self.tem_image,
                        name='TEM_Image',
                        rgb=True
                    )
                
                shape = self.tem_image.shape
                self.tem_img_label.setText(f"✓ TEM Image: {shape[0]}x{shape[1]}")
                print(f"Loaded TEM image: {shape}")
                
            except Exception as e:
                self.tem_img_label.setText("✗ Error loading TEM image")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load TEM image:\n{str(e)}"
                )
    
    def _load_flm_image(self):
        """Load FLM image."""
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
                        colormap='green'
                    )
                elif self.flm_image.ndim == 3:
                    self.viewer.add_image(
                        self.flm_image,
                        name='FLM_Image',
                        rgb=True
                    )
                
                shape = self.flm_image.shape
                self.flm_img_label.setText(f"✓ FLM Image: {shape[0]}x{shape[1]}")
                print(f"Loaded FLM image: {shape}")
                
            except Exception as e:
                self.flm_img_label.setText("✗ Error loading FLM image")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load FLM image:\n{str(e)}"
                )
    
    def _warp_image(self):
        """Apply transformation to warp the FLM image."""
        try:
            # Check if TEM image is loaded to use as reference
            if self.tem_image is None:
                QMessageBox.warning(
                    self, "TEM Image Required",
                    "Please load a TEM image first to use as reference for warping."
                )
                return
            
            # Use TEM image size as reference (ensures same coordinate space)
            tem_ref_size = self.tem_image.shape[:2]  # (height, width)
            
            self.warped_image, _ = warp_image(
                self.flm_image,
                self.transform_matrix,
                ref_size=tem_ref_size  # Use TEM dimensions as reference
            )
            
            # Remove old warped layer if it exists
            if self.warped_layer is not None:
                try:
                    self.viewer.layers.remove(self.warped_layer)
                except ValueError:
                    pass  # Layer already removed
            
            # Display warped image - no offset needed, same coordinate space as TEM
            if self.warped_image.ndim == 2:
                self.warped_layer = self.viewer.add_image(
                    self.warped_image,
                    name='Warped_FLM',
                    colormap='green',
                    blending='additive',
                    opacity=0.5
                )
            elif self.warped_image.ndim == 3:
                self.warped_layer = self.viewer.add_image(
                    self.warped_image,
                    name='Warped_FLM',
                    blending='additive',
                    opacity=0.5
                )
            
            # Enable buttons
            self.save_img_btn.setEnabled(True)
            self.clear_warp_btn.setEnabled(True)
            
            # Update info label
            shape = self.warped_image.shape
            original_shape = self.flm_image.shape
            self.size_info_label.setText(
                f"Original FLM: {original_shape[0]}x{original_shape[1]} → "
                f"Warped to TEM size: {shape[0]}x{shape[1]}\n"
                f"Images now in same coordinate space. Check overlay alignment!"
            )
            
            print(f"Warped FLM to TEM reference size: {self.warped_image.shape}")
            print("Tip: Toggle warped layer visibility or adjust opacity to check alignment")
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to warp image:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _clear_warped_image(self):
        """Remove warped image layer to prepare for re-warping."""
        if self.warped_layer is not None:
            try:
                self.viewer.layers.remove(self.warped_layer)
                self.warped_layer = None
                self.size_info_label.setText("Warped image cleared. Adjust points and warp again.")
                print("Cleared warped image. Ready to re-warp.")
            except ValueError:
                pass  # Already removed
    
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
