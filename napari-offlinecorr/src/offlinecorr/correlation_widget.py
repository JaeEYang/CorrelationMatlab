"""
Napari widget for FLM-TEM correlation and registration.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QComboBox, QGroupBox
)
from magicgui import magic_factory
from napari.layers import Image, Points
import napari

from .transform_utils import (
    compute_transform,
    transform_points,
    warp_image,
    prepare_points_homogeneous,
    extract_2d_points
)


class CorrelationWidget(QWidget):
    """
    Widget for computing and applying FLM-TEM transformations.
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Store loaded data
        self.flm_points = None
        self.tem_points = None
        self.transform_matrix = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h3>FLM-TEM Correlation</h3>")
        layout.addWidget(title)
        
        # Data loading section
        data_group = QGroupBox("1. Load Point Correspondences")
        data_layout = QVBoxLayout()
        
        # FLM points
        flm_layout = QHBoxLayout()
        self.flm_label = QLabel("FLM Points: Not loaded")
        self.flm_btn = QPushButton("Load FLM CSV")
        self.flm_btn.clicked.connect(self._load_flm_points)
        flm_layout.addWidget(self.flm_label)
        flm_layout.addWidget(self.flm_btn)
        data_layout.addLayout(flm_layout)
        
        # TEM points
        tem_layout = QHBoxLayout()
        self.tem_label = QLabel("TEM Points: Not loaded")
        self.tem_btn = QPushButton("Load TEM CSV")
        self.tem_btn.clicked.connect(self._load_tem_points)
        tem_layout.addWidget(self.tem_label)
        tem_layout.addWidget(self.tem_btn)
        data_layout.addLayout(tem_layout)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Transformation section
        transform_group = QGroupBox("2. Compute Transformation")
        transform_layout = QVBoxLayout()
        
        self.compute_btn = QPushButton("Compute Transform")
        self.compute_btn.clicked.connect(self._compute_transform)
        self.compute_btn.setEnabled(False)
        transform_layout.addWidget(self.compute_btn)
        
        self.transform_status = QLabel("Status: Waiting for point data")
        transform_layout.addWidget(self.transform_status)
        
        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)
        
        # Application section
        apply_group = QGroupBox("3. Apply Transformation")
        apply_layout = QVBoxLayout()
        
        # Layer selection for images
        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("FLM Image Layer:"))
        self.flm_image_combo = QComboBox()
        self.flm_image_combo.currentTextChanged.connect(self._update_apply_button)
        image_layout.addWidget(self.flm_image_combo)
        apply_layout.addLayout(image_layout)
        
        # TEM reference size
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("TEM Reference Layer:"))
        self.tem_image_combo = QComboBox()
        self.tem_image_combo.currentTextChanged.connect(self._update_apply_button)
        ref_layout.addWidget(self.tem_image_combo)
        apply_layout.addLayout(ref_layout)
        
        # Apply buttons
        self.apply_image_btn = QPushButton("Warp FLM Image to TEM")
        self.apply_image_btn.clicked.connect(self._apply_to_image)
        self.apply_image_btn.setEnabled(False)
        apply_layout.addWidget(self.apply_image_btn)
        
        self.apply_points_btn = QPushButton("Transform FLM Points to TEM")
        self.apply_points_btn.clicked.connect(self._apply_to_points)
        self.apply_points_btn.setEnabled(False)
        apply_layout.addWidget(self.apply_points_btn)
        
        apply_group.setLayout(apply_layout)
        layout.addWidget(apply_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connect to viewer events
        self.viewer.layers.events.inserted.connect(self._update_layer_combos)
        self.viewer.layers.events.removed.connect(self._update_layer_combos)
        
        # Initial update
        self._update_layer_combos()
    
    def _load_flm_points(self):
        """Load FLM point coordinates from CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load FLM Points", "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                df = pd.read_csv(file_path)
                # Assume CSV has columns 'x' and 'y' or first two columns are x, y
                if 'x' in df.columns and 'y' in df.columns:
                    points = df[['x', 'y']].values
                else:
                    points = df.iloc[:, :2].values
                
                self.flm_points = prepare_points_homogeneous(points)
                self.flm_label.setText(f"FLM Points: {points.shape[0]} points loaded")
                
                # Add to viewer
                self.viewer.add_points(
                    points, 
                    name='FLM_Points',
                    size=10,
                    face_color='cyan',
                    edge_color='white'
                )
                
                self._check_ready_to_compute()
            except Exception as e:
                self.flm_label.setText(f"Error loading FLM points: {str(e)}")
    
    def _load_tem_points(self):
        """Load TEM point coordinates from CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load TEM Points", "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if 'x' in df.columns and 'y' in df.columns:
                    points = df[['x', 'y']].values
                else:
                    points = df.iloc[:, :2].values
                
                self.tem_points = prepare_points_homogeneous(points)
                self.tem_label.setText(f"TEM Points: {points.shape[0]} points loaded")
                
                # Add to viewer
                self.viewer.add_points(
                    points,
                    name='TEM_Points',
                    size=10,
                    face_color='magenta',
                    edge_color='white'
                )
                
                self._check_ready_to_compute()
            except Exception as e:
                self.tem_label.setText(f"Error loading TEM points: {str(e)}")
    
    def _check_ready_to_compute(self):
        """Check if we have enough data to compute transformation."""
        if self.flm_points is not None and self.tem_points is not None:
            if self.flm_points.shape[1] == self.tem_points.shape[1]:
                self.compute_btn.setEnabled(True)
                self.transform_status.setText(
                    f"Ready to compute ({self.flm_points.shape[1]} point pairs)"
                )
            else:
                self.transform_status.setText(
                    "Error: FLM and TEM must have same number of points"
                )
        else:
            self.compute_btn.setEnabled(False)
    
    def _compute_transform(self):
        """Compute the transformation matrix from loaded points."""
        try:
            self.transform_matrix = compute_transform(self.flm_points, self.tem_points)
            self.transform_status.setText("✓ Transform computed successfully")
            
            # Display transform info
            print("Transformation Matrix:")
            print(self.transform_matrix)
            
            # Enable apply buttons
            self._update_apply_button()
            
        except Exception as e:
            self.transform_status.setText(f"Error computing transform: {str(e)}")
    
    def _update_layer_combos(self):
        """Update the layer combo boxes with available image layers."""
        # Store current selections
        current_flm = self.flm_image_combo.currentText()
        current_tem = self.tem_image_combo.currentText()
        
        # Clear and repopulate
        self.flm_image_combo.clear()
        self.tem_image_combo.clear()
        
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.flm_image_combo.addItem(layer.name)
                self.tem_image_combo.addItem(layer.name)
        
        # Restore selections if possible
        idx = self.flm_image_combo.findText(current_flm)
        if idx >= 0:
            self.flm_image_combo.setCurrentIndex(idx)
        
        idx = self.tem_image_combo.findText(current_tem)
        if idx >= 0:
            self.tem_image_combo.setCurrentIndex(idx)
    
    def _update_apply_button(self):
        """Enable/disable apply buttons based on state."""
        has_transform = self.transform_matrix is not None
        has_flm_image = self.flm_image_combo.currentText() != ""
        has_tem_image = self.tem_image_combo.currentText() != ""
        
        self.apply_image_btn.setEnabled(
            has_transform and has_flm_image and has_tem_image
        )
        self.apply_points_btn.setEnabled(has_transform and has_flm_image)
    
    def _apply_to_image(self):
        """Apply transformation to warp FLM image to TEM coordinates."""
        try:
            # Get the selected layers
            flm_layer_name = self.flm_image_combo.currentText()
            tem_layer_name = self.tem_image_combo.currentText()
            
            flm_layer = self.viewer.layers[flm_layer_name]
            tem_layer = self.viewer.layers[tem_layer_name]
            
            # Get images
            flm_img = flm_layer.data
            tem_shape = tem_layer.data.shape
            
            # Warp image
            warped = warp_image(flm_img, self.transform_matrix, tem_shape)
            
            # Add to viewer
            self.viewer.add_image(
                warped,
                name=f'{flm_layer_name}_warped',
                colormap='green',
                blending='additive',
                opacity=0.7
            )
            
            print(f"Warped image added: {flm_layer_name}_warped")
            
        except Exception as e:
            print(f"Error warping image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _apply_to_points(self):
        """Transform FLM points to TEM coordinate system."""
        try:
            # Transform the original FLM points
            transformed = transform_points(self.transform_matrix, self.flm_points)
            
            # Convert back to Nx2 format for napari
            points_2d = extract_2d_points(transformed)
            
            # Add to viewer
            self.viewer.add_points(
                points_2d,
                name='FLM_Points_Transformed',
                size=10,
                face_color='yellow',
                edge_color='white'
            )
            
            print(f"Transformed {points_2d.shape[0]} points to TEM coordinates")
            
        except Exception as e:
            print(f"Error transforming points: {str(e)}")
            import traceback
            traceback.print_exc()


# Magic factory version for simpler integration
@magic_factory(call_button="Compute Transform")
def compute_correlation_transform(
    flm_points: Points,
    tem_points: Points
) -> napari.types.LayerDataTuple:
    """
    Compute transformation from FLM to TEM coordinates.
    
    Parameters
    ----------
    flm_points : Points layer
        Points in FLM coordinate system
    tem_points : Points layer
        Corresponding points in TEM coordinate system
    
    Returns
    -------
    Transformed points layer
    """
    # Get point data
    P = prepare_points_homogeneous(flm_points.data)
    Q = prepare_points_homogeneous(tem_points.data)
    
    # Compute transform
    M = compute_transform(P, Q)
    
    # Transform FLM points
    transformed = transform_points(M, P)
    points_2d = extract_2d_points(transformed)
    
    return (points_2d, {'name': 'Transformed_Points', 'face_color': 'yellow'}, 'points')
