"""
Napari widget for FLM-TEM correlation.
Simple plugin to test computeTransform and transformPoints functions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QGroupBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QMessageBox
)
from qtpy.QtCore import Qt

from .transform_utils import (
    compute_transform,
    transform_points,
    load_registration_points,
    save_transformed_points,
    compare_with_expected
)


class FLMTEMPointTransformWidget(QWidget):  # Changed from FLMTEMCorrelationWidget
    """
    Widget for transforming CSV point data from FLM to TEM coordinates.
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Store loaded data
        self.tem_reg_points = None
        self.flm_reg_points = None
        self.flm_image_points = None
        self.transform_matrix = None
        self.transformed_points = None
        self.output_path = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>FLM-TEM Point Transformation</h2>")
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
        
        # === STEP 3: Apply Transform ===
        apply_group = QGroupBox("Step 3: Transform FLM Image Points")
        apply_layout = QVBoxLayout()
        
        # Load FLM image points
        img_pts_layout = QHBoxLayout()
        self.img_pts_label = QLabel("FLM Image Points: Not loaded")
        self.img_pts_btn = QPushButton("Load FLM Image Points CSV")
        self.img_pts_btn.clicked.connect(self._load_flm_image_points)
        img_pts_layout.addWidget(self.img_pts_label, stretch=1)
        img_pts_layout.addWidget(self.img_pts_btn)
        apply_layout.addLayout(img_pts_layout)
        
        # Apply transformation
        self.apply_btn = QPushButton("Apply Transformation")
        self.apply_btn.clicked.connect(self._apply_transformation)
        self.apply_btn.setEnabled(False)
        apply_layout.addWidget(self.apply_btn)
        
        # Save output
        self.save_btn = QPushButton("Save Transformed Points")
        self.save_btn.clicked.connect(self._save_output)
        self.save_btn.setEnabled(False)
        apply_layout.addWidget(self.save_btn)
        
        apply_group.setLayout(apply_layout)
        main_layout.addWidget(apply_group)
        
        # === Results Display ===
        results_group = QGroupBox("Output Preview")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(200)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Add stretch
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def _load_tem_registration(self):
        """Load TEM registration points (Q matrix)."""
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
        """Load FLM registration points (P matrix)."""
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
            
            # Enable apply button if image points are loaded
            if self.flm_image_points is not None:
                self.apply_btn.setEnabled(True)
                
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
    
    def _load_flm_image_points(self):
        """Load FLM image points to be transformed."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load FLM Image Points (to transform)",
            "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.flm_image_points = load_registration_points(file_path)
                n_points = self.flm_image_points.shape[1]
                self.img_pts_label.setText(f"✓ {n_points} points loaded")
                
                # Enable apply button if transform is computed
                if self.transform_matrix is not None:
                    self.apply_btn.setEnabled(True)
                    
                print(f"Loaded FLM image points: {self.flm_image_points.shape}")
                
            except Exception as e:
                self.img_pts_label.setText(f"✗ Error loading points")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load FLM image points:\n{str(e)}"
                )
    
    def _apply_transformation(self):
        """Apply transformation to FLM image points."""
        try:
            self.transformed_points = transform_points(
                self.transform_matrix,
                self.flm_image_points
            )
            
            # Display in table
            self._display_results()
            
            # Enable save and compare buttons
            self.save_btn.setEnabled(True)
            
            n_points = self.transformed_points.shape[1]
            QMessageBox.information(
                self, "Success",
                f"Transformed {n_points} points successfully!"
            )
            
            print(f"Transformed points shape: {self.transformed_points.shape}")
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to apply transformation:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _display_results(self):
        """Display transformed points in table."""
        if self.transformed_points is None:
            return
    
        x_coords = self.transformed_points[0, :]
        y_coords = self.transformed_points[1, :]
        z_coords = self.transformed_points[2, :]  # Add z column
        n_points = len(x_coords)
    
        # Show only X, Y, Z columns (no Point # column)
        self.results_table.setRowCount(min(n_points, 50))
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['X', 'Y', 'Z'])
    
        for i in range(min(n_points, 50)):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{x_coords[i]:.6f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{y_coords[i]:.6f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{z_coords[i]:.6f}"))
    
        if n_points > 50:
            self.results_table.setRowCount(51)
            self.results_table.setItem(50, 0, QTableWidgetItem("..."))
            self.results_table.setItem(50, 1, QTableWidgetItem(f"({n_points} total)"))
            self.results_table.setItem(50, 2, QTableWidgetItem("..."))
    
    def _save_output(self):
        """Save transformed points to CSV."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transformed Points",
            "TransformedPoints.csv",
            "CSV Files (*.csv)"
        )
        if file_path:
            try:
                save_transformed_points(self.transformed_points, file_path)
                self.output_path = file_path
                self.compare_btn.setEnabled(True)
                
                QMessageBox.information(
                    self, "Success",
                    f"Transformed points saved to:\n{file_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save output:\n{str(e)}"
                )
    
