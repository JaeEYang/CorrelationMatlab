"""
 Controller
    =
application behavior
"""
import numpy as np

from correlation2d3d.session import CorrelationSession
from correlation2d3d.core.transform import apply_affine_matrix
from correlation2d3d.core.orientation import (
    flip_horizontal,
    flip_vertical,
    horizontal_flip_matrix,
    orient_image_from_baseline,
    prepare_rotation_canvas,
    vertical_flip_matrix,
)

from correlation2d3d.core.geometry import Points2D

class OfflineCorrelationController:

    def __init__(
        self,
        viewer,
        session: CorrelationSession,
    ):
        self.viewer = viewer
        self.session = session

    # this becomes out generic loader for modalities make it bit tidy to keep track of session state.
    # retuns the correspoding state object
    def _get_modality(self, role: str):
        if role == "FLM":
            return self.session.flm

        if role == "TEM":
            return self.session.tem

        raise ValueError(
            f"unknown modality role: {role}"
        )

    # helper if layer exists we can remove it if not do nothing.
    def _remove_layer_if_present(
        self,
        layer_name: str,
    ) -> None:
        try:
            layer = self.viewer.layers[layer_name]
        except KeyError:
            return

        self.viewer.layers.remove(layer)
        
    #lets say we did the registration and then changed the image orientation we should able to invalidate the old registraion matrix
    # and also the warping becomes invalidated need to recalulate both 
    def invalidate_registration(self) -> None:
        self.session.registration = None
        self.session.warped_flm = None

        # this removed those two layers aswell
        self._remove_layer_if_present(
            "FLM Landmarks Registered to TEM"
        )

        self._remove_layer_if_present(
            "Warped FLM"
        )

        self._remove_layer_if_present(
            "Registered FLM"
        )
        
    def set_modality_orientation(
        self,
        role: str,
        image: np.ndarray,
        orientation_matrix: np.ndarray,
    ) -> None:
        modality = self._get_modality(role)

        modality.image = image
        modality.orientation_matrix = orientation_matrix

        #always rebuild current points from the
        #orriginal landmarks.
        if modality.original_points is not None:
            modality.points = apply_affine_matrix(
                modality.orientation_matrix,
                modality.original_points,
            )
        else:
            modality.points = None

        # Update the image displayed by napari.
        self.viewer.layers[role].data = modality.image

        # Update landmark display if landmarks exist.
        if modality.points is not None:
            layer_name = f"{role} Landmarks"

            try:
                layer = self.viewer.layers[layer_name]
            except KeyError:
                pass
            else:
                layer.data = modality.points.to_rc()
                
                
    # This helper has two main calls 
    # 1) orient_image_from_baseline (returns the oriented image and the associated matrix)
    # 2) set_modality_orientation (this updates the session info, landmarks, layer data and invalidates the old registration )
    def rebuild_modality_from_baseline(
        self,
        role: str,
    ) -> None:
        """Rebuild pixels and the original-to-working matrix from fixed settings."""

        modality = self._get_modality(role)

        if modality.rotation_base_image is None:
            return

        print(f"the current rotation angle is {modality.rotation_angle}")

        #get back the oriented image (for some θ )
        # return the rotated image and the matrix mapping baseline coordinated to the newly oriented coordinates.
        oriented_image, operation = orient_image_from_baseline(
            modality.rotation_base_image, # fixed padded array
            modality.rotation_angle,
            horizontal_flipped=modality.horizontal_flipped,
            vertical_flipped=modality.vertical_flipped,
        )

        # O = V @ H @ R @ P. 
        # install the calculated results, update landmarks, and update layers.
        self.set_modality_orientation(
            role,
            oriented_image,
            operation @ modality.rotation_base_orientation_matrix,
        )
        
    #  horizontal flip 
    # take input role "FLM" or "TEM"
    def flip_modality_horizontal(
        self,
        role: str,
    ) -> None:
        modality = self._get_modality(role)

        if modality.image is None:
            return

        # Reorder current pixels exactly; do not repeat rotation or change the baseline.
        flipped_image, _ = flip_horizontal(modality.image)

        orientation_matrix = (
            horizontal_flip_matrix(modality.image.shape[1])
            @ modality.orientation_matrix
        )

        modality.horizontal_flipped = not modality.horizontal_flipped # changes false to true (flipped)

        self.set_modality_orientation(
            role,
            flipped_image,
            orientation_matrix,
        )
        
    # veritcal flip 
    def flip_modality_vertical(
        self,
        role: str,
    ) -> None:
        modality = self._get_modality(role)

        if modality.image is None:
            return

        # The image and original-to-working matrix receive the same display-axis flip.
        flipped_image, _ = flip_vertical(modality.image)

        orientation_matrix = (
            vertical_flip_matrix(modality.image.shape[0])
            @ modality.orientation_matrix
        )

        modality.vertical_flipped = not modality.vertical_flipped

        self.set_modality_orientation(
            role,
            flipped_image,
            orientation_matrix,
        )
        
    # this just resets eveything to beginning!
    def reset_modality_orientation(
        self,
        role: str,
    ) -> None:
        modality = self._get_modality(role)

        if modality.rotation_base_image is None:
            return

        modality.rotation_angle = 0.0
        modality.horizontal_flipped = False
        modality.vertical_flipped = False

        self.rebuild_modality_from_baseline(
            role
        )
        
    # do the rotation 
    def set_modality_rotation(
        self,
        role: str,
        angle_degrees: float,
    ) -> None:
        modality = self._get_modality(role) # get the session.flm or .tem 

        if modality.rotation_base_image is None:
            return

        modality.rotation_angle = angle_degrees

        self.rebuild_modality_from_baseline(
            role
        )
        
    #Take this newly loaded image and make it the new FLM or TEM source image.
    def set_modality_image(
        self,
        role: str,
        image: np.ndarray,
    ) -> None:

        # orientation preparation begins here. creates space for later rotations
        # returns two things, the padded image and matrix describing where the orginal image was placed.
        rotation_canvas, padding_matrix = (
            prepare_rotation_canvas(image)
        )

        #update the session object with the loaded image based on the role (FLM or TEM)
        # Our session says: this exact NumPy array is the FLM image for this correlation job
        modality = self._get_modality(role)

        # Saves the copy of original image (the unpaded version)
        modality.original_image = np.array(
            image,
            copy=True, # independent pixel storage
        )

        # saves the copy of the image but places in the larged canvas to allow rotation
        # this is the working image.
        modality.image = np.array(
            rotation_canvas,
            copy=True,
        )

        # Reloading the image clears orientation adjustments, but keeps the centering translation.
        # save the current coordinate mapping which is padding.
        modality.orientation_matrix = np.array(
            padding_matrix,
            dtype=np.float64,
            copy=True,
        )

        # fixed starting image
        modality.rotation_base_image = np.array(
            rotation_canvas,
            copy=True,
        )

        #records how do I get from original image coordinates to this fixed baseline?
        modality.rotation_base_orientation_matrix = np.array(
            padding_matrix,
            dtype=np.float64,
            copy=True,
        )

        # currently want normal image
        modality.rotation_angle = 0.0
        modality.horizontal_flipped = False
        modality.vertical_flipped = False

        #landmarks belonged to the previous image.
        #the user must load/confirm landmarks for this image.
        modality.original_points = None
        modality.points = None

        self._remove_layer_if_present(
            f"{role} Landmarks"
        )

        # Create or update the napari image layer.
        # Recreate the layer so napari detects grayscale/RGB correctly.
        self._remove_layer_if_present(role)

        # napari now has a layer nameed "role" showing padded image
        self.viewer.add_image(
            modality.image,
            name=role,
        )
        
    def set_original_points(
        self,
        role: str,
        points: Points2D,
    ) -> None:

        #update the session object
        modality = self._get_modality(role)

        modality.original_points = points # these belong to orginal unpadded image.

        # this is cool now if we upload the points after we have already flipped or rotated the image.
        # this will apply the correct tranformation to them aswell.
        modality.points = apply_affine_matrix(
            modality.orientation_matrix,
            modality.original_points,
        )

        layer_name = f"{role} Landmarks"

        # convert to napari points convention y,x/ rc these will be recieved by napari frontend
        napari_points = modality.points.to_rc()

        try:
            layer = self.viewer.layers[layer_name]
        except KeyError:
            self.viewer.add_points(
                napari_points,
                name=layer_name,
                size=32,
                face_color="red",
            )
        else:
            layer.data = napari_points
            layer.size = 32
            layer.face_color = "red"