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
        
        # this eventually means which points layer is currently being used for each modality cause there would be multiple csv, user
        self._landmark_layers = {
            "FLM": None,
            "TEM": None,
        }


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
    
    # When orientation changes, transform the session coordinates and move the currently assigned Points layer.
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
       # Update landmark display if landmarks exist.
        if modality.points is not None:
            layer = self._landmark_layers[role] #which Points layer is currently assigned to this image?
            
            # maybe user will delete the layer later protects us from trying to update something that isn't displayed anymore.
            if (
                layer is not None
                and layer in self.viewer.layers
            ):
                layer.data = modality.points.to_rc()
                
                
    # This helper has two main calls 
    # 1) orient_image_from_baseline (returns the oriented image and the associated matrix)
    # 2) set_modality_orientation (this updates the session info, landmarks, layer data and invalidates the old registration (the wrapper does in the offline_correlation cause refactored!) )
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

        self._landmark_layers[role] = None

        # Create or update the napari image layer.
        # Recreate the layer so napari detects grayscale/RGB correctly.
        self._remove_layer_if_present(role)

        # napari now has a layer nameed "role" showing padded image
        self.viewer.add_image(
            modality.image,
            name=role,
        )
        
    
        """
    session modality.points
            |
            | .to_rc()
            v
    napari coordinates
            |
            v
    FLM Landmarks / TEM Landmarks layer
        """
    def _update_landmark_layer(
        self,
        role:str,
    ) -> None:
        
        #session object (session.flm or .tem)
        modality = self._get_modality(role)
        
        if modality.points is None:
            return
        
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
        

        """
        CSV
        Points2D
        transform points to current image orientation
        napari Points layer
        
        Convert an original-coordinate CSV into a normal napari Points candidate.
        """
    def create_points_layer_from_original_points(
        self,
        role: str,
        points: Points2D,
        name: str,
    ):
        modality = self._get_modality(role)

        if modality.image is None:
            raise ValueError(
                f"{role} image must be loaded before importing landmarks"
            )

        working_points = apply_affine_matrix(
            modality.orientation_matrix,
            points,
        )

        layer = self.viewer.add_points(
            working_points.to_rc(),
            name=name,
            size=32,
            face_color="red",
        )

        return layer
        
        
    
    # When the user selected a points layer 
    #  the ui will call controller.use_points_layer ("FLM", selected_layer)
    # Take an existing napari Points layer and assign it as FLM/TEM landmarks.
    def use_points_layer(
        self,
        role: str,
        layer,
    ) -> None:

        # get the correct modality (session.flm or .tem)
        modality = self._get_modality(role)

        #make sure the image exits
        if modality.image is None:
            raise ValueError(
                f"{role} image must be loaded before assigning landmarks"
            )

        # read the points layer actual data (this is gonna be y,x would have to convert)
        layer_data = np.asarray(
            layer.data
        )

        # make sure the data is 2D N points × 2 coordinates 
        if (
            layer_data.ndim != 2
            or layer_data.shape[1] != 2
        ):
            raise ValueError(
                "landmark Points layer must contain 2D coordinates"
            )

        # convert into our internal xy convention
        working_points = Points2D.from_rc(
            layer_data
        )

        # these are already working coordinates (image could have been rotated flipped and then points were selected padded -> rotated 30°-> horizontally flipped)
        modality.points = working_points

        # to get the og we need to get the inverse of the orientation matrix
        inverse_orientation = np.linalg.inv(
            modality.orientation_matrix
        )

        # apply the inverse and after that is can just follow the same path as csv
        modality.original_points = apply_affine_matrix(
            inverse_orientation,
            modality.points,
        )

        # remeber which napari layer is active  
        self._landmark_layers[role] = layer
        
    
    
    # look at the currently selected flm point and and currectly selected tem point, and give  those two points the same new pair_id
        """
        get active FLM layer
        get active TEM layer
            next    
        make sure both exist
            next
        ask each layer:
        "Which points are selected?"
                
        require exactly ONE selected in each
                
        find a new unused pair ID
                
        assign that same ID to both selected points
                
        display that ID beside both points
                
        """
    #  it would return the newly created pair number
    def pair_selected_landmarks(
        self,
    ) -> int:
        
        print("i am in this function")
        
        flm_layer = self._landmark_layers["FLM"]
        tem_layer = self._landmark_layers["TEM"]
        
        if flm_layer is None or tem_layer is None:
            raise ValueError(
                "assign both FLM and TEM landmark layers first"
            )
        
        # these are the points that are currently selected in the napari
        # we wanna make sure only one each layer
        flm_selected = list(flm_layer.selected_data)
        tem_selected = list(tem_layer.selected_data)
        
        if len(flm_selected) != 1:
            raise ValueError(
                "select exactly one FLM landmark"
            )
            
        if len(tem_selected) != 1:
            raise ValueError(
                "select exactly one TEM landmark"
            )
        # get the index of each
        
        flm_index = flm_selected[0]
        tem_index = tem_selected[0]
        
        # keep the track so we can answer what pair numbers have already been used, so what number should I use next?
        existing_pair_ids = []
        
        # look through both FLM and TEM metadata.
        for layer in (
            flm_layer,
            tem_layer,
        ):
            features = layer.features # get the layers metadata table (features)

            if "pair_id" in features: # does it have a column named "pair_id"
                for pair_id in features["pair_id"]:
                    if str(pair_id).strip():
                        existing_pair_ids.append(int(pair_id))
                        

        # find the next id that we will assign to the pair
        next_pair_id = (max(existing_pair_ids, default=0) + 1)
        

        # make a copy cause we are going to modify it later put it back
        flm_features = flm_layer.features.copy()
        tem_features = tem_layer.features.copy()

        # if pair_id column does not exist yet then create it based on however many points we have fill it wi ""
        if "pair_id" not in flm_features:
            flm_features["pair_id"] = [
                ""
            ] * len(flm_layer.data)

        if "pair_id" not in tem_features:
            tem_features["pair_id"] = [
                ""
            ] * len(tem_layer.data)

        # give me the FLM pair_id column as a mumpy array that I can easily edit by point index
        flm_pair_ids = np.asarray(
            flm_features["pair_id"],
            dtype=object,
        ).copy()

        tem_pair_ids = np.asarray(
            tem_features["pair_id"],
            dtype=object,
        ).copy()

        # where the actual pairing happens flm_pair_ids[2] = "1"
        flm_pair_ids[flm_index] = str(
            next_pair_id
        )

        tem_pair_ids[tem_index] = str(
            next_pair_id
        )

        #puts those modified arrays back into our feature-table copies.
        flm_features["pair_id"] = flm_pair_ids
        tem_features["pair_id"] = tem_pair_ids

        #give the updated feature tables back to napari
        flm_layer.features = flm_features
        tem_layer.features = tem_features

        # visually show pair id
        flm_layer.text = "{pair_id}"
        tem_layer.text = "{pair_id}"

        return next_pair_id
    
    # helper function For this Points layer, where is each explicit pair ID located?
    # input: napari layer
    # output: dictionary key  = pair ID , value = point index
    def _get_pair_index_map(self, layer) -> dict[int,int]:
        
        # create dic to hold 
        pair_index_map = {}
        
        # if the feature does not exist 
        if "pair_id" not in layer.features:
            return pair_index_map

        for index, pair_id in enumerate(
            layer.features["pair_id"]
        ):
            pair_id_text = str(pair_id).strip()

            if not pair_id_text:
                continue

            pair_id_number = int(pair_id_text)

            if pair_id_number in pair_index_map:
                raise ValueError(
                    f"pair ID {pair_id_number} appears more than once "
                    f"in {layer.name}"
                )

            # do the actual mapping
            pair_index_map[pair_id_number] = index

        return pair_index_map
            
    
    
    """
    
    this function sits between napari landmark layer and the fit_affine
    Its only job is to prepare two correctly corresponding Points2D objects.  
    
    Example without pairing:

    FLM                    TEM

    index 0                index 0
    index 1                index 1
    index 2                index 2
    index 3                index 3

    returns everything normally.

    Example with explicit pairing:

    FLM                         TEM

    index 0 pair 2              index 0 pair 3
    index 1 unpaired            index 1 pair 1
    index 2 pair 1              index 2 pair 2
    index 3 pair 3

    The function reconstructs:

    Registration FLM        Registration TEM

    pair 1: FLM index 2     pair 1: TEM index 1
    pair 2: FLM index 0     pair 2: TEM index 2
    pair 3: FLM index 3     pair 3: TEM index 0

    Now fit_affine() receives rows that truly correspond.
            
    """
    def get_registration_landmarks(self):

        # get the active layers
        flm_layer = self._landmark_layers["FLM"]
        tem_layer = self._landmark_layers["TEM"]

        # error handeling
        if flm_layer is None or tem_layer is None:
            raise ValueError(
                "assign both FLM and TEM landmark layers first"
            )

        # refresh current coordinates
        self.use_points_layer(
            "FLM",
            flm_layer,
        )

        self.use_points_layer(
            "TEM",
            tem_layer,
        )

        # look for pair_id metadata
        flm_pair_map = self._get_pair_index_map(
            flm_layer
        )

        tem_pair_map = self._get_pair_index_map(
            tem_layer
        )

        # no explicit pairs
        if not flm_pair_map and not tem_pair_map:

            if len(self.session.flm.points.xy) != len(
                self.session.tem.points.xy
            ):
                raise ValueError(
                    "FLM and TEM must have the same number of landmarks"
                )

            # return all points in their existing order
            return (
                self.session.flm.points,
                self.session.tem.points,
            )

        
        #explicit pairs exist
        
        common_pair_ids = sorted(
            set(flm_pair_map)
            & set(tem_pair_map)
        )

        if len(common_pair_ids) < 3:
            raise ValueError(
                "explicit correspondence requires at least 3 matched pairs"
            )

        # match pair IDs
        flm_indices = [
            flm_pair_map[pair_id]
            for pair_id in common_pair_ids
        ]

        tem_indices = [
            tem_pair_map[pair_id]
            for pair_id in common_pair_ids
        ]

        flm_points = Points2D(
            self.session.flm.points.xy[
                flm_indices
            ]
        )

        tem_points = Points2D(
            self.session.tem.points.xy[
                tem_indices
            ]
        )

        #   return only corresponding points in pair-ID order
        return (
            flm_points,
            tem_points,
        )
                
        
        
        
            
