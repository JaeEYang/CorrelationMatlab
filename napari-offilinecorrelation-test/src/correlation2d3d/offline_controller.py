"""
 Controller
    =
application behavior
"""
import numpy as np

from correlation2d3d.session import CorrelationSession
from correlation2d3d.core.transform import (
    apply_affine_matrix,
    affine_xy_to_rc,
    Registration2D,
)
from correlation2d3d.core.orientation import (
    orientation_matrix_from_settings,
)
from correlation2d3d.core.warp import warp_image

from correlation2d3d.core.geometry import Points2D
import imageio.v3 as iio
import tifffile
import json

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
    # pixels unchanged napari renderer will move the image
    def set_modality_orientation(
        self,
        role: str,
        orientation_matrix: np.ndarray,
    ) -> None:
        modality = self._get_modality(role)
        
        #  Which napari Points layer are we currently using for this modality?
        layer = self._landmark_layers[role]
        
        # Only synchronize if an active layer actually exists and hasn't been deleted from napari.
        if (
            layer is not None
            and layer in self.viewer.layers
        ):   
            self.use_points_layer( # synchronize
                role,
                layer,
            )

        modality.orientation_matrix = np.array(
            orientation_matrix,
            dtype=np.float64,
            copy=True,
        )

        #always rebuild current points from the
        #orriginal landmarks.
        if modality.original_points is not None:
            modality.points = apply_affine_matrix(
                modality.orientation_matrix,
                modality.original_points,
            )
        else:
            modality.points = None
            
        image_layer = self.viewer.layers[role]


        image_layer.affine = affine_xy_to_rc(
            modality.orientation_matrix
        )

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
    # going to keep this function name temporarily, even though "baseline" is becoming a misleading name. That minimizes how much code we change in one pass.
    def rebuild_modality_from_baseline(
        self,
        role: str,
    ) -> None:

        modality = self._get_modality(role)

        if modality.image is None:
            return

        height, width = (
            modality.image.shape[:2]
        )

        # get the setting and save to matrix 
        orientation_matrix = (
            orientation_matrix_from_settings(
                height,
                width,
                modality.rotation_angle,
                horizontal_flipped=(
                    modality.horizontal_flipped
                ),
                vertical_flipped=(
                    modality.vertical_flipped
                ),
            )
        )

        self.set_modality_orientation(
            role,
            orientation_matrix,
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

        modality.horizontal_flipped = (
            not modality.horizontal_flipped
        )

        self.rebuild_modality_from_baseline(
            role
        )
        
    # veritcal flip 
    def flip_modality_vertical(
        self,
        role: str,
    ) -> None:

        modality = self._get_modality(role)

        if modality.image is None:
            return

        modality.vertical_flipped = (
            not modality.vertical_flipped
        )

        self.rebuild_modality_from_baseline(
            role
        )
        
    # this just resets eveything to beginning!
    def reset_modality_orientation(
        self,
        role: str,
    ) -> None:
        modality = self._get_modality(role)

        if modality.image is None:
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

        if modality.image is None:
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
            image,
            copy=True,
        )

        # Reloading the image clears orientation adjustments, but keeps the centering translation.
        # save the current coordinate mapping which is padding.
        modality.orientation_matrix = np.eye(
            3,
            dtype=np.float64,
        )


        # fixed starting image
        modality.rotation_base_image = np.array(
            image,
            copy=True,
        )

        #records how do I get from original image coordinates to this fixed baseline?
        modality.rotation_base_orientation_matrix = (
            np.eye(
                3,
                dtype=np.float64,
            )
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
        
        
    # Where does the Registered FLM currently map its pixel coordinates into napari world coordinates?
    # data_to_world method in napari layer Converts from data coordinates to world coordinates.
    def get_registered_flm_transform_rc(
        self,
    ) -> np.ndarray:

        try:
            layer = self.viewer.layers[
                "Registered FLM"
            ]
        except KeyError:
            raise ValueError(
                "Registered FLM does not exist"
            )

        origin = np.asarray(
            layer.data_to_world(
                (0.0, 0.0)
            ),
            dtype=np.float64,
        )

        row_point = np.asarray(
            layer.data_to_world(
                (1.0, 0.0)
            ),
            dtype=np.float64,
        )

        column_point = np.asarray(
            layer.data_to_world(
                (0.0, 1.0)
            ),
            dtype=np.float64,
        )

        row_direction = (
            row_point - origin
        )

        column_direction = (
            column_point - origin
        )

        return np.array([
            [
                row_direction[0],
                column_direction[0],
                origin[0],
            ],
            [
                row_direction[1],
                column_direction[1],
                origin[1],
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ], dtype=np.float64)
        
    def get_registered_images_on_common_grid(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.session.tem.image is None:
            raise ValueError(
                "TEM image must be loaded before creating the registered images"
            )

        try:
            registered_flm_layer = self.viewer.layers[
                "Registered FLM"
            ]
        except KeyError:
            raise ValueError(
                "Registered FLM does not exist"
            )

        tem = np.asarray(
            self.session.tem.image
        )

        flm = np.asarray(
            registered_flm_layer.data
        )

        tem_transform_xy = np.asarray(
            self.session.tem.orientation_matrix,
            dtype=np.float64,
        )

        flm_transform_rc = (
            self.get_registered_flm_transform_rc()
        )

        swap_rc_xy = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        flm_transform_xy = (
            swap_rc_xy
            @ flm_transform_rc
            @ swap_rc_xy
        )

        tem_height, tem_width = tem.shape[:2]

        tem_corners = Points2D(
            np.array([
                [0.0, 0.0],
                [tem_width - 1.0, 0.0],
                [0.0, tem_height - 1.0],
                [tem_width - 1.0, tem_height - 1.0],
            ], dtype=np.float64)
        )

        flm_height, flm_width = flm.shape[:2]

        flm_corners = Points2D(
            np.array([
                [0.0, 0.0],
                [flm_width - 1.0, 0.0],
                [0.0, flm_height - 1.0],
                [flm_width - 1.0, flm_height - 1.0],
            ], dtype=np.float64)
        )

        transformed_tem_corners = (
            apply_affine_matrix(
                tem_transform_xy,
                tem_corners,
            )
        )

        transformed_flm_corners = (
            apply_affine_matrix(
                flm_transform_xy,
                flm_corners,
            )
        )

        all_corners = np.vstack([
            transformed_tem_corners.xy,
            transformed_flm_corners.xy,
        ])

        min_x = np.floor(
            all_corners[:, 0].min()
        )

        min_y = np.floor(
            all_corners[:, 1].min()
        )

        max_x = np.ceil(
            all_corners[:, 0].max()
        )

        max_y = np.ceil(
            all_corners[:, 1].max()
        )

        output_width = int(
            max_x - min_x + 1
        )

        output_height = int(
            max_y - min_y + 1
        )

        common_shift = np.array([
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        tem_to_common = (
            common_shift
            @ tem_transform_xy
        )

        flm_to_common = (
            common_shift
            @ flm_transform_xy
        )

        output_shape = (
            output_height,
            output_width,
        )

        registered_tem = warp_image(
            tem,
            Registration2D(
                matrix=tem_to_common
            ),
            output_shape=output_shape,
            order=1,
        )

        registered_flm = warp_image(
            flm,
            Registration2D(
                matrix=flm_to_common
            ),
            output_shape=output_shape,
            order=1,
        )

        if np.issubdtype(
            tem.dtype,
            np.integer,
        ):
            tem_info = np.iinfo(
                tem.dtype
            )

            registered_tem = np.clip(
                np.rint(registered_tem),
                tem_info.min,
                tem_info.max,
            ).astype(
                tem.dtype
            )

        else:
            registered_tem = (
                registered_tem.astype(
                    tem.dtype,
                    copy=False,
                )
            )

        if np.issubdtype(
            flm.dtype,
            np.integer,
        ):
            flm_info = np.iinfo(
                flm.dtype
            )

            registered_flm = np.clip(
                np.rint(registered_flm),
                flm_info.min,
                flm_info.max,
            ).astype(
                flm.dtype
            )

        else:
            registered_flm = (
                registered_flm.astype(
                    flm.dtype,
                    copy=False,
                )
            )

        return (
            registered_tem,
            registered_flm,
        )    
    
        
    def save_scientific_tiff(
        self,
        path: str,
    ) -> None:

        (
            registered_tem,
            registered_flm,
        ) = (
            self.get_registered_images_on_common_grid()
        )

        if registered_tem.ndim == 2:
            registered_tem = np.repeat(
                registered_tem[..., np.newaxis],
                3,
                axis=-1,
            )

        if registered_flm.ndim == 2:
            registered_flm = np.repeat(
                registered_flm[..., np.newaxis],
                3,
                axis=-1,
            )

        registered_tem = registered_tem[..., :3]
        registered_flm = registered_flm[..., :3]

        scientific_stack = np.stack(
            [
                registered_tem,
                registered_flm,
            ],
            axis=0,
        )

        tifffile.imwrite(
            path,
            scientific_stack,
            photometric="rgb",
            compression=None,
        )


    def open_scientific_tiff(
        self,
        path: str,
    ) -> None:

        with tifffile.TiffFile(
            path
        ) as tif:
            stack = tif.asarray()

        if (
            stack.ndim != 4
            or stack.shape[0] != 2
            or stack.shape[-1] != 3
        ):
            raise ValueError(
                "scientific TIFF must contain two RGB registered images"
            )

        tem = stack[0]
        flm = stack[1]

        self._remove_layer_if_present(
            "Scientific TEM"
        )

        self._remove_layer_if_present(
            "Scientific Registered FLM"
        )

        self.viewer.add_image(
            tem,
            name="Scientific TEM",
            rgb=True,
        )

        self.viewer.add_image(
            flm,
            name="Scientific Registered FLM",
            rgb=True,
            opacity=0.5,
            blending="translucent",
        )
    def capture_visual_overlay(
        self,
    ) -> np.ndarray:

        try:
            tem_layer = self.viewer.layers[
                "TEM"
            ]

            registered_flm_layer = self.viewer.layers[
                "Registered FLM"
            ]

        except KeyError:
            raise ValueError(
                "TEM and Registered FLM must exist before exporting"
            )

        # remebers the currenct state
        visibility = {
            layer: layer.visible
            for layer in self.viewer.layers
        }

        # temporarily makes the viewer: TEM = ON Registered FLM  = ON , everything else OFF
        try:
            for layer in self.viewer.layers:
                layer.visible = (
                    layer is tem_layer
                    or layer is registered_flm_layer
                )

            overlay = self.viewer.export_figure(
                scale_factor=1,
                flash=False,
            )

        finally:
            for layer, was_visible in visibility.items():
                layer.visible = was_visible

        return overlay
    
    def save_visual_overlay(
    self,
    path: str,
) -> None:
        #gets the actual rendered image.
        overlay = self.capture_visual_overlay()

        lower_path = path.lower()

        if lower_path.endswith(
            (".tif", ".tiff")
        ):
            tifffile.imwrite(
                path,
                overlay,
                compression=None,
                photometric="rgb",
            )

        elif lower_path.endswith(".png"):
            iio.imwrite(
                path,
                overlay,
            )

        elif lower_path.endswith(
            (".jpg", ".jpeg")
        ):
            rgb_overlay = overlay[..., :3]

            iio.imwrite(
                path,
                rgb_overlay,
                quality=95,
            )

        else:
            raise ValueError(
                "visual overlay must be TIFF, PNG, or JPEG"
            )
                    
                        
