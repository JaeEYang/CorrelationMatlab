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

class OfflineCorrelationController:

    def __init__(
        self,
        viewer,
        session: CorrelationSession,
    ):
        # keep the viewer and session from the widget here
        # we handle the workflow, the widget still handles buttons, messages and invalidation
        self.viewer = viewer
        self.session = session
        
        # this eventually means which points layer is currently being used for each modality cause there would be multiple csv, user
        self._landmark_layers = {
            "FLM": None,
            "TEM": None,
        }
        # which image layer is being used by each role
        self._image_layers = {
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
        
    # gives the rest of the program the acutal napri image layer currectly assigned to FLM or TEM
    # give me the layer that we assigned to the role
    def get_modality_image_layer(
        self,
        role: str,
    ):

        # check the role first so a typo gives us a useful error instead of a KeyError
        self._get_modality(role)

        # get the saved layer object, not its name, so renaming it doesn't lose the assignment
        layer = self._image_layers[role]

        # we might still have a reference after the layer was deleted, so check it's still in napari
        if (
            layer is None
            or layer not in self.viewer.layers
        ):
            raise ValueError(
                f"{role} image layer is not assigned"
            )

        return layer
       
    # assigning an already loaded napari Image layer as either FLM or TEM 
    # store the layer in _image_layers, copies its pixel data into the session
    # resets orientation settings,
    # clears old landmakrs for that role but leaves that napari layer itself intact
    def use_image_layer(
        self,
        role: str,
        layer,
    ) -> None:

        # modality is basically  self.session.flm
        modality = self._get_modality(
            role
        )

        # lets prevent same image layer be flm and tem
        other_role = (
            "TEM"
            if role == "FLM"
            else "FLM"
        )

        # if current layer is already associated to a role then error handle 
        if (
            layer
            is self._image_layers[other_role]
        ):
            raise ValueError(
                f"{layer.name} is already assigned as {other_role}"
            )

        # add the layer to the dictionaly now and this is associated with the current role
        self._image_layers[role] = layer

        # copy the pixels as they are now
        # these aren't live links, changing layer.data later won't update the session copies
        modality.original_image = np.array(
            layer.data,
            copy=True,
        )

        modality.image = np.array(
            layer.data,
            copy=True,
        )

        # start our plugin settings at identity
        # any transform already on the chosen image stays there and gets read back
        # from the layer
        modality.orientation_matrix = np.eye(
            3,
            dtype=np.float64,
        )

        modality.rotation_angle = 0.0
        modality.horizontal_flipped = False
        modality.vertical_flipped = False

        # the old landmarks belonged to the previous assignment, so stop using them for this role
        # their layers stay in napari, we aren't deleting any points
        # assigning the same source again also clears the landmark link
        modality.original_points = None
        modality.points = None

        self._landmark_layers[role] = None
    
    
    # keep the session pixel data synchronized if an assigned napari Image layer's
    # underlying data array is replaced or processed
    # get the assigned image layer, compare its current pixel array shapr wht session array, copies the new pixels if the shpae is unchanged, and treats a shape chagnes as a fresh assignment
    def sync_modality_image_data(
        self,
        role:str,
    ) -> bool:
        
        modality = self._get_modality(role)
        
        image_layer = self.get_modality_image_layer(role)
        
        current_data = np.asarray(image_layer.data)
        
         # same shape means we assume the pixel coordinate grid is unchanged
        if (
            modality.image is not None
            and current_data.shape == modality.image.shape
        ):
            modality.image = np.array(current_data, copy = True) #just copy the pixel data
            return False # means shape is not not diffrence 
        
        # a different shape can change the meaning of pixel coordinates
        # treat this as a fresh source assignment instead of silently keeping landmarks tied to the old grid
        self.use_image_layer(
            role,
            image_layer,
        )

        return True
        
        
    # do the rotation 
    def set_modality_rotation(
        self,
        role: str,
        angle_degrees: float,
    ) -> None:
        modality = self._get_modality(role) # get the session.flm or .tem 

        if modality.image is None:
            return

        # store the angle we want, don't add it to the old angle
        # so going from 30 to 31 means build 31, not rotate by another 31
        modality.rotation_angle = angle_degrees

        self.rebuild_modality_orientation(
            role
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

        # switch the flip on or off, clicking twice gets back to where we started
        # leave the angle alone and rebuild using both settings, don't change what
        # zero means
        modality.horizontal_flipped = (
            not modality.horizontal_flipped
        )

        self.rebuild_modality_orientation(
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

        # same thing for top/bottom, leave the horizontal flip setting alone
        modality.vertical_flipped = (
            not modality.vertical_flipped
        )

        self.rebuild_modality_orientation(
            role
        )
        
    # this just resets eveything to beginning! whatever done in our not the native napari
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

        self.rebuild_modality_orientation(
            role
        )
       
                
    # rebuild the plugin-owned orientation matrix from the current
    # rotation and flip settings, then compose it with the existing manual transform.
    def rebuild_modality_orientation(
        self,
        role: str,
    ) -> None:

        modality = self._get_modality(role)

        # no image assigned yet, so there's nothing to rotate
        if modality.image is None:
            return

        # get the source height and width
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

        # pass the matrix to the helper that updates the image placement and both versions of the landmarks
        self.set_modality_orientation(
            role,
            orientation_matrix,
        )
        

    
    # When orientation changes, transform the session coordinates and move the currently assigned Points layer.
    # pixels unchanged napari renderer will move the image
    def set_modality_orientation(
        self,
        role: str,
        orientation_matrix: np.ndarray,
    ) -> None:
        modality = self._get_modality(role)

        image_layer = self.get_modality_image_layer(
            role
        )

        # get where the image is right now, including anything moved manually in napari
        # T is that whole transform, O is just our plugin rotation and flips
        current_transform = (
            self.get_modality_transform_xy(role)
        )

        old_orientation = np.asarray(
            modality.orientation_matrix,
            dtype=np.float64,
        )

        # T = U @ O, so take out the old O with T @ inv(O) to get U
        # now we can change the rotation and flips without losing the manual
        # alignment
        manual_transform = (
            current_transform
            @ np.linalg.inv(old_orientation)
        )

        layer = self._landmark_layers[role]

        if (
            layer is not None
            and layer in self.viewer.layers
        ):
            # pick up any landmark edits before moving the image
            # use the old inverse transform to work out where the points belong in
            # the source image
            self.use_points_layer(
                role,
                layer,
            )

        # keep our own copy of the new orientation matrix
        modality.orientation_matrix = np.array(
            orientation_matrix,
            dtype=np.float64,
            copy=True,
        )

        # put the new O together with the U we kept
        # we're rebuilding a matrix here, not rotating the pixels again
        complete_transform = (
            manual_transform
            @ modality.orientation_matrix
        )

        # our math uses x/y but napari wants row/column, so swap the order here
        # this moves the image layer, the pixels themselves stay the same
    
        image_layer.affine = affine_xy_to_rc(
            complete_transform
        )

        # move the saved source points using the same new transform
        # it's fine to rotate an image before loading any landmarks
        if modality.original_points is not None:
            modality.points = apply_affine_matrix(
                complete_transform,
                modality.original_points,
            )
        else:
            modality.points = None

        if (
            modality.points is not None
            and layer is not None
            and layer in self.viewer.layers
        ):
            # give the new coordinates to napari too, updating the session alone
            # won't move anything onscreen
            # the Points data listener also sees this update even though we made it
            # from code
            layer.data = modality.points.to_rc()
    
    """
    asks napari where the source FLM or TEM actually is right now, 
    including our orientation plus native napari rough alignment. 
    2. Input? "FLM" or "TEM". 
    3. Internally? Sample three data coordinates with data_to_world(), reconstruct the affine,
    then convert napari row/column convention into our x/y convention. 
    4. Return? 
    A 3×3 data-to-world affine in x/y coordinates. 
    This is the source-image equivalent of what we already do for Registered FLM. 
    """
    def get_modality_transform_xy(
        self,
        role: str,
    ) -> np.ndarray:

        layer = self.get_modality_image_layer(
            role
        )

        # ask where (0,0), one row down and one column right end up
        # for a 2D affine that's enough to get the shift and the two axis directions
        # data_to_world includes the other layer transform settings too, not just affine
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

        # subtract the moved origin so we get directions, not positions
        row_direction = (
            row_point - origin
        )

        column_direction = (
            column_point - origin
        )

        # columns 0 and 1 store the row and column directions, column 2 stores the
        # shift
        transform_rc = np.array([
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

        # swap on both sides, x/y input -> row/column math -> x/y output
        # just swapping one side would mix up the coordinate orders
        swap_rc_xy = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        return (
            swap_rc_xy
            @ transform_rc
            @ swap_rc_xy
        ) 
        
    """
    move existing active landmarks whenever the source image is transformed in napari. 
    2. Input? "FLM" or "TEM". 
    3. Internally? Read the complete current image transform and apply it to the saved original landmark coordinates. 
    4. Return? Nothing, it updates session state and the Points layer. 
    Landmarks are pinned to structures in the image, so rough-moving the image should carry its landmarks with it. 
    """
    def sync_landmarks_to_modality_transform(
        self,
        role: str,
    ) -> None:

        modality = self._get_modality(role)

        # no saved source points yet, so there are no landmarks to move
        if modality.original_points is None:
            return

        layer = self._landmark_layers[role]

        # the Points layer might be missing or not assigned yet, just leave it alone
        if (
            layer is None
            or layer not in self.viewer.layers
        ):
            return

        current_transform = (
            self.get_modality_transform_xy(role)
        )

        # start from the saved source points every time, not their last onscreen  positions
        # otherwise we'd end up moving the same points twice
        modality.points = apply_affine_matrix(
            current_transform,
            modality.original_points,
        )

        layer.data = modality.points.to_rc()   
        
    
    #cleanly unassing FLM or TEM when its source image layer disappears from napari
    # clears the storage image layer reference, image arrays, orientation state,
    # and active landmark association for that modality
    def clear_image_layer(
        self,
        role: str,
    ) -> None:

        modality = self._get_modality(
            role
        )

        self._image_layers[role] = None

        modality.original_image = None
        modality.image = None

        modality.orientation_matrix = np.eye(
            3,
            dtype=np.float64,
        )

        modality.rotation_angle = 0.0
        modality.horizontal_flipped = False
        modality.vertical_flipped = False

        modality.original_points = None
        modality.points = None

        # stop using the old landmark layer but leave it in the viewer
        # the widget disables the controls and clears the old registration
        self._landmark_layers[role] = None
        
     
        
        
        # generic Points candidates in napari
       
    def create_points_layer_from_original_points(
        self,
        points: Points2D,
        name: str,
    ):
        # no image role yet, just add the raw points as a hidden candidate
        # Use Landmarks will place them with the chosen image's current transform
        layer = self.viewer.add_points(
            points.to_rc(),
            name=name,
            size=32,
            face_color="red",
            visible=False,
            metadata={
                "correlation2d3d_original_points_xy":
                    points.xy.copy(), #these coordinates came directly from a CSV and still represent original source-image coordinates.
            },
        )

        return layer
        
        
    

    """
    CSV-loaded candidate
    original coordinates known

    world = F x original

    ordinary/manually edited Points layer
    world coordinates known

    original = inv(F) x world
    
    """
    def use_points_layer(
        self,
        role: str,
        layer,
    ) -> None:

        # get the correct modality (session.flm or .tem)
        modality = self._get_modality(role)
        
        # don't let the same Points layer be both FLM and TEM landmarks
        other_role = (
            "TEM"
            if role == "FLM"
            else "FLM"
        )

        if (
            layer
            is self._landmark_layers[other_role]
        ):
            raise ValueError(
                f"{layer.name} is already assigned as "
                f"{other_role} landmarks"
            )

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

        # CSV candidates carry source coordinates, ordinary napari points may not
        csv_original_xy = layer.metadata.get(
            "correlation2d3d_original_points_xy"
        )

        # active points are already placed, even if they originally came from CSV
        # this also catches the data event sent by the first placement below
        already_active = (
            self._landmark_layers[role] is layer
        )

        # first assignment or reactivation starts from the saved source coordinates
        if (
            csv_original_xy is not None
            and not already_active
        ):
            original_points = Points2D(
                np.asarray(
                    csv_original_xy,
                    dtype=np.float64,
                )
            )

            current_transform = (
                self.get_modality_transform_xy(role)
            )

            working_points = apply_affine_matrix(
                current_transform,
                original_points,
            )

            modality.original_points = original_points
            modality.points = working_points

            #mark it active before changing layer.data because changing
            #layer.data emits a napari data event
            #so basically says this isn't initial CSV placement anymore
            #this layer is now an active world-coordinate layer
            self._landmark_layers[role] = layer

            # now the points match this image, so it makes sense to show them
            layer.data = working_points.to_rc()
            layer.visible = True

            return

        # ordinary points and already-active CSV points are read as world positions
        # invert the image transform below to remember their source positions
        # convert into our internal xy convention
        working_points = Points2D.from_rc(
            layer_data
        )

        #these are the landmark coordinates currently shown in napari world space.
        modality.points = working_points

        #if manually rough-align the FLM and then click a landmark at: world = (900, 650) calculate: p_original =F_FLM^-1p_world
        current_transform = (
            self.get_modality_transform_xy(role)
        )

        inverse_transform = np.linalg.inv(
            current_transform
        )

        # apply the inverse and after that is can just follow the same path as csv
        modality.original_points = apply_affine_matrix(
            inverse_transform,
            modality.points,
        )

        # remeber which napari layer is active
        self._landmark_layers[role] = layer

        # save active edits in source coordinates for the next reactivation
        # this is the latest source-position snapshot, not an untouched CSV backup
        if csv_original_xy is not None:
            layer.metadata[
                "correlation2d3d_original_points_xy"
            ] = modality.original_points.xy.copy()
        
    
    
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
        
        #these are the points that are currently selected in the napari
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
        
        #keep the track so we can answer what pair numbers have already been used, so what number should I use next?
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
                        

        #find the next id that we will assign to the pair
        next_pair_id = (max(existing_pair_ids, default=0) + 1)
        

        # make a copy cause we are going to modify it later put it back
        flm_features = flm_layer.features.copy()
        tem_features = tem_layer.features.copy()

        #if pair_id column does not exist yet then create it based on however many points we have fill it wi ""
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
        
        # handle duplicate pair id
        flm_existing_pair = str(
            flm_pair_ids[flm_index]
        ).strip()

        tem_existing_pair = str(
            tem_pair_ids[tem_index]
        ).strip()

        if (
            flm_existing_pair
            and tem_existing_pair
            and flm_existing_pair == tem_existing_pair
        ):
            raise ValueError(
                f"selected landmarks are already paired as "
                f"pair {flm_existing_pair}"
            )

        if flm_existing_pair:
            raise ValueError(
                f"selected FLM landmark is already pair "
                f"{flm_existing_pair}; unpair it first"
            )

        if tem_existing_pair:
            raise ValueError(
                f"selected TEM landmark is already pair "
                f"{tem_existing_pair}; unpair it first"
            )

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
            
            
    def unpair_selected_landmarks(
        self,
    ) -> int:

        flm_layer = self._landmark_layers["FLM"]
        tem_layer = self._landmark_layers["TEM"]

        if flm_layer is None or tem_layer is None:
            raise ValueError(
                "assign both FLM and TEM landmark layers first"
            )

        # we can unpair by selecting just one end of a pair
        # selecting both ends is fine too, as long as they're the same pair
        flm_selected = list(
            flm_layer.selected_data
        )

        tem_selected = list(
            tem_layer.selected_data
        )

        if (
            len(flm_selected) > 1
            or len(tem_selected) > 1
        ):
            raise ValueError(
                "select at most one landmark in each layer"
            )

        if (
            not flm_selected
            and not tem_selected
        ):
            raise ValueError(
                "select a paired landmark first"
            )

        # get the pair ID from each selected point
        # one side can have no selection, but a selected point needs an ID to unpair
        selected_pair_ids = []

        for layer, selected in (
            (flm_layer, flm_selected),
            (tem_layer, tem_selected),
        ):
            if not selected:
                continue

            if "pair_id" not in layer.features:
                raise ValueError(
                    "selected landmark is not paired"
                )

            index = selected[0]

            pair_id = str(
                layer.features["pair_id"].iloc[index]
            ).strip()

            if not pair_id:
                raise ValueError(
                    "selected landmark is not paired"
                )

            selected_pair_ids.append(
                pair_id
            )

        # both selected points should give us the same ID
        # if we get two different IDs, leave the labels alone and ask the user to pick again
        unique_pair_ids = set(
            selected_pair_ids
        )

        if len(unique_pair_ids) != 1:
            raise ValueError(
                "selected landmarks belong to different pairs"
            )

        # now we know which pair to clear from both layers
        # we only remove the labels, we don't move or delete either point
        pair_id = unique_pair_ids.pop()

        for layer in (
            flm_layer,
            tem_layer,
        ):
            if "pair_id" not in layer.features:
                continue

            features = layer.features.copy()

            pair_ids = np.asarray(
                features["pair_id"],
                dtype=object,
            ).copy()

            # find the rows with this ID and blank their label, leave the other pairs
            # alone
            pair_ids[
                np.asarray(
                    [
                        str(value).strip()
                        == pair_id
                        for value in pair_ids
                    ]
                )
            ] = ""

            features["pair_id"] = pair_ids

            layer.features = features
            layer.text = "{pair_id}"

        return int(pair_id)
    
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

            # blank means this point isn't paired, not that it's pair zero, so skip
            # it
            if not pair_id_text:
                continue

            pair_id_number = int(pair_id_text)

            # each pair ID needs one point per layer
            # if two points have the same ID we can't know which one to match, so stop here
            if pair_id_number in pair_index_map:
                raise ValueError(
                    f"pair ID {pair_id_number} appears more than once "
                    f"in {layer.name}"
                )

            # do the actual mapping
            pair_index_map[pair_id_number] = index

        return pair_index_map
    
    def clear_orphaned_pairs(
        self,
    ) -> None:

        flm_layer = self._landmark_layers["FLM"]
        tem_layer = self._landmark_layers["TEM"]

        if (
            flm_layer is None
            or tem_layer is None
            or flm_layer not in self.viewer.layers
            or tem_layer not in self.viewer.layers
        ):
            return

        # deleting a point can leave its partner with an ID that no longer has a match
        # compare the remaining IDs in both active layers to find those
        flm_pair_map = self._get_pair_index_map(
            flm_layer
        )
        tem_pair_map = self._get_pair_index_map(
            tem_layer
        )

        #  finds IDs that are only on one side
        # IDs on both sides still have a partner, so keep those
        orphaned_pair_ids = (
            set(flm_pair_map)
            ^ set(tem_pair_map)
        )

        # if a point just moved, the pairs usually haven't changed, so nothing to clean up
        if not orphaned_pair_ids:
            return

        # the maps use integer IDs but the labels are compared as strings with spaces stripped off
        orphaned_pair_text = {
            str(pair_id)
            for pair_id in orphaned_pair_ids
        }

        for layer in (
            flm_layer,
            tem_layer,
        ):
            if "pair_id" not in layer.features:
                continue

            features = layer.features.copy()

            pair_ids = np.asarray(
                features["pair_id"],
                dtype=object,
            ).copy()

            orphaned_mask = np.asarray(
                [
                    str(value).strip()
                    in orphaned_pair_text
                    for value in pair_ids
                ],
                dtype=bool,
            )

            # make those remaining points unpaired again, keep the points themselves
            pair_ids[orphaned_mask] = ""

            features["pair_id"] = pair_ids
            layer.features = features
            layer.text = "{pair_id}"
            
    
    
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
        if (
            flm_layer is None
            or tem_layer is None
            or flm_layer not in self.viewer.layers
            or tem_layer not in self.viewer.layers
        ):
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

        # once either layer has pair IDs, only use IDs that match on both sides
        # don't switch back to row order or pull in unpaired points
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

    # helper if layer exists we can remove it.
    def _remove_layer_if_present(
        self,
        layer_name: str,
    ) -> None:
        try:
            layer = self.viewer.layers[layer_name]
        except KeyError:
            return

        self.viewer.layers.remove(layer)
        
    #lets say we did the registration and then changed the image orientation 
    # we should able to invalidate the old registraion matrix will have to recalculate
  
    def invalidate_registration(self) -> None:
        self.session.registration = None
      
        # this removed those two layers aswell
        self._remove_layer_if_present(
            "FLM Landmarks Registered to TEM"
        )

        self._remove_layer_if_present(
            "Registered FLM"
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

        # read the actual Registered FLM layer, not just the matrix from the fit
        # that way export includes any manual adjustments made to the result
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

        # same idea as the source helper, subtract the shift to get the axis directions
        row_direction = (
            row_point - origin
        )

        column_direction = (
            column_point - origin
        )

        # leave this one in row/column order, export swaps it to x/y
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

        # this is where we make two real output arrays on the same grid
        # the registration preview only places an image layer using a transform
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

        # get TEM pixels from the session copy and FLM pixels from the result layer
        # opacity and contrast settings aren't part of this scientific export
        tem = np.asarray(
            self.session.tem.image
        )

        flm = np.asarray(
            registered_flm_layer.data
        )

        # read the current TEM and Registered FLM transforms
        # both layers might have moved since we calculated registration
        tem_transform_xy = (
            self.get_modality_transform_xy(
                "TEM"
            )
        )

        flm_transform_rc = (
            self.get_registered_flm_transform_rc()
        )

        # these helpers return different coordinate orders
        # put both in x/y before comparing bounds or combining transforms
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

        # use the corner pixel centers, from zero to width/height minus one
        # an affine keeps the edges straight, so the four corners tell us the bounds
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

        # move both sets of corners into world coordinates first so we can compare them
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

        # fit both whole images on the grid, not just the overlap or just the TEM area
        all_corners = np.vstack([
            transformed_tem_corners.xy,
            transformed_flm_corners.xy,
        ])

        # round the bounds outwards so corners between pixels still fit
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

        # count both ends, pixels 0 through 9 need 10 columns, that's why we add 1
        output_width = int(
            max_x - min_x + 1
        )

        output_height = int(
            max_y - min_y + 1
        )

        # an array starts at zero but the world coordinates might be negative or far away
        # shift both images by the same amount so they stay aligned with each other
        common_shift = np.array([
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # read from the right, source pixels -> world -> shared output grid
        tem_to_common = (
            common_shift
            @ tem_transform_xy
        )

        flm_to_common = (
            common_shift
            @ flm_transform_xy
        )

        # the array shape is height then width, even though our matrix uses x/y
        output_shape = (
            output_height,
            output_width,
        )

        # this is where we actually sample new pixel values
        # order=1 interpolates when the source position falls between pixels
        # same image size doesn't mean a one-to-one pixel copy
        # both outputs use the same grid, with zeros where there's no source image
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

        # put each output back into its source dtype
        # for integer images, round and clip first so casting doesn't cut off
        # fractions or wrap values
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

        # both arrays now use the same grid and shape
        # we don't return the world offset, so we keep their alignment but not the  absolute world origin
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

        # the file expects RGB, so copy a grayscale plane into three equal channels
        # we aren't using napari's colormap or contrast settings here
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

        # keep the first three channels, so RGBA loses the alpha channel
        # this format is for grayscale or RGB-style images, not a general
        # multichannel stack
        registered_tem = registered_tem[..., :3]
        registered_flm = registered_flm[..., :3]

        # TEM goes on page 0 and FLM on page 1, shape is (2, height, width, 3)
        # stacking needs one dtype, so mixed source dtypes can get promoted
        scientific_stack = np.stack(
            [
                registered_tem,
                registered_flm,
            ],
            axis=0,
        )

        # save the sampled pixels, not the whole project
        # this doesn't save landmarks, source transforms or physical calibration
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

        # check for the two RGB images our save function writes, this isn't a general TIFF loader
        if (
            stack.ndim != 4
            or stack.shape[0] != 2
            or stack.shape[-1] != 3
        ):
            raise ValueError(
                "scientific TIFF must contain two RGB registered images"
            )

        # these images already share a grid, so no extra transform is needed to
        # overlay them
        tem = stack[0]
        flm = stack[1]

        self._remove_layer_if_present(
            "Scientific TEM"
        )

        self._remove_layer_if_present(
            "Scientific Registered FLM"
        )

        # just add these as display layers
        # opening the TIFF doesn't restore role assignments, landmarks or
        # registration into the session
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

        # this time capture how the overlay looks with the layer display settings
        # it's one picture for viewing, not the two separate scientific arrays
        try:
            tem_layer = self.get_modality_image_layer(
                "TEM"
            )

            registered_flm_layer = self.viewer.layers[
                "Registered FLM"
            ]

        except (KeyError, ValueError):
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

            # use napari's figure export, not a desktop screenshot or a manual blend of the arrays
            overlay = self.viewer.export_figure(
                scale_factor=1,
                flash=False,
            )

        # put all the visibility settings back even if the capture fails
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

        # pick the writer from the file extension, uppercase extensions should work too
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
            # JPEG can't store alpha, so keep RGB only
            # PNG and TIFF get the full capture
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
        
        
                        
                            
