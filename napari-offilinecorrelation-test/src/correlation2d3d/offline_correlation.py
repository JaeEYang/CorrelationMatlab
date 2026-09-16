"""
offline_correlation.py
    =
GUI behavior
"""

from pathlib import Path

import mrcfile
import numpy as np

from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    FloatSlider,
    ComboBox,
)

from skimage import io

from qtpy.QtWidgets import (
    QSizePolicy,
    QFileDialog,
    QScrollArea,
)

from qtpy.QtCore import Qt

from correlation2d3d.session import CorrelationSession
from correlation2d3d.offline_controller import OfflineCorrelationController

from correlation2d3d.fileio.points_csv import read_points_csv

from correlation2d3d.core.transform import fit_affine, affine_xy_to_rc

from correlation2d3d.fileio.imod import (
    try_read_imod_montage,
)

from napari.layers import Points, Image
from time import perf_counter


_MULTISCALE_PIXEL_THRESHOLD = 16_000_000

def _read_image(path: Path) -> np.ndarray:
    '''Reads an image from a file path and returns it as a numpy array.
    The function supports MRC, MRCS, and ST file formats using the mrcfile library, as well as other image formats supported by skimage.io.imread.
    If the file is in MRC, MRCS, or ST format, it uses mrcfile to read the data; otherwise, it uses skimage.io.imread for other formats. '''
    
    suffix = path.suffix.lower()
    
    if suffix == ".st":
        montage = try_read_imod_montage(path)

        if montage is not None:
            return montage

    if suffix in {".mrc", ".mrcs", ".st"}:
        with mrcfile.open( str(path), permissive=True) as mrc: # we wanna open and close and keep the copy, don't effect the og file also we use permissive=True to allow reading of non-standard MRC files without raising an error.
            #mrc.print_header()
            return np.array(mrc.data,copy=True)

    return np.asarray(
        io.imread(str(path))
    )
    
    
def _build_multiscale_pyramid(image: np.ndarray) -> list[np.ndarray]:
    """ Take one large 2D image and create progressively smaller display versions for napari
    1x
    2x  downsample
    4x  downsample
    8x  downsample
    16x downsample
    
    return a list 
    [
    full_resolution,
    half_resolution,
    quarter_resolution,
    ...
    ]   
    """

    if image.ndim != 2:
        raise ValueError(
            "Multiscale display currently requires a 2D image"
        )

    pyramid = [image] # pyramid at zero is untouched scientific image

    downsample_factor = 2

    while True:
        level = image[::downsample_factor, ::downsample_factor] # keeps every 2nd, 4th, 8th row and every 2nd, 4th, 8th column, giving an image half as tall and half as wide after every loop

        pyramid.append(level)

        if max(level.shape) <= 1024: # longest side is < 1024 break.
            break

        downsample_factor *= 2

    return pyramid
    
    
    # This function creates a magicgui container widget for the offline correlation tool.
    # This function is basically our widget factory, call this function and it constructs an object for you! that is awesome gui construction worker 
    # it eventually returns a Container (QScrollArea--> this enables scrolling for smalled screenswhich is the actual gui panel containing all the controls
def make_offline_correlation_widget(viewer) -> QScrollArea:
    '''Create the offline-correlation dock widget and its session/controller state.'''

    # Every offline-correlation widget has a particular CorrelationSession associated with it.
    # the session is the memory of the current correlation job.
    session = CorrelationSession()
    
    # all the callbacks below use this same session and controller
    # defining a callback doesn't run it, connecting it tells the UI when to call it
    controller = OfflineCorrelationController(
    viewer,
    session,
    )
    
    # keep the exact source layer and callback for each role
    # we need both to disconnect the old source when the user picks another one
    _modality_transform_connections = {
        "FLM": {
            "layer": None,
            "callback": None,
        },
        "TEM": {
            "layer": None,
            "callback": None,
        },
    }
    
    # this is if the pixel data is changed
    _modality_data_connections = {
        "FLM": {
            "layer": None,
            "callback": None,
        },
        "TEM": {
            "layer": None,
            "callback": None,
        },
    }
    
    
    # loading just adds normal images to napari
    # choosing one in the image dropdown is what makes it FLM or TEM
    load_image_button = PushButton(
        text="Load Images"
    )
    
    load_csv_button = PushButton(
        text="Load CSVs"
    )
    
    # only return the layer if its actual images layer
    # will return the name and also the object itself
    # list with multiple tuples
    def _get_image_layer_choices(widget=None):
        # show the layer name in the dropdown but keep the actual layer object as its value
        # magicgui can pass the widget into this helper when it asks for choices
        return [
            (layer.name, layer)
            for layer in viewer.layers
            if isinstance(layer, Image)
        ]
                    
            
    # same as above but for points
    def _get_points_layer_choices(widget=None):
        return [
            (layer.name, layer)
            for layer in viewer.layers
            if isinstance(layer, Points)
        ]
        
    flm_image_layer_combo = ComboBox(
        label="FLM Image Layer",
        choices=_get_image_layer_choices,
        nullable=True, # user can select nothing and it will be fine.
    )

    tem_image_layer_combo = ComboBox(
        label="TEM Image Layer",
        choices=_get_image_layer_choices,
        nullable=True,
    )
        
    # create two dropdowns for each modality 
    # mullable = True , means will accept "None" as valid value, cause initially there won't be any layer
    flm_landmark_layer_combo = ComboBox(
        label="FLM Landmark Layer",
        choices=_get_points_layer_choices,
        nullable=True,
    )

    tem_landmark_layer_combo = ComboBox(
        label="TEM Landmark Layer",
        choices=_get_points_layer_choices,
        nullable=True,
    )
    

    

    flm_status = Label(
        value="FLM: not assigned"
    )

    tem_status = Label(
        value="TEM: not assigned"
    )
    
    # choosing a Points layer in the dropdown isn't enough, these buttons make it
    # active
    # the image dropdowns work differently, they assign the image as soon as we
    # choose it
    use_flm_landmarks_button = PushButton(
        text="Use FLM Landmarks"
    )

    use_tem_landmarks_button = PushButton(
        text="Use TEM Landmarks"
    )
    
    

    flm_points_status = Label(
        value="FLM landmarks: not assigned"
    )

    tem_points_status = Label(
        value="TEM landmarks: not assigned"
    )
    
    
    calculate_registration_button = PushButton(
        text="Calculate Registration"
    )

    calculate_registration_button.enabled = False # initially set it to false as no registraion done yet.

    registration_status = Label(
        value="Registration: not calculated"
    )
    
    
    # ascii-exempt: Qt widget label, rendered by the GUI and never written to stdout
    flip_flm_horizontal_button = PushButton(
        text="↔ H"
    )

    flip_flm_vertical_button = PushButton(
        text="↕ V"
    )

    flip_tem_horizontal_button = PushButton(
        text="↔ H"
    )

    flip_tem_vertical_button = PushButton(
        text="↕ V"
    )
    
    # keep a flip button looking pressed while that flip is on
    # _sync_orientation_controls makes the buttons match the saved settings
    for button in (
    flip_flm_horizontal_button,
    flip_flm_vertical_button,
    flip_tem_horizontal_button,
    flip_tem_vertical_button,
    ):
        button.native.setCheckable(True)
        button.enabled = False

    reset_flm_orientation_button = PushButton(text="Reset Orientation", enabled=False)
    reset_tem_orientation_button = PushButton(text="Reset Orientation", enabled=False)

    # keep the flip buttons small and put H/V next to each other
    flip_flm_horizontal_button.max_width = 70
    flip_flm_vertical_button.max_width = 70

    flip_tem_horizontal_button.max_width = 70
    flip_tem_vertical_button.max_width = 70

    
    flm_flip_row = Container(
        widgets=[
            flip_flm_horizontal_button,
            flip_flm_vertical_button,
        ],
        layout="horizontal",
    )

    tem_flip_row = Container(
        widgets=[
            flip_tem_horizontal_button,
            flip_tem_vertical_button,
        ],
        layout="horizontal",
    )

    
    
    # the sliders give us the angle we want and update as we drag because
    # tracking=True
    # for each value we rebuild a transform, not a padded image
    # the core decides which sign means clockwise, the slider just gives us the
    # number
    flm_rotation = FloatSlider(
        label="Rotate °",
        min=-180.0,
        max=180.0,
        step=1.0,
        value=0.0,
        readout=True,
        tracking=True
    )

    tem_rotation = FloatSlider(
        label="Rotate °",
        min=-180.0,
        max=180.0,
        step=1.0,
        value=0.0,
        readout=True,
        tracking=True
    )
    

    # keep rotation disabled until we have an image assigned for that role
    flm_rotation.enabled = False
    tem_rotation.enabled = False
    
    
    # pair and unpair use the selected points in our active landmark layers
    # not just whichever Points layer is highlighted in napari
    pair_selected_button = PushButton(
        text="Pair Selected Points"
    )

    pairing_status = Label(
        value="Pairs: none"
    )
    unpair_selected_button = PushButton(
        text="Unpair Selected"
    )
    
    pairing_buttons_row = Container(
        widgets=[
            pair_selected_button,
            unpair_selected_button,
        ],
        layout="horizontal",
    )
    
    # scientific export saves two aligned arrays, visual export saves one picture of
    # the overlay
    # opening a scientific TIFF just displays it, it doesn't load a whole project
    save_scientific_tiff_button = PushButton(
        text="Save Scientific TIFF"
    )
    open_scientific_tiff_button = PushButton(
        text="Open Scientific TIFF"
    )

    export_status = Label(
        value=""
    )
    
    
    
    save_visual_overlay_button = PushButton(
        text="Save Visual Overlay"
    )

    visual_export_status = Label(
        value="Visual export: not saved"
    )
    
    # don't let a long file path in the status label make the whole dock really wide
    for status in (
            flm_status,
            tem_status,
            flm_points_status,
            tem_points_status,
            export_status,
            visual_export_status,
        ):
            status.native.setMinimumWidth(0)
            status.native.setSizePolicy(
                QSizePolicy.Ignored,
                QSizePolicy.Preferred,
            )
    
    
    # enable the registration buttion is both flm and tem data exist in the session this gets populated in the _load_points
    def _update_registration_button() -> None:
            # this only checks that both point sets exist
            # Calculate does the checks for point counts, matching pairs and
            # non-collinear points
            calculate_registration_button.enabled = (
            session.flm.points is not None
            and session.tem.points is not None
        )
    
    # Invalidate the registion uses the method from controlled.
    def _invalidate_registration() -> None:
        controller.invalidate_registration()

        registration_status.value = (
            "Registration: not calculated"
        )
    
    
    # Display the stored settings without triggering another image update
    # # Display the modality's stored plugin orientation settings without
    # triggering another orientation update.
    # display the stored setting  when called from rebuild_modality_orientation()
    # make the controls display what the session currently stores.
    def _sync_orientation_controls(role: str) -> None:
        
        modality = controller._get_modality(role)
        if role == "FLM":
            slider = flm_rotation
            horizontal_button = flip_flm_horizontal_button
            vertical_button = flip_flm_vertical_button
        else:
            slider = tem_rotation
            horizontal_button = flip_tem_horizontal_button
            vertical_button = flip_tem_vertical_button

        # we're making the slider match the session, not asking for another rotation
        # block changed for this assignment so it doesn't call the rotation callback
        # again
        with slider.changed.blocked():
            slider.value = modality.rotation_angle
        # make the buttons match the saved flip flags
        # setChecked doesn't fire the clicked signal we use for flipping
        horizontal_button.native.setChecked(modality.horizontal_flipped)
        vertical_button.native.setChecked(modality.vertical_flipped)
        
        
        
    def _on_load_image(
        event=None,
    ) -> None:

        # let the user choose several files at once
        # the second returned value is the file filter, we don't need it here
        file_names, _ = QFileDialog.getOpenFileNames(
            None,
            "Load Images",
            "",
            (
                "Images (*.mrc *.mrcs *.st "
                "*.tif *.tiff *.png *.jpg *.jpeg);;"
                "All files (*)"
            ),
        )

        # cancel means do nothing, keep the viewer and assignments as they are
        if not file_names:
            return

        for file_name in file_names:

            path = Path(
                file_name
            )

            try:
                image = _read_image(
                    path
                )

            # if one file fails, report it and keep loading the other selected files
            except (ValueError, OSError, RuntimeError) as error:
                print(
                    f"Image load failed for {path.name}: "
                    f"{error}"
                )
                continue

            # add a normal Image layer, the inserted event updates the dropdowns
            # the user still needs to choose which one is FLM and which one is TEM
            #large 2D? -> yes -> build lightweight pyramid -> viewer.add_image(... multiscale=True)
            
            add_start = perf_counter()
            
            if (
                image.ndim == 2
                and image.size >= _MULTISCALE_PIXEL_THRESHOLD
            ):
                display_data = _build_multiscale_pyramid(
                    image
                )

                viewer.add_image(
                    display_data,
                    name=path.name,
                    multiscale=True,
                )

            else:
                viewer.add_image(
                    image,
                    name=path.name,
                )
            print(
                f"napari add_image: "
                f"{perf_counter() - add_start:.2f} s"
            )
    
    
    # Take whatever Image layer the user selected in one of the dropdowns and assign it to FLM or TEM.
    # reads combo.value, calls controller.use_image_layer(), 
    # enables the correct controls, resets orientation UI, invalidates any old registration.
    def _use_image_layer(
        combo: ComboBox,
        role: str,
        status: Label,
    ) -> None:

        layer = combo.value

        # nothing chosen, so show a prompt and leave any existing assignment alone
        if layer is None:
            status.value = (
                f"{role}: choose an Image layer"
            )
            return
        
        # this try excepts was part of experimentation i did. wanted to tranform registered flm back to og flm
        # this handles that cleanly 
        try:
            registered_flm_layer = viewer.layers[
                "Registered FLM"
            ]
        except KeyError:
            registered_flm_layer = None # i mean if it doesn't exist then we good

        # don't adopt the current result as a source that cleanup could remove
        # this check uses its current name, it isn't a permanent output tag
        if (
            registered_flm_layer is not None
            and layer is registered_flm_layer
        ):
            status.value = (
                f"{role}: Registered FLM is a generated output. "
                "Rename or duplicate it before using it as an input."
            )
            return



        try:
            controller.use_image_layer(
                role,
                layer,
            )
        # for example, the same image can't be FLM and TEM
        # show the error here, this doesn't switch the dropdown back to its old
        # choice
        except ValueError as error:
            status.value = (
                f"{role}: {error}"
            )
            return
        modality = controller._get_modality(
            role
        )
        status.value = (
            f"{role}: "
            f"{layer.name} "
            f"{tuple(modality.image.shape)} "
            f"{modality.image.dtype}"
        )

        # only enable controls once assignment worked
        # assigning an image also cleared its old landmark link, so reset that status
        # too
        if role == "FLM":
            flip_flm_horizontal_button.enabled = True
            flip_flm_vertical_button.enabled = True
            reset_flm_orientation_button.enabled = True
            flm_rotation.enabled = True
            flm_points_status.value = (
                "FLM landmarks: not assigned"
            )


        else:
            flip_tem_horizontal_button.enabled = True
            flip_tem_vertical_button.enabled = True
            reset_tem_orientation_button.enabled = True
            tem_rotation.enabled = True

            tem_points_status.value = (
                "TEM landmarks: not assigned"
            )
        
        # native flm or tem transform , landmarks follow, registration invalidated
        # make sure we follow the latest layer 
        # so assigning and image established both affine listener + data listener
        _connect_modality_transform_events(role)
        
        _connect_modality_data_events(role)

        # show the new zero/false settings without triggering another rotation
        _sync_orientation_controls(role)

        _invalidate_registration()
        _update_registration_button()
    
    # This function makes sure each role listens to affine changes 
    # on its currently assigned image and stops listening to the previous image.
    def _connect_modality_transform_events(
        role: str,
    ) -> None:

        # get the image currently assigned to this role
        image_layer = controller.get_modality_image_layer(
            role
        )

        # look up the listners previoulsy connected to this role if any
        connection = (
            _modality_transform_connections[role]
        )

        old_layer = connection["layer"]
        old_callback = connection["callback"]

        # already listening to this layer, so don't connect the same callback twice
        if (
            old_layer is image_layer
            and old_callback is not None
        ):
            return

        # if the different image was previously assigned remove the listner
        if (
            old_layer is not None
            and old_callback is not None
        ):
            # stop listening to the old source, otherwise moving it would still
            # affect this role
            old_layer.events.affine.disconnect(
                old_callback
            )

        # remember the role now, napari gives us the event later
        # save this exact callback so we can disconnect it next time
        callback = (
            lambda event, role=role:
            _on_modality_transform_changed(role)
        )

        # this listens to affine changes for FLM and TEM
        # it doesn't listen to every separate transform setting or to changes in the
        # pixel data
        image_layer.events.affine.connect(
            callback
        )

        connection["layer"] = image_layer
        connection["callback"] = callback
    
    #React when user rough-aligns FLM/TEM with napari's Transform tool. 
    # Internally? Move attached landmarks and invalidate any precise registration because the source geometry changed
    # Rough alignment changed the coordinate system, so anything derived from the old geometry is stale.
    def _on_modality_transform_changed(
        role: str,
    ) -> None:

        controller.sync_landmarks_to_modality_transform(
            role
        )

        _invalidate_registration()
    
    # this reacts when napari reports that the pixel data of  an assigned source image changed
    # takes input role
    #It asks the controller to synchronize the pixels. If the grid changed, it updates the landmark status and orientation controls. In every case, it invalidates the previous registration.
    def _on_modality_data_changed(
        role: str,
    ) -> None:

        grid_changed = controller.sync_modality_image_data(role)
    
        # a different array shape was treated as a fresh source assignment
        # so the old landmark association no longer belongs to this source grid
        if grid_changed:
            image_layer = controller.get_modality_image_layer(role)
            modality = controller._get_modality(
                role
            )
        
            if role == "FLM":
                flm_status.value = (
                    f"FLM: {image_layer.name} "
                    f"{tuple(modality.image.shape)}"
                )

                flm_points_status.value = (
                    "FLM landmarks: not assigned"
                )

            else:
                tem_status.value = (
                    f"TEM: {image_layer.name} "
                    f"{tuple(modality.image.shape)}"
                )

                tem_points_status.value = (
                    "TEM landmarks: not assigned"
                )

            _sync_orientation_controls(role)

        _invalidate_registration()
        _update_registration_button()
                
    
    # connects the each FLM/TEM roles to the data event of its currently assigned image layer  
    # diconnects previous source's data callback, connects the new source, 
    # and rememebers the exact callback so it can later be disconnected cleanly.
    # basically same as the _connect_modality_transform_events() but for pixel data. 
    def _connect_modality_data_events(
        role: str,
    ) -> None:

        image_layer = controller.get_modality_image_layer(role)

        connection = _modality_data_connections[role]
        

        old_layer = connection["layer"]
        old_callback = connection["callback"]

        if (
            old_layer is image_layer
            and old_callback is not None
        ):
            return

        if (
            old_layer is not None
            and old_callback is not None
        ):
            old_layer.events.data.disconnect(
                old_callback
            )

        callback = (
            lambda event, role=role:
            _on_modality_data_changed(role)
        )

        image_layer.events.data.connect(
            callback
        )

        connection["layer"] = image_layer
        connection["callback"] = callback            

          
    #Figure out how to perform a horizontal flip.
    # All three function below are just wrappers now. real work in controller
    def _flip_modality_horizontal(role: str) -> None:
        controller.flip_modality_horizontal(
            role
        )

        # the controller moves things, this wrapper clears the old fit and updates
        # the controls
        # vertical flip, reset and rotation follow the same order
        _invalidate_registration()
        _sync_orientation_controls(role)
        
    
    def _flip_modality_vertical(role: str) -> None:
        controller.flip_modality_vertical(
            role
        )

        _invalidate_registration()
        _sync_orientation_controls(role)
        
    def _reset_modality_orientation(role: str) -> None:
        controller.reset_modality_orientation(
            role
        )

        _invalidate_registration()
        _sync_orientation_controls(role)
                    
     # connect the button to the callback.   
    def _on_flip_flm_horizontal(event=None):
        _flip_modality_horizontal("FLM")


    def _on_flip_tem_horizontal(event=None):
        _flip_modality_horizontal("TEM")

    def _on_flip_flm_vertical(event=None):
        _flip_modality_vertical("FLM")


    def _on_flip_tem_vertical(event=None):
        _flip_modality_vertical("TEM")
    
    # this is essntially our rotation callback
    def _set_modality_rotation(
        role: str,
        angle_degrees: float,
    ) -> None:
        # send the selected angle to the controller, don't add it to the previous
        # angle
        controller.set_modality_rotation(
            role,
            angle_degrees,
        )

        _invalidate_registration()
        _sync_orientation_controls(role)
        
    def _on_flm_rotation_change(event=None):
        # read the value from the slider instead of relying on what the signal sends
        _set_modality_rotation(
            "FLM",
            float(flm_rotation.value),
        )


    def _on_tem_rotation_change(event=None):
        # same thing for TEM, use its slider value and role
        _set_modality_rotation(
            "TEM",
            float(tem_rotation.value),
        )
    
   
    """
    csv files 
    read_points_csv
    Points2D in original source coordinates 
    generic hidden napari Points candidate
    user chooses FLM or TEM landmark role 
    use_points_layer
    correct source tranform is finally applied
    """
    def _on_load_csv(
        event=None,
    ) -> None:

        # this allows us to open multiple files at the same time
        file_names, _ = QFileDialog.getOpenFileNames(
            None,
            "Load Landmark CSVs",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        # cancel leaves the existing candidates and assignments alone
        if not file_names:
            return

        # each CSV is its own candidate, we don't pick FLM or TEM here
        for file_name in file_names:

            path = Path(
                file_name
            )

            try:
                points = read_points_csv(  # returns Points2D object 
                    path
                )

                # keep source coordinates until Use Landmarks picks an image
                controller.create_points_layer_from_original_points(
                    points,
                    name=path.stem,
                )

            except (ValueError, OSError) as error:
                # one bad file shouldn't stop the other selected files loading
                print(
                    f"CSV load failed for {path.name}: "
                    f"{error}"
                )
                continue
    
    
    
    def _use_landmark_layer(
        combo: ComboBox,
        role: str,
        status: Label,
    ) -> None:

        layer = combo.value # get what the dropdown contains

        # if nothing is selected
        if layer is None:
            status.value = (
                f"{role} landmarks: choose a Points layer first"
            )
            return
        
        # the blue registered points are a result, not a source landmark set
        # they still appear in the dropdown, so check before assigning them
        try:
            registered_landmarks_layer = viewer.layers[
                "FLM Landmarks Registered to TEM"
            ]
        except KeyError:
            registered_landmarks_layer = None

        if (
            registered_landmarks_layer is not None
            and layer is registered_landmarks_layer
        ):
            status.value = (
                f"{role} landmarks: registered landmarks are a generated output. "
                "Rename or duplicate them before using them as input."
            )
            return
        
        # where the actual coordinate logic happens.
        # use FLM landmarks FLM session populated simillary for others aswell.,
        try:
            controller.use_points_layer(
                role,
                layer,
            )
        except ValueError as error:
            status.value = (
                f"{role} landmarks: {error}"
            )
            return

        status.value = (
            f"{role} landmarks: "
            f"{layer.name} "
            f"({len(layer.data)} points)"
        )

        _invalidate_registration() # invalidate as the points have changed 
        _update_registration_button() # recheck if registration is now possible
        
    # wrapped for individual modalites
    def _on_use_flm_landmarks(event=None):
        _use_landmark_layer(
            flm_landmark_layer_combo,
            "FLM",
            flm_points_status,
        )


    def _on_use_tem_landmarks(event=None):
        _use_landmark_layer(
            tem_landmark_layer_combo,
            "TEM",
            tem_points_status,
        )
            
    def _on_points_mode_changed(
        layer,
        event=None,
    ) -> None:

        # we only need this when the user switches to adding points
        if layer.mode != "add":
            return

        if "pair_id" not in layer.features:
            return

        # new points should start unpaired, not copy the last selected point's pair ID
        # this sets the default for new points, it doesn't change existing labels
        layer.feature_defaults[
            "pair_id"
        ] = ""
        
            
            
    def _on_points_data_changed(
        layer,
        event=None,
    ) -> None:

        # all Points layers have this listener, but only our active landmarks matter here
        # ignore changes to unrelated points or generated result points
        active_roles = [
            role
            for role in ("FLM", "TEM")
            if layer is controller._landmark_layers[role]
        ]

        if not active_roles:
            return

        # read the latest point positions and work out their source-image coordinates
        # this also runs when our own sync code sets layer.data, not just when the
        # user edits points
        for role in active_roles:
            controller.use_points_layer(
                role,
                layer,
            )

            if role == "FLM":
                    flm_points_status.value = (
                        f"FLM landmarks: "
                        f"{layer.name} "
                        f"({len(layer.data)} points)"
                    )

            else:
                tem_points_status.value = (
                    f"TEM landmarks: "
                    f"{layer.name} "
                    f"({len(layer.data)} points)"
                )
        # if a point was deleted, its partner might still have the pair label
        # clear that leftover label so it doesn't look like a complete pair
        controller.clear_orphaned_pairs()

        # the input points changed, so the old fit needs to go
        _invalidate_registration()
        _update_registration_button()
    
    
    
    # this is just a wrapper
    """
    BUTTON
    _on_pair_landmarks() 
        asks
    controller.pair_selected_landmarks()
        returns 9
    _on_pair_landmarks()
    "Pairs: 9 paired landmarks"
    """
    
    def _on_pair_selected_landmarks(event=None):

        try:
            pair_id = (
                controller.pair_selected_landmarks()
            )

        except ValueError as error:
            # show why pairing failed, don't make it look like we created a new pair
            pairing_status.value = (
                f"Pairs: {error}"
            )
            return

        pairing_status.value = (
            f"Created pair {pair_id}"
        )

        # pair IDs decide which points match, so changing them means the old fit is no longer valid
        # even if none of the points actually moved
        _invalidate_registration()
    
    def _on_unpair_selected_landmarks(
        event=None,
    ):

        # the controller clears the pair ID from both ends, it doesn't delete the  points
        # if the selection is wrong, show the error and return without clearing the fit
        try:
            pair_id = (
                controller.unpair_selected_landmarks()
            )

        except ValueError as error:
            pairing_status.value = (
                f"Pairs: {error}"
            )
            return

        pairing_status.value = (
            f"Removed pair {pair_id}"
        )

        # the point matching changed, so Calculate needs to run again
        _invalidate_registration()

    
    
    # the actual call back function when registration clicked on
    # this creates the tranformed layer basically.
    def _on_calculate_registration(event=None):
        # clear the old result before trying again
        # if the new fit fails, don't leave the old Registered FLM looking like a new success
        _invalidate_registration()
        if (
            session.flm.points is None
            or session.tem.points is None
        ):
            registration_status.value = (
                "Registration: load both landmark sets first"
            )
            return

        # get the latest active points and match their IDs, or use row order if neither side has IDs
        # then fit_affine checks if the points have enough geometry for a fit
        try:
            (
                registration_flm_points,
                registration_tem_points,
            ) = controller.get_registration_landmarks()

            registration = fit_affine(
                registration_flm_points,
                registration_tem_points,
            )

        except ValueError as error:
            registration_status.value = (
                f"Registration failed: {error}"
            )
            return

        # only save the registration after the fit worked, the error paths above
        # leave it cleared
        session.registration = registration

        # show where the fitted FLM points land using a separate blue result layer
        # leave the source landmarks where they were
        predicted = registration.apply(
            registration_flm_points
        )

        transformed_layer_name = (
            "FLM Landmarks Registered to TEM"
        )

        transformed_rc = predicted.to_rc()

        try:
            layer = viewer.layers[
                transformed_layer_name
            ]
        except KeyError:
            viewer.add_points(
            transformed_rc,
            name=transformed_layer_name,
            size=50,
            face_color="#00007f",
        )
        else:
            layer.data = transformed_rc
            layer.size = 50
            layer.face_color = "#00007f"

        registration_status.value = (
            f"Registration RMSE: "
            f"{registration.rmse:.3f} TEM pixels"
        )
        
        
        if (
            session.flm.image is not None
            and session.tem.image is not None
        ):
            flm_transform_xy = (
                controller.get_modality_transform_xy(
                    "FLM"
                )
            )

            # read from the right, FLM source pixels -> current FLM world positions
            # -> TEM world positions
            # the fit alone doesn't start from the source pixels, so we need both transforms
            registered_affine_xy = (
                registration.matrix
                @ flm_transform_xy
            )

            registered_affine_rc = affine_xy_to_rc(
                registered_affine_xy
            )

            controller._remove_layer_if_present(
                "Registered FLM"
            )

            # reuse the source pixels and give this new layer the combined transform
            # we aren't making a warped export array here
            # the user can adjust Registered FLM later and export reads that updated placement
            viewer.add_image(
                session.flm.image,
                name="Registered FLM",
                affine=registered_affine_rc,
                opacity=0.5,
                blending="translucent",
            )
        
    
    
    def _on_save_scientific_tiff(event=None):

        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Scientific TIFF",
            "registered_scientific.tif",
            "TIFF files (*.tif *.tiff)",
        )

        # cancel without exporting anything or changing the status
        if not path:
            return

        # add the TIFF extension if the filename doesn't already have it
        if not path.lower().endswith(
            (".tif", ".tiff")
        ):
            path += ".tif"

        try:
            # let the controller handle the grid, sampling and TIFF format
            # this wrapper just asks where to save and reports what happened
            controller.save_scientific_tiff(
                path
            )

        except ValueError as error:
            export_status.value = (
                f"Export failed: {error}"
            )
            return

        export_status.value = (
            f"Saved: {path}"
        )
    
    # This GUI callback's only job is to ask the user which TIFF they want to open and pass that path to the controller.
    def _on_open_scientific_tiff(
        event=None,
    ):
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Scientific TIFF",
            "",
            "TIFF files (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            controller.open_scientific_tiff(path)
        except (ValueError, OSError) as error:
            export_status.value = (f"Open failed: {error}")
            return

        export_status.value = (f"Opened: {path}")
    
    def _on_save_visual_overlay(event=None):

        # keep the chosen format in case the filename doesn't have a known extension
        path, selected_filter = (
            QFileDialog.getSaveFileName(
                None,
                "Save Visual Overlay",
                "registered_overlay.tif",
                (
                    "TIFF (*.tif *.tiff);;"
                    "PNG (*.png);;"
                    "JPEG (*.jpg *.jpeg)"
                ),
            )
        )

        # cancel before we change any layer visibility for the capture
        if not path:
            return

        lower_path = path.lower()

        # if the filename already has a known extension, use that
        # otherwise go with the format selected in the dialog
        if not lower_path.endswith(
            (
                ".tif",
                ".tiff",
                ".png",
                ".jpg",
                ".jpeg",
            )
        ):
            if selected_filter.startswith(
                "PNG"
            ):
                path += ".png"

            elif selected_filter.startswith(
                "JPEG"
            ):
                path += ".jpg"

            else:
                path += ".tif"

        try:
            # let the controller capture and save, it puts visibility back even if capture fails
            # only show success after the save call finishes
            controller.save_visual_overlay(
                path
            )

        except ValueError as error:
            visual_export_status.value = (
                f"Visual export failed: {error}"
            )
            return

        visual_export_status.value = (
            f"Saved: {path}"
        )
    
    # want to inspect the layer that napri says was removed
    # Keep the image dropdowns synchronized with the layers that currently exist in napari, 
    # and clean up our FLM/TEM state if an assigned source image gets deleted.
    def _refresh_image_layer_choices(
        event=None,
    ) -> None:

        # a removal event gives us the removed layer in value  rename and insertion refreshes call this without an event, so there might be no removed layer . 
        # If event has an attribute called value, give it to me. Otherwise give me None.
        removed_layer = getattr(
            event,
            "value",
            None,
        )

        # keep track of whether we cleared a role
        # refreshing the dropdown and losing an assigned image aren't the same thing
        active_image_removed = False
        
        # if the exact layer object that napari removed is the same exact object we assigned as FLM ? 
        if (
            removed_layer
            is controller._image_layers["FLM"]
        ):
            # if this was our assigned image, disconnect its listener before clearing the role
            # compare the actual objects since names can change or look similar
            
            connection = (
                _modality_transform_connections["FLM"]
            )

            if (
                connection["layer"] is removed_layer
                and connection["callback"] is not None
            ):  
                # if the layer is being deleted we no longer want the listener
                removed_layer.events.affine.disconnect(
                    connection["callback"]
                )
                connection["layer"] = None
                connection["callback"] = None
                
            data_connection = (
                _modality_data_connections["FLM"]
            )

            if (
                data_connection["layer"] is removed_layer
                and data_connection["callback"] is not None
            ):
                removed_layer.events.data.disconnect(
                    data_connection["callback"]
                )

                data_connection["layer"] = None
                data_connection["callback"] = None    
            
            # this unassignes the roles completly. 
            controller.clear_image_layer(
                "FLM"
            )

            flm_status.value = (
                "FLM: not assigned"
            )

            flm_points_status.value = (
                "FLM landmarks: not assigned"
            )

            # no image now, so disable orientation and chnage the flag active layer removed to true
            flip_flm_horizontal_button.enabled = False
            flip_flm_vertical_button.enabled = False
            reset_flm_orientation_button.enabled = False
            flm_rotation.enabled = False

            active_image_removed = True


        if (
            removed_layer
            is controller._image_layers["TEM"]
        ):
            connection = (
                _modality_transform_connections["TEM"]
            )

            if (
                connection["layer"] is removed_layer
                and connection["callback"] is not None
            ):
                removed_layer.events.affine.disconnect(
                    connection["callback"]
                )
                connection["layer"] = None
                connection["callback"] = None

            data_connection = (
                _modality_data_connections["TEM"]
            )

            if (
                data_connection["layer"] is removed_layer
                and data_connection["callback"] is not None
            ):
                removed_layer.events.data.disconnect(
                    data_connection["callback"]
                )

                data_connection["layer"] = None
                data_connection["callback"] = None
            
            controller.clear_image_layer(
                "TEM"
            )

            # clear TEM's labels and controls too, leave the FLM assignment alone
            tem_status.value = (
                "TEM: not assigned"
            )

            tem_points_status.value = (
                "TEM landmarks: not assigned"
            )

            flip_tem_horizontal_button.enabled = False
            flip_tem_vertical_button.enabled = False
            reset_tem_orientation_button.enabled = False
            tem_rotation.enabled = False


            active_image_removed = True

        # get the dropdown choices again from the Image layers currently in napari
        flm_image_layer_combo.reset_choices()
        tem_image_layer_combo.reset_choices()

        # clear any result that used the old source, then check if Calculate should
        # be enabled
        if active_image_removed:
            _invalidate_registration()
            _update_registration_button()
    

    # need to tell the ComboBox: Something changed. Calculate your choices again.   
    """napari says:
    "this layer was removed"
    was it our active FLM/TEM landmark layer? if yes  
    clear controller reference
    clear session coordinates
    invalidate registration
    disable Calculate if appropriate
    """
    def _refresh_points_layer_choices(event=None):

        #  dynamically retrieve the value of an object's (event) attribute using its string name (value)
        removed_layer = getattr(
            event,
            "value",
            None,
        )

        active_landmark_removed = False

        if (
            removed_layer
            is controller._landmark_layers["FLM"]
        ):
            # clear both versions of the points so we don't keep using a deleted landmark layer
            controller._landmark_layers["FLM"] = None

            session.flm.points = None
            session.flm.original_points = None

            flm_points_status.value = (
                "FLM landmarks: not assigned"
            )

            active_landmark_removed = True

        if (
            removed_layer
            is controller._landmark_layers["TEM"]
        ):
            controller._landmark_layers["TEM"] = None

            session.tem.points = None
            session.tem.original_points = None

            tem_points_status.value = (
                "TEM landmarks: not assigned"
            )

            active_landmark_removed = True

        # update the dropdown names and objects even if the removed layer wasn't one we were using
        flm_landmark_layer_combo.reset_choices()
        tem_landmark_layer_combo.reset_choices()

        if active_landmark_removed:
            _invalidate_registration()
            _update_registration_button()


    def _on_points_layer_renamed(event=None):
        # just update the names in the menu, we still remember the same layer object
        _refresh_points_layer_choices()

    # layer list inserted callback
    def _on_layer_inserted(event):
        layer = event.value # give me the actual layer object that was just added

        if isinstance(layer, Points): # was the thing that was just inserted a Points layer?
            layer.events.name.connect(
                _on_points_layer_renamed
            )
            # layer=layer makes each lambda remember the right layer for later
            # without it, the loop below could end up using the last layer for every callback
            layer.events.mode.connect(
                lambda event, layer=layer:
                    _on_points_mode_changed(
                        layer,
                        event,
                    )
            )
            # listen for points moving, being added or being deleted
            # if these are active landmarks we need to update the session and clear the old fit
            layer.events.data.connect(
                lambda event, layer=layer:
                    _on_points_data_changed(
                        layer,
                        event,
                    )
            )
        # keep track of name changes for all image choices
        # _connect_modality_transform_events only adds the affine listener once we
        # assign a role. this basically setsup the listener for name change
        if isinstance(layer, Image):
            layer.events.name.connect(
                lambda event:
                    _refresh_image_layer_choices()
            )
            
        _refresh_points_layer_choices()
        _refresh_image_layer_choices()


    # there might already be layers when we open the plugin, so connect those too
    # the inserted event only tells us about layers added after this
    for layer in viewer.layers:
        if isinstance(layer, Points):
            layer.events.name.connect(
                _on_points_layer_renamed
            )
            layer.events.mode.connect(
                lambda event, layer=layer:
                    _on_points_mode_changed(
                        layer,
                        event,
                    )
            )
            
            layer.events.data.connect(
                lambda event, layer=layer:
                    _on_points_data_changed(
                        layer,
                        event,
                    )
            )
            
        if isinstance(layer, Image):
            layer.events.name.connect(
                lambda event:
                    _refresh_image_layer_choices()
            )

    # keep both dropdowns updated when layers get added or removed
    viewer.layers.events.removed.connect(
        _refresh_image_layer_choices
    )
    viewer.layers.events.inserted.connect(
        _on_layer_inserted
    )

    viewer.layers.events.removed.connect(
        _refresh_points_layer_choices
    )
        
        
    # these lambdas wait for a selection change, they don't run while we build the
    # widget
    # they pass the right role and controls, then the helper reads combo.value
    flm_image_layer_combo.changed.connect(
    lambda event=None:
        _use_image_layer(
            flm_image_layer_combo,
            "FLM",
            flm_status,
        )
    )

    tem_image_layer_combo.changed.connect(
        lambda event=None:
            _use_image_layer(
                tem_image_layer_combo,
                "TEM",
                tem_status,
            )
    )
    
    load_image_button.clicked.connect(
        _on_load_image
    )
    
    load_csv_button.clicked.connect(
        _on_load_csv
    )
           
    
    # registers that function as a listener. Then later click calls it
    # connects the click to function object
    use_flm_landmarks_button.clicked.connect(
        _on_use_flm_landmarks
    )

    use_tem_landmarks_button.clicked.connect(
        _on_use_tem_landmarks
    )  

        
    calculate_registration_button.clicked.connect(
        _on_calculate_registration
    )
        
        
    # connect needs a function to call later, not _reset_modality_orientation("FLM") running now
    # the lambda waits for the click, ignores the event and passes in the role
    reset_flm_orientation_button.clicked.connect(
        lambda event=None: _reset_modality_orientation("FLM")
    )
    reset_tem_orientation_button.clicked.connect(
        lambda event=None: _reset_modality_orientation("TEM")
    )

    flip_flm_horizontal_button.clicked.connect(
        _on_flip_flm_horizontal
    )

    flip_flm_vertical_button.clicked.connect(
        _on_flip_flm_vertical
    )

    flip_tem_horizontal_button.clicked.connect(
        _on_flip_tem_horizontal
    )

    flip_tem_vertical_button.clicked.connect(
        _on_flip_tem_vertical
    )
        
    flm_rotation.changed.connect(
        _on_flm_rotation_change
    )

    tem_rotation.changed.connect(
        _on_tem_rotation_change
    )
        
    
    pair_selected_button.clicked.connect(
        _on_pair_selected_landmarks
    )
        
    unpair_selected_button.clicked.connect(
        _on_unpair_selected_landmarks
    )


    save_scientific_tiff_button.clicked.connect(
        _on_save_scientific_tiff
    )


    save_visual_overlay_button.clicked.connect(
        _on_save_visual_overlay
    )


    open_scientific_tiff_button.clicked.connect(
        _on_open_scientific_tiff
    )
    
    
                
    # above we created the controls and connected their callbacks
    # this list just puts them in top-to-bottom order, it doesn't run the callbacks
    # in that order
    content = Container(
        widgets=[
            Label(
                value="Offline Correlation"
            ),
            # load general image, csv
            load_image_button,
            load_csv_button,

            # FLM
            flm_image_layer_combo,
            flm_status,
            flm_flip_row,
            flm_rotation,
            reset_flm_orientation_button,
            flm_landmark_layer_combo,
            use_flm_landmarks_button,
            flm_points_status,

            # TEM
            tem_image_layer_combo,
            tem_status,
            tem_flip_row,
            tem_rotation,
            reset_tem_orientation_button,
            tem_landmark_layer_combo,
            use_tem_landmarks_button,
            tem_points_status,

            pairing_buttons_row,
            pairing_status,

            # Registration / warp
            calculate_registration_button,
            registration_status,

            save_scientific_tiff_button,
            open_scientific_tiff_button,
            export_status,

            save_visual_overlay_button,
            visual_export_status,
        ]
    )

    # put the magicgui controls inside a Qt scroll area so the long form fits in a small dock
    scroll_area = QScrollArea()

    #Resize the plugin content to the width of the dock instead of letting it preserve some enormous preferred width.
    scroll_area.setWidgetResizable(
        True
    )
    #This plugin should fit horizontally. Do not let the user wander left/right through a giant form.
    scroll_area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )

    # only show the vertical scrollbar when the dock is too short for the content
    scroll_area.setVerticalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )

    content.native.setMinimumWidth(
        0
    )

    scroll_area.setWidget(
        content.native
    )

    # keep the Python magicgui object alive along with its Qt widget
    # we return the outer scroll area, the actual controls are inside it
    scroll_area._magicgui_content = content

    return scroll_area
        

"""
Offline correlation data flow
=============================

The offline-correlation separates three different kinds of state:

1. Files on disk
2. Computational state in CorrelationSession
3. Interactive visualization state in napari


IMAGE LOADING
-------------

_read_image()

    file on disk
        ↓
    NumPy array

The generic Load Images button only loads image data into napari.

For example:

    image_01.tif
    image_02.st
        ↓
    _read_image()
        ↓
    NumPy arrays
        ↓
    napari Image layers

At this point the images are not yet FLM or TEM.


ROLE ASSIGNMENT
---------------

The user chooses an existing napari Image layer from:

    FLM Image Layer
    TEM Image Layer

Assigning a layer gives that layer a semantic role in the correlation
workflow.

For example:

    napari Image layer A
        ↓
    assigned as FLM

    napari Image layer B
        ↓
    assigned as TEM

The controller stores the actual layer objects:

    controller._image_layers["FLM"]
    controller._image_layers["TEM"]

The role therefore does not depend on the napari layer name.

SESSION STATE
-------------

When an Image layer is assigned to FLM or TEM, its pixel data is copied
into the corresponding modality state:

    session.flm.image
    session.tem.image

The session stores computational information needed by the correlation
workflow, including:

    image data
    orientation settings
    original landmark coordinates
    current landmark coordinates
    registration result

The session is computational state.

It is not responsible for drawing the image on screen.


NAPARI STATE
------------

napari owns the interactive layers shown to the user.

An assigned source image can have geometry such as:

    translation
    rotation
    scale
    affine transformation

The controller asks the actual napari layer where its pixels currently
map in world coordinates.

So the complete source transform is obtained from the current Image
layer rather than assuming that the image is still at its original
position.


LANDMARKS
---------

Landmark Points layers are also ordinary napari layers.

When a Points layer is assigned as FLM or TEM landmarks, the controller
stores its relationship to the corresponding source image.

Original landmark coordinates are preserved in source-image pixel
coordinates.

Their displayed positions are reconstructed using the current source
image transform:

    original image coordinates
        ↓
    current source transform
        ↓
    napari world coordinates

This allows landmarks to remain attached to structures when the source
image is roughly translated, rotated, or scaled.


REGISTRATION
------------

Registration uses corresponding FLM and TEM landmarks in their current
working coordinates.

    FLM landmarks
        ↓
    affine registration
        ↓
    TEM landmark coordinates

The resulting registration is used to create:

    Registered FLM

Registered FLM is a separate napari Image layer.

It represents the registration result and can be manually refined
without modifying the original FLM source layer.


SOURCE CHANGES AFTER REGISTRATION
---------------------------------

FLM and TEM source layers are inputs to registration.

If either source geometry changes after registration:

    source transform changes
        ↓
    landmarks follow the source
        ↓
    old registration is no longer valid
        ↓
    registration is invalidated

Manual refinement should instead be performed on:

    Registered FLM


EXPORT
------

Scientific TIFF export uses the actual current geometry of:

    assigned TEM source
    Registered FLM

Both are rasterized onto a common output grid.

Visual Overlay export captures their rendered appearance in napari.


OVERALL FLOW
------------

files on disk
    ↓
_read_image()
    ↓
napari Image layers
    ↓
assign FLM / TEM roles
    ↓
CorrelationSession + controller track semantic state
    ↓
rough source alignment
    ↓
assign / edit landmarks
    ↓
pair corresponding landmarks
    ↓
calculate affine registration
    ↓
Registered FLM
    ↓
optional manual refinement
    ↓
scientific TIFF / visual overlay


The important separation is:

    CorrelationSession
        stores computational correlation state

    OfflineCorrelationController
        coordinates application behavior

    napari
        owns interactive layers and their current display geometry

The controller connects these pieces without requiring source layers
to have fixed names such as "FLM" or "TEM".


Use a named callback when  body needs explanation, multiple steps, or debugging breakpoints. 
A lambda is reasonable for a single obvious argument-binding step.
"""
