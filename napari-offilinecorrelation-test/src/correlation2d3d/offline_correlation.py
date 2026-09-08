"""
offline_correlation.py
    =
GUI behavior
"""

from pathlib import Path

import mrcfile
import numpy as np
from magicgui.widgets import Container, FileEdit, Label, PushButton, FloatSlider # This is a magicgui container that will hold the widgets for the offline correlation tool. 
#It includes a label, file edit widgets for loading images and points, and push buttons for performing actions such as warping images and computing registrations.
from skimage import io

from correlation2d3d.session import CorrelationSession
from correlation2d3d.offline_controller import OfflineCorrelationController

from correlation2d3d.fileio.points_csv import read_points_csv

from correlation2d3d.core.warp import warp_image
from correlation2d3d.core.transform import fit_affine, affine_xy_to_rc
from qtpy.QtWidgets import QSlider, QDoubleSpinBox


'''user clicks button
      ↓
Qt emits clicked event
      ↓
magicgui receives it
      ↓
_on_load_flm()
      ↓
_load_image()
      ↓
_read_image()
      ↓
session + napari updated

similar execution for other things aswell.

'''


def _read_image(path: Path) -> np.ndarray:
    '''Reads an image from a file path and returns it as a numpy array.
    The function supports MRC, MRCS, and ST file formats using the mrcfile library, as well as other image formats supported by skimage.io.imread.
    If the file is in MRC, MRCS, or ST format, it uses mrcfile to read the data; otherwise, it uses skimage.io.imread for other formats. '''
    
    suffix = path.suffix.lower()

    if suffix in {".mrc", ".mrcs", ".st"}:
        with mrcfile.open( str(path), permissive=True) as mrc: # we wanna open and close and keep the copy, don't effect the original file also we use permissive=True to allow reading of non-standard MRC files without raising an error.
            return np.array(mrc.data,copy=True)

    return np.asarray(
        io.imread(str(path))
    )
    
    # This function creates a magicgui container widget for the offline correlation tool.
    # This function is basically our widget factory, call this function and it constructs an object for you! that is awesome
    # gui construction worker 
    # it It eventually returns a Container which is the actual gui panel containing all the controls
def make_offline_correlation_widget(viewer) -> Container:
    '''Creates a magicgui container widget for the offline correlation tool.
    The widget includes file edit widgets for loading FLM and TEM images, push buttons for loading the images, and labels to display the status of the loaded images ...
    The function initializes a CorrelationSession object to store the state of the correlation session, including the loaded images, points, and registration information.'''
    
    
    # Every offline-correlation widget has a particular CorrelationSession associated with it.
    # he session is the memory of the current correlation job.
    session = CorrelationSession()
    
    controller = OfflineCorrelationController(
    viewer,
    session,
    )
    
    flm_file = FileEdit(
        label="FLM Image",
        mode="r",
        filter="*.mrc *.mrcs *.st *.tif *.tiff *.png *.jpg *.jpeg",
    )

    tem_file = FileEdit(
        label="TEM Image",
        mode="r",
        filter="*.mrc *.mrcs *.st *.tif *.tiff *.png *.jpg *.jpeg",
    )

    load_flm_button = PushButton(
        text="Load FLM"
    )

    load_tem_button = PushButton(
        text="Load TEM"
    )

    flm_status = Label(
        value="FLM: not loaded"
    )

    tem_status = Label(
        value="TEM: not loaded"
    )
    
    flm_points_file = FileEdit(
    label="FLM Landmarks",
    mode="r",
    filter="*.csv",
    )

    tem_points_file = FileEdit(
        label="TEM Landmarks",
        mode="r",
        filter="*.csv",
    )

    load_flm_points_button = PushButton(
        text="Load FLM Landmarks"
    )

    load_tem_points_button = PushButton(
        text="Load TEM Landmarks"
    )

    flm_points_status = Label(
        value="FLM landmarks: not loaded"
    )

    tem_points_status = Label(
        value="TEM landmarks: not loaded"
    )
    
    calculate_registration_button = PushButton(
        text="Calculate Registration"
    )

    calculate_registration_button.enabled = False # initially set it to false as no registraion done yet.

    registration_status = Label(
        value="Registration: not calculated"
    )
    
    warp_button = PushButton(
        text="Warp FLM to TEM"
    )

    warp_button.enabled = False

    warp_status = Label(
        value="Warp: not calculated"
    )
    
    warped_opacity = FloatSlider(
        label = "Warped FLM Opacity",
        min = 0.0,
        max = 1.0,
        step = 0.05,
        value = 0.5
    )
    warped_opacity.enabled = False
    
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

    # Checked buttons show which display-axis flips are currently selected.
    for button in (
        flip_flm_horizontal_button,
        flip_flm_vertical_button,
        flip_tem_horizontal_button,
        flip_tem_vertical_button,
    ):
        button.native.setCheckable(True)
        button.tooltip = "Toggle a flip along the displayed image axes, after rotation."

    reset_flm_orientation_button = PushButton(text="Reset Orientation", enabled=False)
    reset_tem_orientation_button = PushButton(text="Reset Orientation", enabled=False)
    for button in (reset_flm_orientation_button, reset_tem_orientation_button):
        button.tooltip = "Clear rotation and both flips; keep the loaded image and landmarks."

    flip_flm_horizontal_button.max_width = 70
    flip_flm_vertical_button.max_width = 70

    flip_tem_horizontal_button.max_width = 70
    flip_tem_vertical_button.max_width = 70

    flip_flm_horizontal_button.enabled = False
    flip_flm_vertical_button.enabled = False

    flip_tem_horizontal_button.enabled = False
    flip_tem_vertical_button.enabled = False
    
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

    flip_flm_horizontal_button.enabled = False
    flip_tem_horizontal_button.enabled = False
    
    flm_rotation = FloatSlider(
        label="Rotate °",
        min=-180.0,
        max=180.0,
        step=1.0,
        value=0.0,
        readout=True,
        tracking=False
    )

    tem_rotation = FloatSlider(
        label="Rotate °",
        min=-180.0,
        max=180.0,
        step=1.0,
        value=0.0,
        readout=True,
        tracking=False
    )
    
    flm_qslider = flm_rotation.native.findChild(
    QSlider
    )

    flm_rotation_readout = flm_rotation.native.findChild(
        QDoubleSpinBox
    )

    tem_qslider = tem_rotation.native.findChild(
        QSlider
    )

    tem_rotation_readout = tem_rotation.native.findChild(
        QDoubleSpinBox
    )

    flm_rotation.enabled = False
    tem_rotation.enabled = False
    for slider in (flm_rotation, tem_rotation):
        slider.tooltip = "Rotation from the original padded image, before selected display-axis flips."
     
    
    # a small helper to decide if warping is possible, do we have the images and the registration matrix.
    def _update_warp_button() -> None:
        warp_button.enabled = (
            session.flm.image is not None
            and session.tem.image is not None
            and session.registration is not None
        )
    
    # enable the registration buttion is both flm and tem data exist in the session this gets populated in the _load_points
    def _update_registration_button() -> None:
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

        warp_status.value = (
            "Warp: not calculated"
        )

        warp_button.enabled = False
        warped_opacity.enabled = False
    
    
    # Display the stored settings without triggering another image update
    # form _load_image()  makes the slider show 0° and unchecks H/V.
    # display the stored setting  when called from _rebuild_modality_from_baseline()
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

        with slider.changed.blocked():
            slider.value = modality.rotation_angle
        horizontal_button.native.setChecked(modality.horizontal_flipped)
        vertical_button.native.setChecked(modality.vertical_flipped)
    

    
    
        """
        FileEdit
        validate path
        _read_image()
        controller.set_modality_image()
        invalidate previous registration
        update GUI 
        """
    def _load_image( file_widget: FileEdit, role: str, status: Label) -> None:
        
        """Loads an image from a file path specified in the file_widget and updates the corresponding status label.
        The function checks if the file path is valid and reads the image using the _read_image function. 
        It then updates the CorrelationSession object with the loaded image and adds it to the napari viewer. 
        If the file path is invalid or the file does not exist, it updates the status label accordingly. 
        
        Input: file_widget : FileEdit -> the control containing the selected path could be either " flm_file" or "tem_file" they both have .value attribute which is basically the path user selected.
                
                role: str -> FLM or TEM image helps to modify the CorrelationSession. which session state and image layer to update.
                status: Label -> it has a .value changes based different conditions. which status label to update.
                
        
        """
        value = file_widget.value # get the location of the image
        # did the user actually choose anything? is no path return
        if value is None or str(value) in {"", "."}: # reason for this is that empty path is not really empty it had ".", ""
            status.value = f"{role}: choose an image first"
            return


        path = Path(value) # convert the GUI value into path and make it Path object

        # what if file at that location does not exist ?
        if not path.is_file():
            status.value = f"{role}: file does not exist"
            return

        image = _read_image(path) # read the image.
        
        controller.set_modality_image(
            role,
            image,
        )

        #previous registration/warp can no longer
        # be trusted after replacing an image.
        _invalidate_registration()
        _update_registration_button()

        status.value = (
            f"{role}: {path.name} "
            f"{tuple(image.shape)}"
        )
        

        # Orientation becomes available only after
        # an image has successfully loaded.
        if role == "FLM":
            flip_flm_horizontal_button.enabled = True
            flip_flm_vertical_button.enabled = True
            reset_flm_orientation_button.enabled = True
            
            flm_rotation.enabled = True

            flm_points_status.value = (
                "FLM landmarks: not loaded"
            )
        else:
            flip_tem_horizontal_button.enabled = True
            flip_tem_vertical_button.enabled = True
            reset_tem_orientation_button.enabled = True
    
            
            tem_rotation.enabled = True

            tem_points_status.value = (
                "TEM landmarks: not loaded"
            )
        _sync_orientation_controls(role)
        
    # connect the buttons to the _load_image function with the appropriate parameters
    # small adapted supplies the flm specific role to the shared loading helper.
    def _on_load_flm(event=None):
        _load_image(
            flm_file,
            "FLM",
            flm_status,
        )
    
    def _on_load_tem(event=None):
        _load_image(
            tem_file,
            "TEM",
            tem_status,
        )
    # When the user clicks load_flm_button, load_tem_button button, call the respective function.
    load_flm_button.clicked.connect(
        _on_load_flm
    )

    load_tem_button.clicked.connect(
        _on_load_tem
    )

    """
    validate CSV
    read CSV
    controller.set_original_points()
    status
    invalidate
    """
    def _load_points( file_widget: FileEdit, role: str, status: Label) -> None:
        
        '''Loads points from a CSV file specified in the file_widget and updates the corresponding status label.
        The function checks if the file path is valid and reads the points using the read_points_csv function.
        It then updates the CorrelationSession object with the loaded points and adds them to the napari viewer. 
        If the file path is invalid or the file does not exist, it updates the status label accordingly.'''
        
        if not file_widget.value: # did the user actually choose anything?
            status.value = (
                f"{role} landmarks: choose a CSV first"
            )
            return

        path = Path(file_widget.value)  # get the path and make it Path object

        if not path.is_file():
            status.value = (
                f"{role} landmarks: file does not exist"
            )
            return

        points = read_points_csv(path)
        
        controller.set_original_points(
            role,
            points,
        )
        

        status.value = (
            f"{role} landmarks: "
            f"{path.name} "
            f"({len(points)} points)"
        )
        '''
        So changing the inputs means
        the previous registration is no longer trustworthy.
        '''
        
        _invalidate_registration()
        _update_registration_button()
        
    # connect the buttons load the points same as above
    def _on_load_flm_points(event=None):
        _load_points(
            flm_points_file,
            "FLM",
            flm_points_status,
        )

    def _on_load_tem_points(event=None):
        _load_points(
            tem_points_file,
            "TEM",
            tem_points_status,
        )
    
    load_flm_points_button.clicked.connect(
        _on_load_flm_points
    )

    load_tem_points_button.clicked.connect(
        _on_load_tem_points
    )
    
    # the actual call back function when registration clicked on
    # this creates the tranformed layer basically.
    def _on_calculate_registration(event=None):
        # invalidate an old warp when recalculating registration
        session.warped_flm = None
        warp_status.value = "Warp: not calculated"
        
        if (
            session.flm.points is None
            or session.tem.points is None
        ):
            registration_status.value = (
                "Registration: load both landmark sets first"
            )
            return

        try:
            registration = fit_affine(
                session.flm.points,
                session.tem.points,
            )
        except ValueError as error:
            registration_status.value = (
                f"Registration failed: {error}"
            )
            return

        session.registration = registration
        session.registration = registration

        # Analyze rotation and affine skew
        A = registration.matrix[:2, :2]

        # direction of tranformed x 
        horizontal_angle = np.degrees(
            np.arctan2(-A[1, 0], A[0, 0])
        )

        vertical_angle = np.degrees(
            np.arctan2(A[0, 1], A[1, 1])
        )

        angle_difference = (
            vertical_angle - horizontal_angle
        )

        

        print(
            f"Horizontal direction: {horizontal_angle:.3f}°"
        )
        print(
            f"Vertical direction: {vertical_angle:.3f}°"
        )
        print(
            f"Axis-angle difference: {angle_difference:.3f}°"
        )
       

        _update_warp_button()

        predicted = registration.apply(
            session.flm.points
        )
        
        _update_warp_button() # this is where we enable it because now the registration is done. 

        predicted = registration.apply(
            session.flm.points
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
            registered_affine_rc = affine_xy_to_rc(
                registration.matrix
            )

            controller._remove_layer_if_present(
                "Registered FLM"
            )

            viewer.add_image(
                session.flm.image,
                name="Registered FLM",
                affine=registered_affine_rc,
                opacity=0.5,
                blending="translucent",
            )
        
    calculate_registration_button.clicked.connect(
        _on_calculate_registration
    )
    
    def _on_warp(event=None):
        if (
            session.flm.image is None
            or session.tem.image is None
            or session.registration is None
        ):
            warp_status.value = (
                "Warp: load images and calculate registration first"
            )
            return
        #
        #suppose FLM is (732,782,2) and TEM (2046, 2880) then output shape is (2046, 2880)
        #Take the FLM image, transform it using the FLM -> TEM registration, and create the result on a 2046 × 2880 TEM-sized canvas.
        # Because the FLM is RGB the result should be Warped FLM (2046, 2880, 3)
        warped = warp_image(
            session.flm.image,
            session.registration,
            output_shape=session.tem.image.shape[:2],
        )

        session.warped_flm = warped

        layer_name = "Warped FLM"

        try:
            layer = viewer.layers[layer_name]
        except KeyError:
            viewer.add_image(
                warped,
                name=layer_name,
                opacity=float(warped_opacity.value), # can change the opacity based on slider
                blending="translucent",
            )
        else:
            layer.data = warped
            layer.opacity = float(
            warped_opacity.value
            )
        
        # The warp has now succeeded
        warped_opacity.enabled = True
        
        warp_status.value = (
            f"Warped FLM: {tuple(warped.shape)}"
        )
        
    warp_button.clicked.connect(
    _on_warp
    )
    
    # callback for the Warped Opactiy
    def _on_warped_opacity_change(event = None):
        
        try:
            layer = viewer.layers["Warped FLM"]
        except KeyError:
            return
        layer.opacity = float(
            warped_opacity.value
        )
    warped_opacity.changed.connect(
        _on_warped_opacity_change
    )
    

    # this is just a wrapped around the controller rebuild function    
    def _rebuild_modality_from_baseline(role: str) -> None:
        controller.rebuild_modality_from_baseline(
            role
        )

        _invalidate_registration()
        _sync_orientation_controls(role)
        
    #Figure out how to perform a horizontal flip.
    # All three function below are just wrappers now. real work in controller
    def _flip_modality_horizontal(role: str) -> None:
        controller.flip_modality_horizontal(
            role
        )

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
        
        
    reset_flm_orientation_button.clicked.connect(
        lambda event=None: _reset_modality_orientation("FLM")
    )
    reset_tem_orientation_button.clicked.connect(
        lambda event=None: _reset_modality_orientation("TEM")
    )
                    
     # connect the button to the callback.   
    def _on_flip_flm_horizontal(event=None):
        _flip_modality_horizontal("FLM")


    def _on_flip_tem_horizontal(event=None):
        _flip_modality_horizontal("TEM")

    def _on_flip_flm_vertical(event=None):
        _flip_modality_vertical("FLM")


    def _on_flip_tem_vertical(event=None):
        _flip_modality_vertical("TEM")

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
    
    # this is essntially our rotation callback
    def _set_modality_rotation(
        role: str,
        angle_degrees: float,
    ) -> None:
        controller.set_modality_rotation(
            role,
            angle_degrees,
        )

        _invalidate_registration()
        _sync_orientation_controls(role)
        
    def _on_flm_rotation_change(event=None):
        _set_modality_rotation(
            "FLM",
            float(flm_rotation.value),
        )


    def _on_tem_rotation_change(event=None):
        _set_modality_rotation(
            "TEM",
            float(tem_rotation.value),
        )
        
    flm_rotation.changed.connect(
        _on_flm_rotation_change
    )

    tem_rotation.changed.connect(
        _on_tem_rotation_change
    )
    
    
    def _update_rotation_readout(slider_widget,qslider,readout,position: int,) -> None:

        native_min = qslider.minimum()
        native_max = qslider.maximum()

        if native_max == native_min: # avoid diving by zero
            return

        # what percentage of the way is slider from min to max  0.625 for 45 degree
        fraction = (
            (position - native_min)
            / (native_max - native_min)
        )

        # and convert that to degrees
        # if the rotation was 45 degree
        #value = -180 + 0.625 * (180 - (-180)) = 45 
       
        value = (
            float(slider_widget.min)
            + fraction
            * (
                float(slider_widget.max)
                - float(slider_widget.min)
            )
        )

        # Change only what is displayed.
        # Do not tell magicgui that the value changed yet.
        signals_were_blocked = readout.blockSignals(
            True
        )

        readout.setValue(
            value
        )

        # allow the normal read out single again
        readout.blockSignals(
            signals_were_blocked
        )
    
    flm_qslider.sliderMoved.connect(
        lambda position: _update_rotation_readout(
            flm_rotation,
            flm_qslider,
            flm_rotation_readout,
            position,
        )
    )

    tem_qslider.sliderMoved.connect(
        lambda position: _update_rotation_readout(
            tem_rotation,
            tem_qslider,
            tem_rotation_readout,
            position,
        )
    )
            
    return Container(
        widgets=[
            Label(
                value="Offline Correlation"
            ),
            # FLM
            flm_file,
            load_flm_button,
            flm_status,
            flm_flip_row,
            flm_rotation,
            reset_flm_orientation_button,
            flm_points_file,
            load_flm_points_button,
            flm_points_status,

            # TEM
            tem_file,
            load_tem_button,
            tem_status,
            tem_flip_row,
            tem_rotation,
            reset_tem_orientation_button,
            tem_points_file,
            load_tem_points_button,
            tem_points_status,

            # Registration / warp
            calculate_registration_button,
            registration_status,

            warp_button,
            warp_status,
            warped_opacity,
            
        ]
    )
        

    
"""_read_image()
    disk → NumPy


make_offline_correlation_widget()
    owns CorrelationSession
    creates controls


callbacks
    user action → session → napari
    
    make_offline_correlation_widget()
        │
        ├── session
        │
        ├── flm_file
        │
        ├── tem_file
        │
        └── callback functions
                 │
                 └── remember session

The callbacks remain connected to the buttons, so Python keeps the objects they reference alive. 


There are three representations of data in here 

If we load an FLM image.

That image exists in three conceptually different places.

Place 1: disk

For example:

C:\data\FLM_image.tif

Just bytes in a file.

Place 2: Python/session

After reading:

session.flm_image

might contain:

np.ndarray

Now Python can calculate with it.

Place 3: napari

Napari has:

FLM layer

This is the visualization of the image.

So:

FILE ON DISK
     │
     │ _read_image()
     ▼
NUMPY ARRAY
     │
     ├──────────────► session.flm_image
     │
     └──────────────► napari "FLM" layer

That distinction is fundamental.

The session does not exist primarily to display the image.

Napari does not exist primarily to hold the computational state.

They have different jobs.

"""
