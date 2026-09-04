from pathlib import Path

import mrcfile
import numpy as np
from magicgui.widgets import Container, FileEdit, Label, PushButton, FloatSlider # This is a magicgui container that will hold the widgets for the offline correlation tool. 
#It includes a label, file edit widgets for loading images and points, and push buttons for performing actions such as warping images and computing registrations.
from skimage import io

from correlation2d3d.session import CorrelationSession

from correlation2d3d.fileio.points_csv import read_points_csv

from correlation2d3d.core.warp import warp_image

from correlation2d3d.core.orientation import flip_horizontal, horizontal_flip_matrix

from correlation2d3d.core.transform import apply_affine_matrix, fit_affine, affine_xy_to_rc



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
    # this becomes out generic loader for modalities make it bit tidy to keep track of session state.
    def _get_modality(role: str):
        if role == "FLM":
            return session.flm

        if role == "TEM":
            return session.tem

        raise ValueError(
            f"unknown modality role: {role}"
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
        text="↔ Flip H"
    )

    # ascii-exempt: Qt widget label, rendered by the GUI and never written to stdout
    flip_tem_horizontal_button = PushButton(
        text="↔ Flip H"
    )

    flip_flm_horizontal_button.enabled = False
    flip_tem_horizontal_button.enabled = False
    
    # helper if layer exists we can remove it if not do nothing.
    def _remove_layer_if_present(
        layer_name: str,
    ) -> None:
        try:
            layer = viewer.layers[layer_name]
        except KeyError:
            return

        viewer.layers.remove(layer)
        
    
    
    
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
    
    # lets say we did the registration and then changed the image orientation we should able to invalidate the old registraion matrix
    # and also the warping becomes invalidated need to recalulate both 
    def _invalidate_registration() -> None:
        session.registration = None
        session.warped_flm = None

        registration_status.value = (
            "Registration: not calculated"
        )

        warp_status.value = (
            "Warp: not calculated"
        )

        warp_button.enabled = False
        warped_opacity.enabled = False

        # this removed those two layers aswell
        _remove_layer_if_present(
            "FLM Landmarks Registered to TEM"
        )

        _remove_layer_if_present(
            "Warped FLM"
        )
        _remove_layer_if_present(
            "Registered FLM"
        )
    
    
    # 
    def _load_image( file_widget: FileEdit, role: str, status: Label) -> None:
        
        """Loads an image from a file path specified in the file_widget and updates the corresponding status label.
        The function checks if the file path is valid and reads the image using the _read_image function. 
        It then updates the CorrelationSession object with the loaded image and adds it to the napari viewer. 
        If the file path is invalid or the file does not exist, it updates the status label accordingly. 
        
        Input: file_widget : FileEdit -> could be either " flm_file" or "tem_file" they both have .value attribute which is basically the path user selected.
                
                role: str -> FLM or TEM image helps to modify the CorrelationSession
                status: Label -> it has a .value changes based different conditions.
                
        
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
        
        

        #update the session object with the loaded image based on the role (FLM or TEM)
        # Our session says: this exact NumPy array is the FLM image for this correlation job
        modality = _get_modality(role)
        
        # both og and image will be same initially
        modality.original_image = np.array(
            image,
            copy=True,
        )
        modality.image = np.array(
            image,
            copy=True,
        )
        # this resets the orintation back to identity incase user load the flm again after flip (reset-on-reload).
        modality.orientation_matrix = np.eye(
            3,
            dtype=np.float64,
        )
        
        #landmarks belonged to the previous image.
        #the user must load/confirm landmarks for this image.
        modality.original_points = None
        modality.points = None
        
        _remove_layer_if_present(
            f"{role} Landmarks"
        )

        #previous registration/warp can no longer
        # be trusted after replacing an image.
        _invalidate_registration()
        _update_registration_button()

        # Create or update the napari image layer.
                # Recreate the layer so napari detects grayscale/RGB correctly.
        _remove_layer_if_present(role)

        viewer.add_image(
            modality.image,
            name=role,
        )

        status.value = (
            f"{role}: {path.name} "
            f"{tuple(image.shape)}"
        )

        # Orientation becomes available only after
        # an image has successfully loaded.
        if role == "FLM":
            flip_flm_horizontal_button.enabled = True
            flm_points_status.value = (
                "FLM landmarks: not loaded"
            )
        else:
            flip_tem_horizontal_button.enabled = True
            tem_points_status.value = (
                "TEM landmarks: not loaded"
            )
        
    # connect the buttons to the _load_image function with the appropriate parameters
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
        
        # same as above udpdate the session object
        modality = _get_modality(role)

        modality.original_points = points
        modality.points = apply_affine_matrix(
        modality.orientation_matrix,
        modality.original_points,
        )

        layer_name = f"{role} Landmarks"

        napari_points = modality.points.to_rc() # convert to napari points convention y,x/ rc these will be recieved by napari frontend
       
        try:
            layer = viewer.layers[layer_name]
        except KeyError:
            viewer.add_points(
            napari_points,
            name=layer_name,
            size=32,
            face_color="red",
        )
        else:
            layer.data = napari_points
            layer.size = 32
            layer.face_color = "red"

        status.value = (
            f"{role} landmarks: "
            f"{path.name} "
            f"({len(points)} points)"
        )
        '''
        So changing the inputs means:
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
            size=32,
            face_color="#00007f",
        )
        else:
            layer.data = transformed_rc
            layer.size = 32
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

            _remove_layer_if_present(
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
    
    def _flip_modality_horizontal(role:str)->None:
        modality = _get_modality(role)
        
        # oreientation should work even if landmarks have not been loaded yet.
        if modality.image is None:
            return

        width = modality.image.shape[1]
        # get the flipped image and the corrsponding flipped landmarks#
        flip_matrix = horizontal_flip_matrix(width)
        
         # Transform the current image.
        flipped_image, _ = flip_horizontal(
            modality.image
        )

        modality.image = flipped_image

        # Record how original coordinates map
        # into the new current coordinates.
        modality.orientation_matrix = (
            flip_matrix
            @ modality.orientation_matrix
        )

        #rebuild current points from the ORIGINAL points.
        #never progressively modify current points.
        if modality.original_points is not None:
            modality.points = apply_affine_matrix(
                modality.orientation_matrix,
                modality.original_points,
            )
        else:
            modality.points = None

        # Update image shown in napari.
        viewer.layers[role].data = modality.image

        # Update landmarks only if they currently exist.
        if modality.points is not None:
            layer_name = f"{role} Landmarks"

            try:
                layer = viewer.layers[layer_name]
            except KeyError:
                pass
            else:
                layer.data = modality.points.to_rc()

        _invalidate_registration()
            
     # connect the button to the callback.   
    def _on_flip_flm_horizontal(event=None):
        _flip_modality_horizontal("FLM")


    def _on_flip_tem_horizontal(event=None):
        _flip_modality_horizontal("TEM")


    flip_flm_horizontal_button.clicked.connect(
        _on_flip_flm_horizontal
    )

    flip_tem_horizontal_button.clicked.connect(
        _on_flip_tem_horizontal
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
            flip_flm_horizontal_button,

            flm_points_file,
            load_flm_points_button,
            flm_points_status,

            # TEM
            tem_file,
            load_tem_button,
            tem_status,
            flip_tem_horizontal_button,

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