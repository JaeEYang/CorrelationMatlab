from pathlib import Path

import mrcfile
import numpy as np
from magicgui.widgets import Container, FileEdit, Label, PushButton, FloatSlider # This is a magicgui container that will hold the widgets for the offline correlation tool. 
#It includes a label, file edit widgets for loading images and points, and push buttons for performing actions such as warping images and computing registrations.
from skimage import io

from correlation2d3d.session import CorrelationSession

from correlation2d3d.fileio.points_csv import read_points_csv

from correlation2d3d.core.transform import fit_affine

from correlation2d3d.core.warp import warp_image

from correlation2d3d.core.orientation import flip_horizontal


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

# This function creates a magicgui container widget for the offline correlation tool.
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
    
    flip_horizontal_button = PushButton(
    text="↔ Flip H"
    )
    
    
    
    # a small helper to decide if warping is possible, do we have the images and the registration matrix.
    def _update_warp_button() -> None:
        warp_button.enabled = (
            session.flm_image is not None
            and session.tem_image is not None
            and session.registration is not None
        )
    
    # enable the registration buttion is both flm and tem data exist in the session this gets populated in the _load_points
    def _update_registration_button() -> None:
            calculate_registration_button.enabled = (
            session.flm_points is not None
            and session.tem_points is not None
        )
    
    
    # 
    def _load_image( file_widget: FileEdit, role: str, status: Label) -> None:
        
        '''Loads an image from a file path specified in the file_widget and updates the corresponding status label.
        The function checks if the file path is valid and reads the image using the _read_image function. 
        It then updates the CorrelationSession object with the loaded image and adds it to the napari viewer. 
        If the file path is invalid or the file does not exist, it updates the status label accordingly. 
        
        Input: file_widget : FileEdit -> could be either " flm_file" or "tem_file" they both have .value attribute which is basically the path user selected.
                
                role: str -> FLM or TEM image helps to modify the CorrelationSession
                status: Label -> it has a .value changes based different conditions.
                
        
        '''
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
        if role == "FLM":
            session.flm_image = image
        else:
            session.tem_image = image
        
        # use napari's ability to access by its name
        try:
            layer = viewer.layers[role]
        except KeyError:
            viewer.add_image(
                image,
                name=role,
            )
        else:
            layer.data = image # If an FLM layer already exists Maybe you loaded another FLM previously. Instead of creating: FLM, FLM [1]. FLM[2] ...just use the existing layer and update with new image

        status.value = (
            f"{role}: {path.name} "
            f"{tuple(image.shape)}"
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
        if role == "FLM":
            session.flm_points = points
        else:
            session.tem_points = points

        layer_name = f"{role} Landmarks"

        napari_points = points.to_rc() # convert to napari points convention y,x/ rc these will be recieved by napari frontend
       
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
        
        session.registration = None
        registration_status.value = (
            "Registration: not calculated"
        )

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
            session.flm_points is None
            or session.tem_points is None
        ):
            registration_status.value = (
                "Registration: load both landmark sets first"
            )
            return

        try:
            registration = fit_affine(
                session.flm_points,
                session.tem_points,
            )
        except ValueError as error:
            registration_status.value = (
                f"Registration failed: {error}"
            )
            return

        session.registration = registration
        
        _update_warp_button() # this is where we enable it because now the registration is done. 

        predicted = registration.apply(
            session.flm_points
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
            face_color="red",
        )
        else:
            layer.data = transformed_rc
            layer.size = 32
            layer.face_color = "red"

        registration_status.value = (
            f"Registration RMSE: "
            f"{registration.rmse:.3f} TEM pixels"
        )
        
    calculate_registration_button.clicked.connect(
        _on_calculate_registration
    )
    
    def _on_warp(event=None):
        if (
            session.flm_image is None
            or session.tem_image is None
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
            session.flm_image,
            session.registration,
            output_shape=session.tem_image.shape[:2],
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
    
    def _on_flip_horizontal(event=None):
        # if no image just return
        if (
            session.flm_image is None
            or session.flm_points is None
        ):
            return

        # get the flipped image and the corrsponding flipped landmarks#
        flipped_image, flipped_points = flip_horizontal(
            session.flm_image,
            session.flm_points,
        )

        # save them in the CorrelationSession to update the session state
        session.flm_image = flipped_image
        session.flm_points = flipped_points

        # for each layer update the correspoint image data and landmarks data
        viewer.layers["FLM"].data = flipped_image
        viewer.layers["FLM Landmarks"].data = (
            flipped_points.to_rc()
        )

        # now we gotta reset the registration and warped  cause it would "change" 
        session.registration = None
        session.warped_flm = None

        #update the status message for the registration
        registration_status.value = (
            "Registration: not calculated"
        )

        warp_status.value = (
            "Warp: not calculated"
        )

        # change the warp button back to false cause we need to redo the registration and then turn it back on same goes with opacity.
        warp_button.enabled = False
        warped_opacity.enabled = False
        
     # connect the button to the callback.   
    flip_horizontal_button.clicked.connect(
        _on_flip_horizontal
    )

    
        
    return Container(
        widgets=[
            Label(
                value="Offline Correlation"
            ),

            flm_file,
            load_flm_button,
            flm_status,

            tem_file,
            load_tem_button,
            tem_status,

            flm_points_file,
            load_flm_points_button,
            flm_points_status,

            tem_points_file,
            load_tem_points_button,
            tem_points_status,
            
            calculate_registration_button,
            registration_status,
            
            warp_button,
            warp_status,
            warped_opacity,
            
            flip_horizontal_button,
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