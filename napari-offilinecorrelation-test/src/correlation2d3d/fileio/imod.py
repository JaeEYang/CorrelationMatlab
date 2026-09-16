from pathlib import Path
import shutil
import subprocess
import tempfile
import mrcfile
import numpy as np

from time import perf_counter
"""
given an .st file. ask IMOD whether it contains montage piece coordinates. 
if it does build the montage with blendmont and return the final 2d image.
if not a montage, just return none.
inputs:
    path : to .st 
    
check that the IMOD is installed.
make a temporary scratch folder.
run extractpieces to pull tile positions out of the file. If that fails or there's only one tile, it's not a montage, so return None.
run blendmont to stitch the tiles. If that fails, raise an error.
read the stitched MRC into NumPy, make sure it's 2D, and return it.
The temp folder deletes itself automatically.

"""

def try_read_imod_montage(
    path: Path,
) -> np.ndarray | None:
    
    image_path = path.resolve() # convert to full absolute path
    
    # search the system PATH , like how the terminal finds the commands and the return the full path to the program
    extractpieces = shutil.which("extractpieces")
    blendmont = shutil.which("blendmont")
    
    # if the tool doesn't exist raise the error. To fix the setup problem.
    if extractpieces is None:
        raise RuntimeError(
            "IMOD extractpieces was not found"
        )

    if blendmont is None:
        raise RuntimeError(
            "IMOD blendmont was not found"
        )
    
    # create a temporary folder, everything imod produces goes here for now.
    with tempfile.TemporaryDirectory(prefix="correlation2d3d_imod_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        
        piece_list_path = (
            temporary_path / "pieces.pl"
        )
        
        start_time = perf_counter() # lets track the fking time
        
        # this basically extractpieces grid.mrc pieces.pl in a terminal
        extract_result = subprocess.run(
            [
                extractpieces,
                str(image_path),
                str(piece_list_path),
            ],
            capture_output=True, # grab what the program prints instead of dumping it on the console
            text=True, # give the output as a string
            check=False, # will check the result myself, don't throw exception
        )
        
        after_extract = perf_counter()
        print(
            f"extractpieces: "
            f"{after_extract - start_time:.2f} s"
        )
        
        # zero means success. If failed or "succeeded" but didn't write the file. the header probably had no monatge info, so this ins't montage return None 
        if (
            extract_result.returncode != 0
            or not piece_list_path.exists()
        ):
            return None
        
        # reads the text file into a NumPy array of integers. 
        piece_coordinates = np.loadtxt(
            piece_list_path,
            dtype=np.int64,
            ndmin=2, #even if there is only one row, keep the result at least 2d.
        )
        
        after_piece_read = perf_counter()
        print(
            f"piece list read: "
            f"{after_piece_read - after_extract:.2f} s"
        )

        # three checks any one means not a usable montage.
        #Not a 2D table: something weird got parsed.
        #Not exactly 3 columns: not the X, Y, Z format expected.
        #One tile or fewer: a single tile isn't a montage, nothing to stitch.
        if (
            piece_coordinates.ndim != 2 # this becomes kinda redudent cause of above ndminn =2 (but doen't hurt to have)
            or piece_coordinates.shape[1] != 3
            or len(piece_coordinates) <= 1
        ):
            return None

        # stitch with blendmont prep the files
        output_path = (
            temporary_path / "blend.mrc" # final image. 
        )

        aligned_path = (
            temporary_path / "aligned.pl"
        )

        root_path = (
            temporary_path / "blend"
        )

        blend_result = subprocess.run(
            [
                blendmont,  # command itself
                "-imin",
                str(image_path), # path to .st image containing tiles
                "-plin",
                str(piece_list_path), # this is .pl where the tiles are supposed to go. this gets extracted from the header using extractpieces commands which runs above
                "-imout",
                str(output_path), # the new final stitched image .mrc
                "-aligned",
                str(aligned_path), # write the new piece coordinate list here
                "-rootname",
                str(root_path), # sets the base name for the intermediate files blendmont creates.
                "-sloppy",  # tells blendmont the tiles may be off from their recorded positions by a fair amount, so it searches harder when matching overlaps.
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        
        after_blend = perf_counter()
        print(
            f"blendmont: "
            f"{after_blend - after_piece_read:.2f} s"
        )

        # by this time we know this file is a real montage with mutlipe tiles. 
        # so if stitching fails, that is real issue. raise the erorr and inclue the blendmont's error output and normal output message. will help with debug
        if blend_result.returncode != 0:
            raise RuntimeError(
                "IMOD blendmont failed:\n"
                f"{blend_result.stderr}\n"
                f"{blend_result.stdout}"
            )

        # Opens the stitched MRC.
        with mrcfile.open(
            str(output_path),
            permissive=True,
        ) as mrc:
            image = np.array(
                mrc.data,
                copy=True,
            )
            
        after_mrc_read = perf_counter()
        print(
            f"MRC read/copy: "
            f"{after_mrc_read - after_blend:.2f} s"
        )

        if (
            image.ndim == 3
            and image.shape[0] == 1
        ):
            image = image[0]

        if image.ndim != 2:
            raise ValueError(
                "Blended montage must be a 2D image"
            )
            
        print(
            f"IMOD montage total: "
            f"{after_mrc_read - start_time:.2f} s"
        )

        return image



 