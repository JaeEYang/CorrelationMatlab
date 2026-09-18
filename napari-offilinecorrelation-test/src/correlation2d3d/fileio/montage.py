"""we will get meta data -> pixels -> raster"""
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np

from correlation2d3d.fileio.mdoc import (
    MontageMetadata,
    MontagePieceMetadata,
    extract_montage_metadata,
    parse_mdoc,
)


@dataclass(frozen=True)
class MontageResult:
    """
    Result of reconstructing one montage.

    data:
        Final reconstructed 2D raster.

    metadata:
        Typed montage geometry derived from the MDOC.

    source_path:
        ST/MRC file that supplied the image pixels.

    mdoc_path:
        MDOC file that supplied the montage geometry.
        None is allowed because other reconstruction backends may not use
        an external MDOC file.

    method:
        Reconstruction strategy used.

    overlap_policy:
        Rule used for pixels in overlapping tile regions.
    """

    data: np.ndarray
    metadata: MontageMetadata
    source_path: Path
    mdoc_path: Path | None
    method: str
    overlap_policy: str


def _aligned_piece_raster_slices(
    piece: MontagePieceMetadata,
    metadata: MontageMetadata,
) -> tuple[slice, slice]:
    """
    Convert one tile's SerialEM AlignedPieceCoords into output-array slices.

    Input:
        piece:
            Metadata for one montage tile.

        metadata:
            Geometry of the complete aligned montage.

    Output:
        (row_slice, column_slice)

        These slices tell reconstruct_aligned_montage() where this tile
        belongs in the output NumPy array.

    Important coordinate convention:
        SerialEM montage coordinates are right-handed:
            +X points right
            +Y points up

        MRC image storage also starts at the lower-left, so increasing
        array row index corresponds to increasing montage Y.

        Therefore we translate X and Y into a zero-based coordinate system,
        but we DO NOT invert Y here.

        A viewer such as napari may display array row 0 at the top. That is
        a display convention and should not be confused with the scientific
        MRC/montage coordinate system.
    """

    # Direct aligned reconstruction requires a solved tile position.
    if piece.aligned_piece_coordinates is None:
        raise ValueError(
            f"ZValue {piece.z_value} has no AlignedPieceCoords"
        )

    # extract_montage_metadata() calculated these from all aligned pieces.
    if metadata.aligned_origin_xy is None:
        raise ValueError(
            "Montage has no aligned origin"
        )

    if metadata.aligned_size_xy is None:
        raise ValueError(
            "Montage has no aligned output size"
        )

    # SerialEM stores placement coordinates as x, y, z.
    # This montage is 2D, so z is currently not used for raster placement.
    aligned_x, aligned_y, _aligned_z = (
        piece.aligned_piece_coordinates
    )

    # The minimum solved coordinate may be negative.
    #
    # For this dataset:
    #     min_x = -29
    #     min_y = -60
    min_x, min_y = metadata.aligned_origin_xy

    # Metadata uses x/y ordering:
    #     width, height
    canvas_width, canvas_height = (
        metadata.aligned_size_xy
    )

    tile_width, tile_height = (
        metadata.tile_size_xy
    )

    # --------------------------------------------------------------
    # Convert aligned X into a zero-based array column.
    #
    # Example for Z0:
    #
    #     aligned_x = -29
    #     min_x     = -29
    #
    #     col_start = 0
    # --------------------------------------------------------------
    col_start = aligned_x - min_x
    col_end = col_start + tile_width

    # --------------------------------------------------------------
    # Convert aligned Y into a zero-based MRC-style array row.
    #
    # This is the important correction.
    #
    # Both SerialEM montage Y and MRC storage row position increase from
    # the lower side of the image toward the upper side, so there is no
    # Y inversion here.
    #
    # Example for Z0:
    #
    #     aligned_y = -60
    #     min_y     = -60
    #
    #     row_start = 0
    # --------------------------------------------------------------
    row_start = aligned_y - min_y
    row_end = row_start + tile_height

    # Catch a coordinate/bounds bug explicitly instead of allowing NumPy
    # indexing to hide it by clipping or producing confusing results.
    if (
        row_start < 0
        or col_start < 0
        or row_end > canvas_height
        or col_end > canvas_width
    ):
        raise ValueError(
            f"Aligned tile ZValue {piece.z_value} "
            "falls outside the calculated montage bounds"
        )

    # NumPy still indexes as [row, column].
    # Here row is being used in MRC-native storage order.
    return (
        slice(row_start, row_end),
        slice(col_start, col_end),
    )
    

def reconstruct_aligned_montage(
    st_path: Path,
    metadata: MontageMetadata,
) -> MontageResult:
    
    """
    Reconstruct a 2D montage directly from an ST/MRC tile stack using
    SerialEM AlignedPieceCoords.

    Input:
        st_path:
            Path to the source ST/MRC montage stack.

        metadata:
            Typed montage geometry produced by
            extract_montage_metadata().

    Output:
        MontageResult containing:
            reconstructed image pixels
            montage metadata
            source-file provenance
            reconstruction method
            overlap policy

    Current overlap policy:
        Tiles are copied in ZValue order. Where tiles overlap, a later
        tile overwrites pixels from an earlier tile.

        This is intentionally a simple first reconstruction strategy. Works actually really well.
    """
   
    if metadata.aligned_size_xy is None:
        raise ValueError(
            "Aligned montage reconstruction requires "
            "AlignedPieceCoords for every piece"
        )

    canvas_width, canvas_height = (
        metadata.aligned_size_xy
    )

    tile_width, tile_height = (
        metadata.tile_size_xy
    )

    # --------------------------------------------------------------
    # Open the ST/MRC file using memory mapping.
    #
    # mrcfile.mmap() exposes the image stack without first copying the
    # complete file into normal RAM.
    #
    # Individual tiles are accessed below with:
    #
    #     mrc.data[piece.z_value]
    #
    # This is important because real montage files may eventually be
    # many gigabytes.
    # --------------------------------------------------------------
    with mrcfile.mmap(str(st_path),mode="r",permissive=True) as mrc:

        data = mrc.data

        # ----------------------------------------------------------
        # Validate the source structure.
        #
        # For this montage we expect:
        #
        #     data.shape == (24, 1936, 2748)
        #
        # which means:
        #
        #     24 image sections
        #     1936 rows per tile
        #     2748 columns per tile
        # ----------------------------------------------------------
        if data.ndim != 3:
            raise ValueError(
                "Aligned montage reconstruction expects "
                f"a 3D tile stack, got shape {data.shape}"
            )

        source_sections = data.shape[0]

        if source_sections <= max(
            piece.z_value
            for piece in metadata.pieces
        ):
            raise ValueError(
                "MDOC refers to a ZValue that is outside "
                "the ST/MRC image stack"
            )

        source_tile_height = data.shape[1]
        source_tile_width = data.shape[2]

        if (
            source_tile_width != tile_width
            or source_tile_height != tile_height
        ):
            raise ValueError(
                "ST/MRC tile dimensions do not match MDOC "
                f"ImageSize: source is "
                f"{source_tile_width} x {source_tile_height}, "
                f"MDOC says {tile_width} x {tile_height}"
            )

        # ----------------------------------------------------------
        # Allocate the final aligned raster.
        #
        # Metadata uses x/y:
        #
        #     (width, height)
        #
        # NumPy arrays use row/column:
        #
        #     (height, width)
        #
        # so the ordering is intentionally reversed here.
        # ----------------------------------------------------------
        canvas = np.zeros(
            (
                canvas_height,
                canvas_width,
            ),
            dtype=data.dtype,
        )

        # ----------------------------------------------------------
        # Copy each source tile to its solved aligned position.
        #
        # metadata.pieces was sorted by ZValue when it was created,
        # so this loop follows source-section order.
        # ----------------------------------------------------------
        for piece in metadata.pieces:

            # ZValue tells us which plane in the ST/MRC stack contains
            # this particular montage tile.
            tile = data[piece.z_value]

            # Convert the SerialEM aligned x/y position into NumPy
            # row/column slices on the output canvas.
            row_slice, col_slice = (
                _aligned_piece_raster_slices(
                    piece,
                    metadata,
                )
            )

            # Copy the tile into its output location.
            # In overlap regions this assignment means the current tile
            # replaces pixels written by an earlier tile.
            canvas[
                row_slice,
                col_slice,
            ] = tile

    # Return pixels together with enough metadata to understand  what this raster represents later in the application.
    # also gotta resolve the path so the result records the concrete source file, rather than depending on the caller's current working directory.
    return MontageResult(
        data=canvas,
        metadata=metadata,
        source_path=Path(st_path).resolve(),
        mdoc_path=None,
        method="mdoc_aligned",
        overlap_policy="overwrite",
    )
    
def load_aligned_montage(
    st_path: Path,
    mdoc_path: Path,
) -> MontageResult:
    """
    Load and reconstruct a SerialEM montage using stored MDOC alignment.

    Input:
        st_path:
            ST/MRC file containing the source image tiles.

        mdoc_path:
            MDOC file containing the montage metadata and
            AlignedPieceCoords.

    Output:
        MontageResult containing:
            reconstructed pixels
            typed montage geometry
            source ST/MRC path
            source MDOC path
            reconstruction method
            overlap policy

    Basically connect the metadata parser, geometry extractor, and direct reconstruction functions into one workflow.
    """

   
    st_path = Path(st_path).resolve()
    mdoc_path = Path(mdoc_path).resolve()

    # Fail clearly if the user-selected files do not exist.
    if not st_path.exists():
        raise FileNotFoundError(
            f"ST/MRC file does not exist: {st_path}"
        )

    if not mdoc_path.exists():
        raise FileNotFoundError(
            f"MDOC file does not exist: {mdoc_path}"
        )

    # Parse the MDOC without discarding unknown metadata.
    mdoc = parse_mdoc(mdoc_path)

    # Convert the raw MDOC representation into typed montage geometry.
    metadata = extract_montage_metadata(mdoc)

    # Reconstruct the raster using the stored AlignedPieceCoords.
    #
    # This lower-level function knows about:
    #     source pixels
    #     tile geometry
    #
    # but deliberately does not know which MDOC file produced metadata.
    result = reconstruct_aligned_montage(
        st_path,
        metadata,
    )

    # Create the final result with complete provenance.
    #
    # We construct a new immutable MontageResult rather than modifying
    # the existing frozen dataclass instance.
    return MontageResult(
        data=result.data,
        metadata=result.metadata,
        source_path=result.source_path,
        mdoc_path=mdoc_path,
        method=result.method,
        overlap_policy=result.overlap_policy,
    )
    
def suggest_exact_mdoc_path(
st_path: Path,
) -> Path | None:
    """
    
    Input:
        st_path:
            Path to an ST/MRC image stack.
    Output:
        Path:
            When a file named exactly "<source filename>.mdoc" exists
            beside the source image.
        None:
            When that exact companion file does not exist.

    We are gonna keep this simple. Just one try and then back to user
    """

    st_path = Path(st_path)


    #   82map_2.st.mdoc
    candidate = st_path.with_name(
        st_path.name + ".mdoc"
    )

    # Only suggest the candidate when it is an actual file.
    #
    # is_file() is stronger than exists() here because a directory with the
    # same name would not be a usable MDOC.
    if candidate.is_file():
        return candidate.resolve()

    # No exact math we good. 
    return None