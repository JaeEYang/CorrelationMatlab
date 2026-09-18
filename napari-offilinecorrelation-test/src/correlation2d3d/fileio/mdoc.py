"""

82map_2.st.mdoc
        |
        v
loss-preserving autodoc parser
        |
        +-- global fields
        |
        +-- title sections
        |
        +-- ZValue sections
        |
        +-- MontSection sections
        |
        v
typed montage view
        |
        +-- section index
        +-- PieceCoordinates
        +-- AlignedPieceCoords
        +-- StagePosition
        +-- pixel spacing
        +-- tile size
        +-- FullMontSize
        +-- keep every unknown field still preserved
        
MdocSection
|
+-- section_type = "ZValue"
|
+-- section_value = "0"
|
+-- fields
      |
      +-- TiltAngle
      +-- PieceCoordinates
      +-- StagePosition
      +-- ...

"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MontagePieceMetadata:
    """
    For 1 image/tile this is the typed geometry (only three things we care about right now for montaging)

    z_value:
        Index of the image section in the MRC/ST file.

    piece_coordinates:
        Nominal x, y, z placement planned for the montage.

    aligned_piece_coordinates:
        Final solved x, y, z montage placement when SerialEM alignment
        information is available.

    stage_position:
        Microscope stage x/y position in microns when present.
    """

    z_value: int

    piece_coordinates: tuple[
        int,
        int,
        int,
    ]

    aligned_piece_coordinates: (
        tuple[int, int, int] | None
    )

    stage_position: (
        tuple[float, float] | None
    )
    
@dataclass(frozen=True)
class MontageMetadata:
    """
    geometry metadata of entire montage so collection of  MontagePieceMetadata

    pieces:
        All image sections that make up the montage.

    tile_size_xy:
        Width and height of one source tile in pixels.
        This is stored explicitly as (x_size, y_size), not NumPy (rows, cols).

    nominal_size_xy:
        Width and height of the regular montage described by
        PieceCoordinates / FullMontSize.

    grid_size_xy:
        Number of distinct nominal tile positions along x and y.

    pixel_spacing_angstrom:
        Physical pixel spacing stored by SerialEM, in Angstroms per pixel.

    aligned_origin_xy:
        Minimum x/y coordinate among AlignedPieceCoords. This can be negative. will use for shifting the orgin

    aligned_size_xy:
        Width and height needed to contain every aligned tile without cropping.
    """

    pieces: tuple[MontagePieceMetadata, ...]

    tile_size_xy: tuple[int, int]

    nominal_size_xy: tuple[int, int]

    grid_size_xy: tuple[int, int]

    pixel_spacing_angstrom: float

    aligned_origin_xy: (
        tuple[int, int] | None
    )

    aligned_size_xy: (
        tuple[int, int] | None
    )

""" PixelSpacing = 2202.2 """
@dataclass(frozen=True)
class MdocField:
    key: str
    value: str
    

"""

[ZValue = 0]

TiltAngle = -0.0067243
PieceCoordinates = 0 0 0
StagePosition = 798.817 761.77
...

"""
@dataclass(frozen=True)
class MdocSection:
    section_type: str  # name of the section e.g "ZValue"
    section_value: str # this is the value of that set  "0"
    fields: tuple[MdocField, ...] # what is inside each section represented as MdocField 
    
    
@dataclass(frozen=True)
class MdocFile:
    global_fields: tuple[MdocField, ...]
    sections: tuple[MdocSection, ...]
    
 
 
 

"""
GLOBAL MODE
    |
    | encounter [ZValue = 0]
    v
SECTION MODE: ZValue 0
    |
    | collect fields
    |
    | encounter [ZValue = 1]
    v
save ZValue 0
start ZValue 1
"""
    
def parse_mdoc(
    path: Path
) -> MdocFile:
    # these two lists are construction workspace
    global_fields: list[MdocField] = []
    sections: list[MdocSection] = []

    #these variables describe where we currently are
    current_section_type: str | None = None
    current_section_value: str | None = None
    current_fields: list[MdocField] = []

    # make sure eveything is collected for each section before moving on.
    def finish_current_section() -> None:
        # we are reassigning them here so without nonlocal, assignment would create new local variables inside the nested function.
        nonlocal current_section_type
        nonlocal current_section_value
        nonlocal current_fields

        if current_section_type is None:
            return

        # convert the current mutable parsin state into the one immutatble completed section.
        sections.append(
            MdocSection(
                section_type=current_section_type,
                section_value=current_section_value or "",
                fields=tuple(current_fields),
            )
        )

        # reset the parser state
        current_section_type = None
        current_section_value = None
        current_fields = []

    # this is where real parsing actaully begins
    with Path(path).open("r",encoding="utf-8") as file:
        
        for line_number, raw_line in enumerate(file,start=1): # get index and data
            line = raw_line.strip() 

            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                finish_current_section() # pacakage up what we have collected so far and push it to sections

                section_text = line[1:-1] # for [ZValue = 0] gives  ZValue = 0

                if "=" not in section_text: # every field has key value unexpected syntax let me knoww. 
                    raise ValueError(
                        f"Malformed MDOC section header "
                        f"on line {line_number}: {line}"
                    )

                section_type, section_value = section_text.split("=", 1) # split at first = 

                current_section_type = section_type.strip()
               
                current_section_value = section_value.strip()
                

                continue

            if "=" not in line:
                raise ValueError(
                    f"Malformed MDOC field "
                    f"on line {line_number}: {line}"
                )

            key, value = line.split("=", 1)

            field = MdocField(
                key=key.strip(),
                value=value.strip(),
            )

            # preserve same key diffrent scope 
            if current_section_type is None:
                global_fields.append(field)
            else:
                current_fields.append(field)

    finish_current_section() # gotta save the final section that is why called here again. usally next saves previous

    return MdocFile(
        global_fields=tuple(global_fields),
        sections=tuple(sections),
    )
    
def get_field_values(
    fields: tuple[MdocField,...],
    key: str,
) -> tuple[str,...]:
    """
    return every raw values associated with one metadata key
    return all the matching values instead of silently choosing values
    because raw MDOC representation intentionally preserves duplicate fields.
    """

    # look in order and keep the one that matches
    matching_values = tuple(
        field.value
        for field in fields
        if field.key == key
    )

    return matching_values

def get_single_field_value(
    fields: tuple[MdocField, ...],
    key: str,
) -> str | None:
    """
    Return one raw value for a metadata key that is expected to be unique

    Returns None when the field is absent

    Raises ValueError when the same key occurs more than once
    """

    # Reuse the lower-level lookup function so duplicate handling is kept in
    # one place.
    values = get_field_values(
        fields,
        key,
    )

    if len(values) == 0:
        return None

    # normal
    if len(values) == 1:
        return values[0]

    raise ValueError(
        f"Expected at most one '{key}' field, "
        f"but found {len(values)}"
    )
    
def _parse_int_values(
    value: str,
    *,
    expected_count: int,
    field_name: str,
) -> tuple[int, ...]:
    """
    Convert one space-separated MDOC value into a tuple of integers.

    Example:
        "-29 -60 0"

    becomes:
        (-29, -60, 0)

    expected_count protects us from accepting incomplete coordinate fields.
    """

    # MDOC numeric vectors are stored as whitespace-separated text.
    parts = value.split()

    # Coordinate fields have known dimensions.
    # e.g PieceCoordinates must contain x, y, z, so we expect 3.
    if len(parts) != expected_count:
        raise ValueError(
            f"{field_name} must contain "
            f"{expected_count} values, "
            f"got {len(parts)}: {value!r}"
        )

    try:
        # Convert each textual number into a Python integer.
        return tuple(
            int(part)
            for part in parts
        )
    except ValueError as error:
        raise ValueError(
            f"{field_name} must contain integers: "
            f"{value!r}"
        ) from error
 
        
def _parse_float_values(
    value: str,
    *,
    expected_count: int,
    field_name: str,
) -> tuple[float, ...]:
    """
    Same as above but for floats
    """
   
    parts = value.split()

 
    if len(parts) != expected_count:
        raise ValueError(
            f"{field_name} must contain "
            f"{expected_count} values, "
            f"got {len(parts)}: {value!r}"
        )

    try:
        return tuple(
            float(part)
            for part in parts
        )
    except ValueError as error:
        raise ValueError(
            f"{field_name} must contain numeric values: "
            f"{value!r}"
        ) from error
        

       
def _montage_piece_from_section(
    section: MdocSection,
) -> MontagePieceMetadata:
    """
    Convert one raw [ZValue = ...] MDOC section into typed montage metadata

    This function interprets only fields whose scintific meaning I currently
    understand and will be needed for me. The original MdocSection remains unchanged and still contains
    every raw metadata field. so we can get other things later aswell
    """

    # conversion only makes sense for image sections
    #  [MontSection = ...] or [T = ...] section represents something else
    if section.section_type != "ZValue":
        raise ValueError(
            "Montage pieces must come from "
            f"ZValue sections, got "
            f"{section.section_type!r}"
        )

    try:
        # ZValue identifies which image plane in the ST/MRC stack contains this tile.
        z_value = int(section.section_value)
    except ValueError as error:
        raise ValueError(
            "ZValue section identifier must "
            f"be an integer: "
            f"{section.section_value!r}"
        ) from error

    # Every montage piece needs nominal PieceCoordinates.
    piece_value = get_single_field_value(
        section.fields,
        "PieceCoordinates",
    )

    if piece_value is None:
        raise ValueError(
            f"ZValue {z_value} has no "
            "PieceCoordinates"
        )

    piece_values = _parse_int_values(
        piece_value,
        expected_count=3,
        field_name="PieceCoordinates",
    )

    # We know this has exactly three entries because the parser above checked
    # expected_count=3. Naming them explicitly makes their meaning obvious.
    piece_coordinates = (
        piece_values[0],
        piece_values[1],
        piece_values[2],
    )

    # AlignedPieceCoords are optional because an MDOC may describe a montage
    # for which no solved alignment has been stored.
    aligned_value = get_single_field_value(
        section.fields,
        "AlignedPieceCoords",
    )

    if aligned_value is None:
        aligned_piece_coordinates = None
    else:
        aligned_values = _parse_int_values(
            aligned_value,
            expected_count=3,
            field_name="AlignedPieceCoords",
        )

        aligned_piece_coordinates = (
            aligned_values[0],
            aligned_values[1],
            aligned_values[2],
        )

    # StagePosition gives the physical microscope stage location in x/y.
    # it is useful later for connecting montage/image geometry to microscope
    # coordinates, but it is not required merely to place tiles in a raster.
    stage_value = get_single_field_value(
        section.fields,
        "StagePosition",
    )

    if stage_value is None:
        stage_position = None
    else:
        stage_values = _parse_float_values(
            stage_value,
            expected_count=2,
            field_name="StagePosition",
        )

        stage_position = (
            stage_values[0],
            stage_values[1],
        )

    # at this point we have crossed from raw MDOC strings into a typed scientific description of this one montage tile.
    return MontagePieceMetadata(
        z_value=z_value,
        piece_coordinates=piece_coordinates,
        aligned_piece_coordinates=(
            aligned_piece_coordinates
        ),
        stage_position=stage_position,
    )
    
def extract_montage_metadata(
    mdoc: MdocFile,
) -> MontageMetadata:
    """
    Interpret a parsed MDOC as one complete SerialEM montage.

    Input:
        MdocFile produced by parse_mdoc().

    Output:
        MontageMetadata containing typed piece geometry, tile dimensions,
        nominal montage dimensions, physical pixel spacing, grid dimensions,
        and the full aligned bounding rectangle.

    This function does not load or reconstruct image pixels. Its job is only
    to answer: "What is the geometry of this montage?"
    """

    # ------------------------------------------------------------------
    # 1. Read the dimensions of one source tile.
    #
    # SerialEM stores ImageSize in x/y order:
    #
    #     ImageSize = width height
    #
    # For this dataset:
    #
    #     ImageSize = 2748 1936
    #
    # This is intentionally kept as (x_size, y_size). A NumPy image later
    # uses the opposite shape convention: (rows, columns) = (y_size, x_size).
    # ------------------------------------------------------------------
    image_size_value = get_single_field_value(
        mdoc.global_fields,
        "ImageSize",
    )

    if image_size_value is None:
        raise ValueError(
            "MDOC has no global ImageSize field"
        )

    image_size_values = _parse_int_values(
        image_size_value,
        expected_count=2,
        field_name="ImageSize",
    )

    tile_size_xy = (
        image_size_values[0],
        image_size_values[1],
    )

    tile_width, tile_height = tile_size_xy

    # ------------------------------------------------------------------
    # 2. Read the physical pixel spacing.
    # The raw file stores this as text:
    #
    #     PixelSpacing = 2202.2
    #
    # SerialEM reports this value in Angstroms per pixel for this dataset.
    # We preserve the unit explicitly in the variable and dataclass name.
    # ------------------------------------------------------------------
    pixel_spacing_value = get_single_field_value(
        mdoc.global_fields,
        "PixelSpacing",
    )

    if pixel_spacing_value is None:
        raise ValueError(
            "MDOC has no global PixelSpacing field"
        )

    pixel_spacing_values = _parse_float_values(
        pixel_spacing_value,
        expected_count=1,
        field_name="PixelSpacing",
    )

    pixel_spacing_angstrom = pixel_spacing_values[0]

    # ------------------------------------------------------------------
    # 3. Convert every [ZValue = ...] section into typed piece metadata.
    #
    # Other MDOC sections such as [T = ...] and [MontSection = ...]
    # describe something different, so they are not montage image pieces.
    # ------------------------------------------------------------------
    piece_sections = tuple(
        section
        for section in mdoc.sections
        if section.section_type == "ZValue"
    )

    if not piece_sections:
        raise ValueError(
            "MDOC contains no ZValue image sections"
        )

    pieces = tuple(
        _montage_piece_from_section(section)
        for section in piece_sections
    )

    # sort the pieces acccounding to the z value. basicaly does kinda nothing here cause mdoc is sort but just in case for future.
    pieces = tuple(
        sorted(
            pieces,
            key=lambda piece: piece.z_value,
        )
    )

    # ------------------------------------------------------------------
    # 4. Determine the nominal grid dimensions.
    #
    # PieceCoordinates describe the planned regular montage positions.
    #
    # In the example file dataset the distinct X positions are:
    #
    #     0, 2316, 4632, 6948
    #
    # and the distinct Y positions are:
    #
    #     0, 1504, 3008, 4512, 6016, 7520
    #
    # giving a 4 x 6 grid.
    # loops though all but keeps on the distinct ones becuase of {} set
    # ------------------------------------------------------------------
    nominal_x_positions = {  
        piece.piece_coordinates[0]
        for piece in pieces
    }

    nominal_y_positions = {
        piece.piece_coordinates[1]
        for piece in pieces
    }

    grid_size_xy = (
        len(nominal_x_positions),
        len(nominal_y_positions),
    )

    # ------------------------------------------------------------------
    # 5. Read SerialEM's nominal full montage size.
    #
    # This is stored in [MontSection = ...], not in the global fields or
    # individual ZValue sections.
    # ------------------------------------------------------------------
    montage_sections = tuple(
        section
        for section in mdoc.sections
        if section.section_type == "MontSection"
    )

    if len(montage_sections) != 1:
        raise ValueError(
            "Expected exactly one MontSection, "
            f"found {len(montage_sections)}"
        )

    full_size_value = get_single_field_value(
        montage_sections[0].fields,
        "FullMontSize",
    )

    if full_size_value is None:
        raise ValueError(
            "MontSection has no FullMontSize"
        )

    full_size_values = _parse_int_values(
        full_size_value,
        expected_count=2,
        field_name="FullMontSize",
    )

    nominal_size_xy = (
        full_size_values[0],
        full_size_values[1],
    )

    # ------------------------------------------------------------------
    # 6. Determine whether aligned coordinates are available for every tile.
    # A partially aligned montage would be ambiguous for reconstruction: We therefore only construct aligned
    # montage geometry when every piece has AlignedPieceCoords.
    # ------------------------------------------------------------------
    aligned_coordinates = tuple(
        piece.aligned_piece_coordinates
        for piece in pieces
        if piece.aligned_piece_coordinates is not None
    )

    if len(aligned_coordinates) == 0:
        # No solved alignment is available. The nominal geometry is still
        # valid, but there is no aligned coordinate system to describe.
        aligned_origin_xy = None
        aligned_size_xy = None

    elif len(aligned_coordinates) != len(pieces):
        # Some pieces have aligned coordinates and some do not. Mixing them
        # would create a montage whose geometry has no clear meaning.
        raise ValueError(
            "AlignedPieceCoords are present for only "
            f"{len(aligned_coordinates)} of "
            f"{len(pieces)} montage pieces"
        )

    else:
        # --------------------------------------------------------------
        # 7. Find the minimum aligned x/y coordinate.
        #
        # These values define the origin of the solved montage coordinate
        # system. They are allowed to be negative.
        #
        # For 82map_2:
        #
        #     min_x = -29
        #     min_y = -60
        # --------------------------------------------------------------
        min_x = min(
            coordinate[0]
            for coordinate in aligned_coordinates
        )

        min_y = min(
            coordinate[1]
            for coordinate in aligned_coordinates
        )

        aligned_origin_xy = (
            min_x,
            min_y,
        )

        # --------------------------------------------------------------
        # 8. Find the farthest right and top tile edges.
        #
        # AlignedPieceCoords gives the starting coordinate of each tile.
        # To get its outer edge we must add the source tile dimensions.
        #
        # Example:
        #
        #     tile starts at x = 6974
        #     tile width       = 2748
        #
        #     right edge       = 9722
        # --------------------------------------------------------------
        max_right = max(
            coordinate[0] + tile_width
            for coordinate in aligned_coordinates
        )

        max_top = max(
            coordinate[1] + tile_height
            for coordinate in aligned_coordinates
        )

        # --------------------------------------------------------------
        # 9. Convert the aligned bounding rectangle into raster dimensions.
        # this is basically shifiting the origin
        #
        # Width is not simply max_right because the coordinate system may
        # begin at a negative value.
        # For this dataset:
        #
        #     width = 9722 - (-29) = 9751
        #
        # The same logic applies vertically.
        # --------------------------------------------------------------
        aligned_size_xy = (
            max_right - min_x,
            max_top - min_y,
        )

    # ------------------------------------------------------------------
    # 10. Package everything into one immutable scientific description.
    #
    # Reconstruction code can now consume this object without knowing how
    # MDOC text is organized internally.
    # ------------------------------------------------------------------
    return MontageMetadata(
        pieces=pieces,
        tile_size_xy=tile_size_xy,
        nominal_size_xy=nominal_size_xy,
        grid_size_xy=grid_size_xy,
        pixel_spacing_angstrom=(
            pixel_spacing_angstrom
        ),
        aligned_origin_xy=aligned_origin_xy,
        aligned_size_xy=aligned_size_xy,
    )