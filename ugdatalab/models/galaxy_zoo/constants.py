"""Galaxy Zoo 2 constants: label names, descriptive names, and hierarchical tree.

Label names and descriptive names follow Willett et al. 2013, Table 2.
The hierarchical tree encodes parent-child dependencies between classification
questions in the Galaxy Zoo 2 decision tree.
"""

N_LABELS = 37

LABEL_COLUMNS = [
    "Class1.1", "Class1.2", "Class1.3",
    "Class2.1", "Class2.2",
    "Class3.1", "Class3.2",
    "Class4.1", "Class4.2",
    "Class5.1", "Class5.2", "Class5.3", "Class5.4",
    "Class6.1", "Class6.2",
    "Class7.1", "Class7.2", "Class7.3",
    "Class8.1", "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6",
    "Class8.7",
    "Class9.1", "Class9.2", "Class9.3",
    "Class10.1", "Class10.2", "Class10.3",
    "Class11.1", "Class11.2", "Class11.3", "Class11.4", "Class11.5",
    "Class11.6",
]

LABEL_DESCRIPTIVE = {
    "Class1.1": "Smooth",
    "Class1.2": "Features/Disk",
    "Class1.3": "Star/Artifact",
    "Class2.1": "Edge-on: Yes",
    "Class2.2": "Edge-on: No",
    "Class3.1": "Bar",
    "Class3.2": "No bar",
    "Class4.1": "Spiral: Yes",
    "Class4.2": "Spiral: No",
    "Class5.1": "No bulge",
    "Class5.2": "Just noticeable bulge",
    "Class5.3": "Obvious bulge",
    "Class5.4": "Dominant bulge",
    "Class6.1": "Odd: Yes",
    "Class6.2": "Odd: No",
    "Class7.1": "Completely round",
    "Class7.2": "In between",
    "Class7.3": "Cigar-shaped",
    "Class8.1": "Odd: Ring",
    "Class8.2": "Odd: Lens/Arc",
    "Class8.3": "Odd: Disturbed",
    "Class8.4": "Odd: Irregular",
    "Class8.5": "Odd: Other",
    "Class8.6": "Odd: Merger",
    "Class8.7": "Odd: Dust lane",
    "Class9.1": "Rounded bulge",
    "Class9.2": "Boxy bulge",
    "Class9.3": "No bulge",
    "Class10.1": "Spiral: Tight",
    "Class10.2": "Spiral: Medium",
    "Class10.3": "Spiral: Loose",
    "Class11.1": "Spiral: 1 arm",
    "Class11.2": "Spiral: 2 arms",
    "Class11.3": "Spiral: 3 arms",
    "Class11.4": "Spiral: 4 arms",
    "Class11.5": "Spiral: 5+ arms",
    "Class11.6": "Spiral: Can't tell",
}

LABEL_LATEX = {
    col: rf"\textrm{{{desc}}}"
    for col, desc in LABEL_DESCRIPTIVE.items()
}

MERGER_LABEL = "Class8.6"

# The 7 labels used for Task 25 extreme-example analysis (from lab manual)
PROTOTYPE_LABELS = [
    "Class1.1",   # smooth
    "Class1.3",   # star/artifact
    "Class2.1",   # edge-on disk
    "Class8.1",   # odd: ring
    "Class8.2",   # odd: lens/arc
    "Class11.2",  # spiral: 2 arms
    "Class8.6",   # odd: merger
]

# Hierarchical label tree: parent question → child question columns.
# A child question is only asked if the parent answer exceeds a threshold.
# This encodes the Galaxy Zoo 2 decision tree structure.
LABEL_TREE = {
    # Q1: Smooth, Features/Disk, or Star?
    "Class1.1": ["Class7.1", "Class7.2", "Class7.3"],          # smooth → Q7 (roundness)
    "Class1.2": ["Class2.1", "Class2.2"],                       # features → Q2 (edge-on?)
    # Q2: Edge-on?
    "Class2.1": ["Class9.1", "Class9.2", "Class9.3"],          # edge-on yes → Q9 (bulge shape)
    "Class2.2": ["Class3.1", "Class3.2"],                       # edge-on no → Q3 (bar?)
    # Q3: Bar? → Q4 (spiral?)
    "Class3.1": ["Class4.1", "Class4.2"],
    "Class3.2": ["Class4.1", "Class4.2"],
    # Q4: Spiral?
    "Class4.1": ["Class10.1", "Class10.2", "Class10.3",        # spiral yes → Q10 (tightness)
                  "Class11.1", "Class11.2", "Class11.3",        # and Q11 (arm count)
                  "Class11.4", "Class11.5", "Class11.6"],
    # Q4/Q3 → Q5 (bulge prominence)
    "Class4.2": ["Class5.1", "Class5.2", "Class5.3", "Class5.4"],
    # Q6: Odd? (asked for all non-star)
    "Class6.1": ["Class8.1", "Class8.2", "Class8.3", "Class8.4",
                  "Class8.5", "Class8.6", "Class8.7"],          # odd yes → Q8 (odd features)
}

_GALAXY_ZOO_SCHEMA = {
    int: ["GalaxyID"],
    float: LABEL_COLUMNS.copy(),
}
