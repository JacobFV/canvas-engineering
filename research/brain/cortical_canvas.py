"""Cortical brain architecture as canvas types with real cortical wiring.

Maps Destrieux atlas ROIs to canvas regions organized by functional
networks (visual, auditory, language, frontal, default mode, subcortical).
Connections mirror known cortical pathways: ventral/dorsal visual streams,
language pathway (A1 -> Wernicke -> Broca), default mode network,
frontal executive control, and cross-modal integration.

Usage:
    from research.brain.cortical_canvas import build_cortical_program

    bound, program, region_map = build_cortical_program(d_model=128)
    # region_map maps ROI friendly names -> canvas region paths
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Ensure canvas_engineering is importable
_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from canvas_engineering import (
    Field,
    Connection,
    CanvasTopology,
    ConnectivityPolicy,
    compile_program,
    CanvasProgram,
    RegionProgram,
    ConnectionProgram,
    ClockSpec,
    RegionScheduler,
    AttentionDispatcher,
    ResidualAccumulator,
    ResidualSpec,
)
from canvas_engineering.schema import CanvasSchema


# ---- ROI -> Canvas region mapping (from Destrieux atlas) ----
# These match the ROI_LABEL_MAP in /Users/jacob/fun/brain-model/core/roi.py

ROI_TO_CANVAS = {
    "Visual (V1/V2)":     "visual.v1",
    "Occipital":          "visual.v2_v4",
    "Fusiform (FFA)":     "visual.fusiform",
    "Auditory (A1)":      "auditory.a1",
    "Wernicke's area":    "auditory.wernicke",
    "Broca's area":       "language.broca",
    "Angular/TPJ":        "language.angular",
    "Temporal mid.":      "language.temporal_mid",
    "Frontal sup.":       "frontal.prefrontal",
    "Frontal mid.":       "frontal.premotor",
    "Motor":              "frontal.motor",
    "Precuneus":          "default_mode.precuneus",
    "Cingulate ant.":     "default_mode.cingulate",
    "Cingulate post.":    "default_mode.cingulate",
    "Temporal pole":      "default_mode.temporal_pole",
    "Insula":             "subcortical.insula",
    "Somatosensory":      "subcortical.somatosensory",
    "Parahipp. (PPA)":    "visual.fusiform",
    "Temporal inf.":      "language.temporal_mid",
    "Orbital frontal":    "frontal.prefrontal",
}

# Reverse: canvas path -> list of ROI names that map to it
CANVAS_TO_ROIS: Dict[str, List[str]] = {}
for roi, path in ROI_TO_CANVAS.items():
    CANVAS_TO_ROIS.setdefault(path, []).append(roi)


# ---- Type hierarchy mirroring cortical organization ----

@dataclass
class VisualCortex:
    v1: Field = Field(
        4, 4, family="observation",
        semantic_type="primary visual cortex V1/V2 calcarine",
    )
    v2_v4: Field = Field(
        3, 3, family="state", tags=("belief",),
        semantic_type="extrastriate visual V2/V4 occipital",
    )
    fusiform: Field = Field(
        2, 2, family="state", tags=("object",),
        semantic_type="fusiform face area FFA ventral temporal",
    )


@dataclass
class AuditoryCortex:
    a1: Field = Field(
        3, 3, family="observation",
        semantic_type="primary auditory cortex A1 Heschl gyrus",
    )
    wernicke: Field = Field(
        3, 3, family="state", tags=("belief",),
        semantic_type="Wernicke area superior temporal comprehension",
    )


@dataclass
class LanguageNetwork:
    broca: Field = Field(
        3, 3, family="state", tags=("parser",),
        semantic_type="Broca area inferior frontal language production",
    )
    angular: Field = Field(
        2, 2, family="state", tags=("belief",),
        semantic_type="angular gyrus TPJ semantic integration",
    )
    temporal_mid: Field = Field(
        2, 2, family="state",
        semantic_type="middle temporal gyrus lexical semantics",
    )


@dataclass
class FrontalCortex:
    prefrontal: Field = Field(
        3, 3, family="state", tags=("belief", "goal"),
        semantic_type="prefrontal cortex executive control working memory",
    )
    motor: Field = Field(
        3, 3, family="action",
        semantic_type="primary motor cortex M1 precentral",
    )
    premotor: Field = Field(
        2, 2, family="state", tags=("planning",),
        semantic_type="premotor dorsal PMd motor planning",
    )


@dataclass
class DefaultModeNetwork:
    precuneus: Field = Field(
        2, 2, family="state", tags=("self",),
        semantic_type="precuneus self-referential episodic memory",
    )
    cingulate: Field = Field(
        2, 2, family="state",
        semantic_type="cingulate cortex conflict monitoring salience",
    )
    temporal_pole: Field = Field(
        1, 2, family="memory", tags=("semantic",),
        semantic_type="temporal pole semantic memory social concepts",
    )


@dataclass
class SubcorticalSystems:
    insula: Field = Field(
        2, 2, family="state", tags=("interoception",),
        semantic_type="insular cortex interoception emotion awareness",
    )
    somatosensory: Field = Field(
        2, 2, family="observation",
        semantic_type="primary somatosensory S1 postcentral tactile",
    )


@dataclass
class CorticalBrain:
    visual: VisualCortex = dc_field(default_factory=VisualCortex)
    auditory: AuditoryCortex = dc_field(default_factory=AuditoryCortex)
    language: LanguageNetwork = dc_field(default_factory=LanguageNetwork)
    frontal: FrontalCortex = dc_field(default_factory=FrontalCortex)
    default_mode: DefaultModeNetwork = dc_field(default_factory=DefaultModeNetwork)
    subcortical: SubcorticalSystems = dc_field(default_factory=SubcorticalSystems)
    prediction_error: Field = Field(
        2, 4, family="residual",
        semantic_type="global prediction error tracking",
    )


# ---- Known cortical pathways as explicit connections ----

# Each tuple: (src_canvas_path, dst_canvas_path, operator, weight)
CORTICAL_PATHWAYS: List[Tuple[str, str, str, float]] = [
    # Ventral visual stream: V1 -> V2/V4 -> fusiform (object recognition)
    ("visual.v1", "visual.v2_v4", "predict", 1.0),
    ("visual.v2_v4", "visual.fusiform", "predict", 1.0),
    ("visual.fusiform", "visual.v2_v4", "correct", 0.5),
    ("visual.v2_v4", "visual.v1", "correct", 0.5),

    # Dorsal visual stream: V1 -> angular (parietal) -> premotor (action)
    ("visual.v1", "language.angular", "predict", 0.8),
    ("language.angular", "frontal.premotor", "predict", 0.8),

    # Language pathway: A1 -> Wernicke -> Broca (comprehension -> production)
    ("auditory.a1", "auditory.wernicke", "observe", 1.0),
    ("auditory.wernicke", "language.broca", "predict", 1.0),
    ("language.broca", "auditory.wernicke", "correct", 0.5),

    # Semantic integration: Wernicke -> angular -> temporal_mid
    ("auditory.wernicke", "language.angular", "integrate", 0.7),
    ("language.angular", "language.temporal_mid", "integrate", 0.7),
    ("language.temporal_mid", "language.angular", "integrate", 0.7),

    # Default mode network: precuneus <-> cingulate <-> temporal_pole
    ("default_mode.precuneus", "default_mode.cingulate", "integrate", 0.8),
    ("default_mode.cingulate", "default_mode.precuneus", "integrate", 0.8),
    ("default_mode.cingulate", "default_mode.temporal_pole", "integrate", 0.8),
    ("default_mode.temporal_pole", "default_mode.cingulate", "integrate", 0.8),
    ("default_mode.precuneus", "default_mode.temporal_pole", "integrate", 0.6),
    ("default_mode.temporal_pole", "default_mode.precuneus", "retrieve", 0.6),

    # Frontal control: prefrontal -> premotor -> motor (goal -> plan -> act)
    ("frontal.prefrontal", "frontal.premotor", "predict", 1.0),
    ("frontal.premotor", "frontal.motor", "act", 1.0),
    ("frontal.motor", "frontal.premotor", "correct", 0.5),
    ("frontal.premotor", "frontal.prefrontal", "correct", 0.5),

    # Cross-modal: visual -> language
    ("visual.fusiform", "language.temporal_mid", "predict", 0.7),
    ("visual.v2_v4", "language.angular", "predict", 0.6),

    # Cross-modal: auditory -> language
    ("auditory.wernicke", "language.temporal_mid", "predict", 0.7),

    # Frontal executive reads from everywhere
    ("frontal.prefrontal", "visual.v2_v4", "attend", 0.5),
    ("frontal.prefrontal", "auditory.wernicke", "attend", 0.5),
    ("frontal.prefrontal", "language.broca", "attend", 0.6),
    ("frontal.prefrontal", "default_mode.precuneus", "attend", 0.4),

    # Insular cortex: interoception -> cingulate, prefrontal
    ("subcortical.insula", "default_mode.cingulate", "observe", 0.6),
    ("subcortical.insula", "frontal.prefrontal", "observe", 0.5),

    # Somatosensory -> motor (sensorimotor loop)
    ("subcortical.somatosensory", "frontal.motor", "observe", 0.8),
    ("frontal.motor", "subcortical.somatosensory", "predict", 0.5),

    # Prediction error connections (all state/obs regions emit to residual)
    ("visual.v1", "prediction_error", "emit_residual", 0.3),
    ("visual.v2_v4", "prediction_error", "emit_residual", 0.3),
    ("auditory.a1", "prediction_error", "emit_residual", 0.3),
    ("auditory.wernicke", "prediction_error", "emit_residual", 0.3),
    ("frontal.prefrontal", "prediction_error", "emit_residual", 0.3),

    # Memory retrieval: temporal_pole -> fusiform, angular
    ("default_mode.temporal_pole", "visual.fusiform", "retrieve", 0.5),
    ("default_mode.temporal_pole", "language.angular", "retrieve", 0.5),
]


def _build_cortical_connections(bound_field_names: List[str]) -> List[Connection]:
    """Build Connection objects for known cortical pathways.

    Only includes connections where both src and dst exist in the
    compiled schema's field names.
    """
    valid_names = set(bound_field_names)
    connections = []
    for src, dst, operator, weight in CORTICAL_PATHWAYS:
        if src in valid_names and dst in valid_names:
            connections.append(Connection(
                src=src, dst=dst,
                operator=operator,
                weight=weight,
            ))
    return connections


def build_cortical_program(
    T: int = 4,
    d_model: int = 128,
) -> Tuple["BoundSchema", CanvasProgram, Dict[str, str]]:
    """Build the full cortical canvas with real wiring.

    Returns:
        (bound_schema, program, roi_to_canvas_map)
        - bound_schema: BoundSchema with layout and topology
        - program: CanvasProgram with region programs and connections
        - roi_to_canvas_map: dict mapping ROI friendly names to canvas paths
    """
    brain = CorticalBrain()

    # Compile with isolated intra-connectivity (we add explicit cortical
    # connections instead of dense auto-wiring within each sub-network)
    connectivity = ConnectivityPolicy(
        intra="isolated",
        array_element="isolated",
        temporal="dense",
    )

    bound, program = compile_program(
        brain,
        T=T,
        d_model=d_model,
        connectivity=connectivity,
    )

    # Add explicit cortical pathway connections
    cortical_conns = _build_cortical_connections(bound.field_names)

    # Merge with existing topology connections
    existing_conns = []
    if program.schema.topology:
        existing_conns = list(program.schema.topology.connections)

    all_conns = existing_conns + cortical_conns

    # Rebuild schema with merged connections
    merged_topology = CanvasTopology(connections=all_conns)
    merged_schema = CanvasSchema(
        layout=program.schema.layout,
        topology=merged_topology,
        version=program.schema.version,
        metadata=program.schema.metadata,
    )

    # Rebuild program with merged schema
    program = CanvasProgram(
        schema=merged_schema,
        regions=program.regions,
        connections=program.connections,
        version=program.version,
    )

    # Also update the bound schema reference
    bound.schema = merged_schema

    return bound, program, ROI_TO_CANVAS


def get_region_names() -> List[str]:
    """Return all canvas region paths for the cortical brain."""
    brain = CorticalBrain()
    bound, _ = compile_program(brain, T=1, d_model=64)
    return bound.field_names


# ---- Stimulus categories ----
# Extended from virtual_eeg with motor, visual, emotion categories

STIMULUS_CATEGORIES = {
    "animal": [
        "A golden retriever bounding across a field, ears flapping, tongue out, chasing a frisbee through the grass",
        "An eagle circling high above a mountain valley, wings barely moving, riding the thermals in silence",
        "A cat curled up on a warm windowsill, purring, eyes half closed, watching rain slide down the glass",
        "A whale breaching the surface in a spray of white water, crashing back down with a thunderous splash",
        "A spider spinning a web between two branches, each thread catching the morning dew like tiny diamonds",
        "A horse galloping along a beach at sunset, hooves splashing through shallow waves, mane flying",
        "A swarm of fireflies blinking in a dark meadow, each one a tiny beacon drifting through the warm night air",
        "Two wolves howling together on a ridge under a full moon, their voices carrying across the frozen valley",
    ],
    "music": [
        "A solo violin playing a haunting melody in an empty cathedral, each note echoing off ancient stone walls",
        "Heavy drums pounding a tribal rhythm, the beat so deep you feel it vibrating in your chest and teeth",
        "A jazz piano improvising over a walking bass line, notes tumbling out in unexpected cascading runs",
        "An orchestra building toward a massive crescendo, every instrument adding to a wall of overwhelming sound",
        "A guitar playing fingerpicked arpeggios by a campfire, gentle and warm, the wood crackling between phrases",
        "Electronic music with a deep bass drop, the sub-frequencies shaking the floor of a dark nightclub",
        "A choir singing in perfect harmony, voices layered so tightly they seem to become a single instrument",
        "A music box playing a simple lullaby, the tiny metallic notes pinging in a quiet dark room",
    ],
    "danger": [
        "A rattlesnake coiled on the path, its rattle buzzing as you freeze mid-step, heart suddenly pounding",
        "The car ahead swerves and you slam the brakes, tires screaming on wet asphalt, everything in slow motion",
        "Smoke pouring under the hotel room door at three in the morning, the fire alarm shrieking overhead",
        "A crack in the ice shooting across the frozen lake beneath your feet, the groan of something about to give",
        "Lightning striking a tree twenty feet away with a deafening crack, the air tasting of ozone and fear",
        "The elevator cable snapping, the sudden lurch downward, the lights flickering, a moment of weightlessness",
        "A massive wave rising behind the boat, dark green and curling, blocking out the entire horizon",
        "Walking through a dark alley and hearing footsteps behind you that match your pace exactly",
    ],
    "spatial": [
        "Standing at the edge of the Grand Canyon, looking down a thousand feet of layered red stone to the river",
        "A cathedral ceiling soaring a hundred feet overhead, light filtering through stained glass in colored beams",
        "Floating in the middle of a dark ocean at night, no land visible in any direction, stars reflected below",
        "A narrow tunnel deep underground, barely wide enough to crawl through, stone pressing in on all sides",
        "The view from the top of a skyscraper, the city spread out below like a circuit board of lights and streets",
        "An enormous empty warehouse, footsteps echoing, your voice bouncing back from walls you can barely see",
        "A dense forest where the canopy blocks all sunlight, the trunks forming a maze with no visible path",
        "Standing on the wing of a plane on the tarmac, looking out at the flat expanse of runway stretching to the horizon",
    ],
    "social": [
        "Your best friend calling at midnight because they need to talk, and you sit on the kitchen floor and listen",
        "Standing at a podium giving a speech to five hundred people, every single pair of eyes fixed on you",
        "A baby gripping your finger for the first time, tiny hand wrapped tight, looking up at your face",
        "A stranger on the bus smiling at you for no reason, and that smile lifting your whole morning",
        "Two old men on a park bench playing chess in complete silence, decades of friendship needing no words",
        "Walking into a surprise party and thirty people shouting your name, faces you love all in one room",
        "Holding someone while they cry, not saying anything, just being there, their shoulders shaking against yours",
        "A job interview where the panel stares at you in silence for ten seconds after your answer",
    ],
    "language": [
        "Reading a poem where every word is common but the arrangement unlocks a meaning you never had words for before",
        "The word petrichor, meaning the smell of rain on dry earth, a word for something you always knew but could never name",
        "Someone speaking a language you have never heard, melodic and rhythmic, and understanding their meaning from gesture alone",
        "The frustration of knowing exactly what you mean but the right word hovering just beyond your reach, refusing to come",
        "Reading a letter from a hundred years ago and feeling the writer's personality come alive through their handwriting",
        "Two people finishing each other's sentences, their thoughts synchronized, a conversation that flows like one mind",
        "A child making up a word for something that doesn't have one, and the word being somehow perfect",
        "Translating a joke into another language and watching the humor evaporate completely despite accurate translation",
    ],
    "motor": [
        "Swinging a hammer down onto a nail, the satisfying thud as the head sinks flush into the wood grain",
        "Threading a needle under a magnifying glass, fingers trembling, trying to hold absolutely still",
        "Sprinting as fast as you can down a track, arms pumping, legs burning, lungs gasping for air",
        "Playing a fast scale on a piano, each finger landing precisely on the right key in rapid succession",
        "Catching a ball thrown from behind without looking, your hand snapping up at exactly the right moment",
        "Balancing on one foot on a narrow beam, arms out, every muscle making tiny corrections to stay upright",
        "Typing without looking at the keyboard, fingers dancing from key to key, words appearing on screen",
        "Throwing a perfect spiral with a football, the ball rotating tightly as it arcs through the air",
    ],
    "visual": [
        "A sunset painting the sky in layers of orange, pink, and purple, the sun a molten disk touching the horizon",
        "A kaleidoscope of colors shifting and morphing, geometric patterns folding into each other endlessly",
        "Looking through a microscope at a drop of pond water, tiny creatures darting and spinning in all directions",
        "A field of sunflowers stretching to the horizon, every face turned toward the afternoon sun",
        "Neon signs reflecting off wet pavement at night, reds and blues and greens swimming in the puddles",
        "A single candle flame flickering in complete darkness, the light barely reaching the edges of the room",
        "Watching a murmuration of starlings, thousands of birds moving as one liquid shape across the evening sky",
        "A flash of lightning illuminating an entire landscape for a split second, every detail frozen in white light",
    ],
    "emotion": [
        "The crushing weight of grief when you realize someone you love is truly gone and will never come back",
        "The bubbling joy of receiving unexpected good news, a laugh escaping before you can even process it",
        "A slow burn of anger building in your chest as someone cuts in front of you for the third time",
        "The peaceful calm of waking up on a Saturday morning with nothing to do and nowhere to be",
        "A wave of nostalgia washing over you when you smell something from your childhood, vivid and specific",
        "The electric anticipation before opening a letter that could change everything, hands slightly shaking",
        "Deep contentment sitting by a fire with people you love, no one talking, everyone perfectly comfortable",
        "The hollow emptiness of loneliness in a crowded room where everyone is talking but no one to you",
    ],
}


if __name__ == "__main__":
    bound, program, roi_map = build_cortical_program(T=4, d_model=128)
    print(bound.summary())
    print()
    print(program.summary())
    print()
    print(f"ROI -> Canvas mapping ({len(roi_map)} ROIs):")
    for roi, path in sorted(roi_map.items()):
        print(f"  {roi:25s} -> {path}")
    print()
    print(f"Canvas -> ROI mapping:")
    for path, rois in sorted(CANVAS_TO_ROIS.items()):
        print(f"  {path:30s} <- {rois}")
    print()
    if program.schema.topology:
        print(program.schema.topology.summary())
