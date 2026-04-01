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
        "A chameleon creeping along a branch, each foot gripping deliberately, its turret eyes swiveling independently to scan for insects in the dense foliage",
        "A pod of dolphins leaping in synchronized arcs through turquoise water, their sleek bodies catching the sunlight before slicing back under the surface",
        "A barn owl swooping silently through a moonlit field, talons extended, face disk funneling the faintest rustle of a vole beneath the grass",
        "A massive grizzly bear standing upright on a riverbank, water dripping from its fur, a silver salmon clamped in its jaws",
        "An octopus changing color and texture in an instant, rippling from rough brown to smooth white as it flows across the coral reef floor",
        "A column of army ants marching across the jungle floor, carrying leaf fragments many times their size, an endless living conveyor belt",
        "A hummingbird hovering at a trumpet flower, wings beating so fast they blur into a silver halo, its iridescent throat flashing ruby red",
        "A tortoise lumbering across sun-baked earth, its ancient shell scarred and mossy, each step slow and deliberate and impossibly patient",
        "A rattling kingfisher diving headfirst into a still pond, emerging a second later with a wriggling minnow pinched in its long orange beak",
        "A pride of lions lazing under an acacia tree in the midday heat, tawny fur blending with dry grass, tails flicking at flies",
        "A jellyfish pulsing through dark ocean water, its translucent bell trailing long luminescent tentacles that glow faint blue in the abyss",
        "A red fox trotting through fresh snow at dawn, each paw placed precisely in the track of the one before, breath visible in the cold air",
        "A gorilla sitting in a patch of jungle sunlight, cradling an infant against its broad chest, enormous dark fingers grooming the baby gently",
        "A peacock fanning its tail feathers into a shimmering wall of iridescent blue and green eyes, rattling the quills in a display of trembling color",
        "A gecko clinging upside down to a glass ceiling, its padded toes splayed out, flicking its tongue to taste the humid tropical air",
        "A swarm of monarch butterflies blanketing a pine tree in Mexico, thousands of orange wings opening and closing slowly like breathing stained glass",
        "A crocodile lying perfectly still in murky water with only its nostrils and eyes visible, a living fossil waiting with infinite patience",
        "A parrot cracking open a walnut with its powerful curved beak, turning the nut deftly with one scaly foot while shell fragments fall away",
        "A seal pup calling for its mother on a rocky shore, its white fur matted with spray, dark eyes scanning the churning surf",
        "A praying mantis swaying on a leaf, triangular head rotating to track a fly, forelegs folded like jackknives ready to snap forward in milliseconds",
        "A school of tropical fish swirling around a coral head in a flash of yellow and electric blue, moving as one shimmering cloud",
        "A moose wading chest-deep through a misty lake at dawn, water streaming from its massive rack of antlers, steam rising from its dark back",
        "A bat erupting from a cave at dusk with ten thousand others, the colony forming a swirling black ribbon against the fading violet sky",
        "A hermit crab scuttling across tidal rocks, its borrowed shell bobbing, antennae probing the salt air as it searches for a larger home among the barnacles",
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
        "A cellist drawing the bow across the lowest string, a rich dark tone that resonates in your ribcage like a second heartbeat",
        "Steel drums ringing out a calypso melody on a beach at sunset, each bright metallic ping hanging in the warm salty air",
        "A distorted electric guitar wailing a blues solo, the notes bending and sustaining with raw grit through a cranked amplifier",
        "Wind chimes tinkling on a porch in a gentle breeze, random silver notes overlapping into an accidental melody that never repeats",
        "A tabla player's fingers flying across the drumheads, producing rapid-fire patterns of sharp taps and deep resonant booms",
        "An opera singer holding a high note that fills the entire theater, the vibrato shimmering like heat rising off summer pavement",
        "A street musician playing accordion on a cobblestone corner, the wheezing bellows pumping out a bittersweet waltz for passersby",
        "The low moan of a didgeridoo vibrating through the ground, circular breathing creating an unbroken drone that seems to come from the earth itself",
        "A marching band approaching from two blocks away, the snare drums crisp and the brass swelling louder with each step closer",
        "A harpist plucking glissandos that cascade like water falling down glass stairs, each string releasing a shimmering tone into the quiet hall",
        "The scratchy warmth of a vinyl record spinning on a turntable, the needle riding grooves, faint crackles adding texture to a old jazz standard",
        "A flamenco guitarist attacking the strings with furious rasgueados, the percussive strumming building to a frenzy of rhythm and passion",
        "Tibetan singing bowls being struck in sequence, their overtones layering into a shimmering cloud of metallic harmonics that pulse and interfere",
        "A saxophone player on a fire escape at midnight, the breathy vibrato floating down to the empty street below like liquid gold",
        "Children singing a round in a school gymnasium, their voices staggered and overlapping, the melody chasing itself in joyful circles",
        "A pipe organ filling a stone church with a thundering chord, the bass pipes shaking the wooden pews while the high pipes pierce the air",
        "The rhythmic click and strum of a flamenco dancer's heels and a guitar in perfect lockstep, building speed in furious unison",
        "A theremin producing an eerie wavering tone without being touched, the player's hands sculpting pitch and volume from thin air",
        "A string quartet playing a slow adagio, four instruments breathing together, the harmonies so tight the sound seems to hover in the room",
        "Raindrops falling on a tin roof in varying rhythms, nature's own percussion piece accelerating from scattered taps to a steady roar",
        "A sitar unspooling a long raga in a dim room, the sympathetic strings buzzing and droning beneath the melody like a hive of golden bees",
        "A beatboxer layering kicks, snares, and hi-hats with only their mouth, the sounds so precise they are indistinguishable from a drum machine",
        "A glass harmonica spinning under wet fingertips, producing ethereal tones that seem to come from nowhere and everywhere at once",
        "A banjo picking a rapid bluegrass breakdown, the bright twangy notes tumbling over each other in a joyful stampede of rhythm and speed",
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
        "A tornado siren wailing across a flat prairie, the sky turning sickly green, hail beginning to bounce off the windshield like gravel",
        "The bridge swaying violently underfoot in high wind, cables groaning, the river far below churning brown and fast between concrete pillars",
        "A dog growling low and baring its teeth three feet from your face, hackles raised, every muscle coiled and ready to lunge",
        "Waking to the sound of breaking glass downstairs in the middle of the night, lying frozen, ears straining in the pitch dark",
        "The ground shaking beneath your feet in an earthquake, books falling from shelves, the doorframe cracking, the floor tilting impossibly",
        "A bee sting on your neck followed by swelling tightness in your throat, each breath becoming harder, the panic rising with the closing airway",
        "Driving on a mountain road with no guardrail, the shoulder crumbling inches from your tires, a sheer thousand-foot drop on the passenger side",
        "A gas stove burner left on without a flame, the sharp chemical smell of mercaptan filling the kitchen, a faint hissing from the range",
        "The pilot announcing severe turbulence as the plane drops suddenly, overhead bins popping open, passengers gasping, the oxygen masks deploying",
        "Swimming in open water and feeling something large brush against your leg from below, unseen and unknown in the murky green darkness",
        "A tree cracking and beginning to fall directly toward you, the slow-motion splintering of wood, the shadow growing over your frozen body",
        "Stepping onto a ledge that gives way, the sickening lurch as your foot finds nothing, arms windmilling, gravel cascading into the void below",
        "A flash flood surging around a canyon corner, a wall of brown water carrying logs and debris, the roar reaching you before the water does",
        "The unmistakable rattle of a machine gun in the distance getting closer, the ground kicking up dust in a line advancing toward your position",
        "A black bear standing on its hind legs ten yards ahead on the trail, sniffing the air, its small dark eyes locking onto yours",
        "The elevator doors opening to reveal a flooded hallway, water ankle deep and rising, sparking wires hanging from the ceiling above",
        "A boulder rolling loose on the hillside above, picking up speed and bouncing toward the campsite, each impact shaking the ground harder",
        "Standing in waist-deep water during a flood, the current pulling harder with every second, your fingers slipping from the fence post you cling to",
        "The crack of thin ice spreading in a spiderweb pattern beneath your weight as you stand in the middle of a frozen pond alone",
        "Smelling smoke in the airplane cabin at thirty thousand feet, passengers looking around wide-eyed, the flight attendants moving quickly toward the cockpit",
        "A power line snapping in a storm and whipping down to the wet street, throwing blue sparks, blocking the only path to safety",
        "Realizing the rope is fraying as you hang halfway down a cliff face, threads popping one by one, the harness shifting under your weight",
        "A riptide pulling you steadily out to sea despite your strongest swimming, the shore getting smaller, your arms growing heavy with exhaustion",
        "The warning siren of a tsunami echoing across a coastal town, the ocean retreating impossibly far from the shore, exposing bare seabed",
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
        "A spiral staircase winding upward inside a lighthouse, the walls curving away, each step narrower, the top impossibly far above your head",
        "Lying on your back in a vast open desert at night, the Milky Way arching overhead, the ground flat and featureless to every horizon",
        "A hall of mirrors where reflections of reflections stretch into infinity, your own image repeating and shrinking into an endless tunnel of glass",
        "Standing at the bottom of a deep well looking up at a small circle of blue sky, the stone walls rising close and damp around you",
        "The interior of a hot air balloon basket drifting over a patchwork of green fields, farmhouses shrinking to toys, roads becoming thin white threads",
        "A frozen waterfall rising vertically above you, the blue-white ice pillars towering stories high, the rock face behind it barely visible through the translucence",
        "A long suspension bridge disappearing into fog at both ends, the cables swooping down and back up, the center of the span invisible and swaying",
        "The mouth of an enormous cave opening before you, the ceiling vaulting upward into blackness, the floor sloping down toward a distant drip of water",
        "A rooftop garden on the hundredth floor, the wind whipping past, neighboring skyscrapers close enough to see people in their windows, streets like canyons below",
        "A submarine viewport showing the ocean floor at crushing depth, the seafloor stretching flat and gray under floodlights, the water pressing dark and infinite above",
        "Standing in the center of Stonehenge at dawn, the massive slabs arranged in their ancient circle, the open plain rolling away under a pale sky",
        "A narrow rope bridge over a jungle gorge, each plank swaying, the green canopy far below, the far side a dizzying distance across the gap",
        "An ice cave with smooth blue walls curving overhead in a frozen arch, daylight filtering through the ice and casting the space in aquamarine glow",
        "The view from a glass-bottomed boat over a coral reef, the ocean floor vivid and close, fish passing directly beneath your feet in clear water",
        "A vast underground lake in a limestone cavern, the still black water reflecting stalactites perfectly, the silence and darkness stretching beyond the reach of your headlamp",
        "A hedge maze seen from above, the green walls forming intricate corridors and dead ends, tiny figures wandering lost inside the geometric pattern",
        "Standing inside an empty Olympic stadium, the rows of seats rising steeply on all sides, the scale of the space dwarfing your body at its center",
        "A vertigo-inducing glass walkway jutting out from a cliff face over a river valley, the transparent floor revealing a two-thousand-foot drop straight down",
        "The cramped interior of a space capsule, every surface covered in switches and screens, the small porthole revealing the curved blue edge of Earth against black space",
        "A winding mountain path cut into a cliff, the trail barely two feet wide, the rock wall on one side and a sheer drop on the other",
        "A mangrove swamp where roots arch above and below the waterline, the tangled wooden lattice creating a maze of channels and shadowed passages",
        "The nave of a Gothic cathedral stretching forward in perfect symmetry, ribbed vaults drawing the eye along their arcs up to the distant rose window",
        "A salt flat stretching perfectly level to the horizon in every direction, the white surface reflecting the sky so completely that ground and air merge into one",
        "A treehouse perched high in an ancient oak, the wooden platform swaying gently, the ground far below glimpsed through gaps in the leaf canopy",
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
        "A grandmother teaching her granddaughter to braid hair, their hands moving together, voices soft with patience and gentle correction",
        "Making eye contact with someone across a crowded room and feeling an instant unspoken connection, the noise falling away for a moment",
        "A group of strangers huddled under an awning during a sudden downpour, laughing together at their shared misfortune, barriers dissolving in the rain",
        "Receiving a handwritten thank-you note from someone you helped years ago and had almost forgotten, their gratitude detailed and specific",
        "A couple dancing slowly in their kitchen to a song on the radio, bare feet on tile, no audience, completely absorbed in each other",
        "A child running toward you at the airport gate, arms open, screaming your name, crashing into your legs with the full force of their joy",
        "Sitting in a circle of friends around a campfire, everyone telling stories, the laughter rising and falling, faces glowing warm in the firelight",
        "The awkward silence when you wave back at someone who was waving at the person behind you, heat flooding your face instantly",
        "A nurse holding the hand of an elderly patient in a quiet hospital room, speaking gently, adjusting the blanket with practiced tenderness",
        "Two strangers helping each other push a car out of a snowbank, grunting and coordinating without introduction, shaking hands afterward and parting forever",
        "Your dog pressing its forehead against your knee when you are sitting alone and sad, as if it knows exactly what you need",
        "A teacher pulling a struggling student aside after class, kneeling to eye level, speaking quietly with encouragement no one else can hear",
        "The charged moment before a first kiss, faces inches apart, breath held, the entire world narrowing to the space between two people",
        "A father walking his daughter down the aisle, her hand on his arm, both of them trying not to cry, each step feeling like a lifetime",
        "Overhearing two teenagers defending you to their friends when they think you cannot hear, their loyalty fierce and unprompted and surprising",
        "A crowd of thousands singing the same anthem in unison at a concert, strangers linked arm in arm, voices merging into one enormous sound",
        "The relief of confessing something you have carried for years and watching the other person's face soften with understanding instead of judgment",
        "A barista remembering your name and your order on only your third visit, the small recognition making the whole coffee shop feel like home",
        "Watching a group of elderly women laughing so hard at lunch that one of them has tears streaming, their friendship decades deep and still growing",
        "A soldier returning home and kneeling to embrace a child who sprints across the tarmac, the reunion captured in a single desperate hug",
        "Sitting next to a stranger on a long train ride and slowly, over hours, exchanging life stories neither of you has told anyone else",
        "A mentor giving you honest criticism that stings at first but later proves to be the truest advice anyone ever offered you",
        "The moment a heated argument breaks when both people start laughing at how absurd it has become, the tension cracking like thin ice",
        "A volunteer handing a warm meal to a homeless person and staying to eat beside them on the bench, two strangers sharing food and quiet conversation",
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
        "Hearing a foreign accent shape familiar words in a new way, the vowels stretched, the consonants softened, the meaning identical but the music changed",
        "A toddler speaking in complete gibberish with perfect intonation and conviction, pausing for your response as if the meaning should be obvious",
        "Reading a passage of dense legal text and feeling each clause coil around the previous one, the meaning buried under layers of qualified precision",
        "The moment a new word in a language you are learning suddenly clicks and you understand it without translating, the meaning arriving directly",
        "Listening to a poet recite their work from memory, their voice adding pauses and emphasis that the printed page could never convey",
        "Writing a sentence and deleting it five times, each version almost right but not quite, the gap between thought and expression maddening and precise",
        "A bilingual child switching between languages mid-sentence without effort, pulling vocabulary from two systems as naturally as breathing",
        "Reading a novel where the narrator is unreliable and every sentence carries a second hidden meaning beneath the surface of the story",
        "The satisfaction of finding the exact right metaphor, the image locking into the idea with a click, illuminating what literal language could not reach",
        "An auctioneer chanting in rapid rhythmic patter, the numbers and bids flowing in a musical cadence that only the trained ear can parse",
        "Discovering that a word you have used your entire life actually means something slightly different from what you always assumed it meant",
        "A sign language interpreter at a concert translating lyrics into sweeping expressive gestures, their body becoming the music for the deaf audience",
        "Reading ancient graffiti on a Roman wall and realizing the joke is still funny two thousand years later, the humor bridging the centuries",
        "The way a single comma changes the meaning of a sentence entirely, the tiny mark redirecting the flow of logic like a switch on a railroad track",
        "A storyteller in a village square holding a crowd spellbound with nothing but words and gesture, the narrative woven so tightly no one moves",
        "The eerie feeling of reading your own handwriting from ten years ago and not recognizing the voice on the page as your own",
        "Learning that some languages have no word for a concept you consider fundamental, and wondering what else your own language hides from you",
        "A courtroom interpreter translating emotional testimony in real time, maintaining the witness's tone and grief while switching every word into another tongue",
        "The rhythmic chanting of a protest crowd, the slogan condensing a complex political position into six syllables that a thousand voices can shout in unison",
        "Reading a paragraph of perfectly grammatical prose that somehow communicates absolutely nothing, each word correct but the meaning evaporating between the sentences",
        "A crossword clue that uses a double meaning so cleverly that solving it produces a burst of delight at the wordplay involved",
        "The strangeness of hearing your own voice on a recording, the pitch and cadence unfamiliar, the disconnect between internal and external sound jarring",
        "A whispered secret passed ear to ear around a circle of children, the original message transforming with each retelling into something unrecognizable",
        "Reading a palindrome for the first time and marveling at how the letters mirror perfectly, the sentence reading identically forward and backward like a linguistic reflection",
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
        "Reaching forward to pick up a coffee mug from the table, wrapping fingers around the warm ceramic handle and lifting it smoothly to your lips",
        "Kneading bread dough on a floured surface, pushing with the heels of your hands, folding it over, pushing again, the rhythm meditative and steady",
        "Climbing a rock wall, fingers crimping into a narrow hold, toes pressing into a tiny ledge, core tight, reaching for the next grip above",
        "Skipping a flat stone across a glassy lake, the sidearm flick of the wrist, the stone kissing the surface three, four, five times before sinking",
        "A surgeon tying a suture with needle drivers, the precise rotation of wrist and fingers looping thread through tissue in a controlled knot",
        "Pedaling a bicycle uphill, standing on the pedals, legs burning, shifting weight side to side with each stroke to maintain momentum",
        "Shuffling a deck of cards with a riffle shuffle, thumbs bending the two halves, the cards interlacing with a satisfying zipper sound",
        "Chopping vegetables with a chef's knife, the rapid rocking motion of the blade, fingers curled under to guide each precise cut",
        "Diving off a high board, the push from the balls of your feet, the tuck and spin in midair, the stretch before entering the water cleanly",
        "Turning a pottery wheel, wet clay spinning under your palms, thumbs pressing the center open, the vessel rising between steady hands",
        "A gymnast executing a backflip on a balance beam, the explosive push, the midair rotation, the blind landing on four inches of padded wood",
        "Lacing up ice skates and gliding onto the rink, pushing off with alternating feet, the blades carving smooth arcs into the fresh ice surface",
        "Drawing a straight line freehand with a pen, the controlled tension from shoulder to fingertip, the ink flowing in a single unbroken stroke",
        "Swinging a golf club through a full arc, hips rotating ahead of shoulders, the clubface meeting the ball with a clean metallic click",
        "Folding an origami crane, each crease pressed sharp with a thumbnail, the paper transforming through precise folds from flat square to winged bird",
        "A drummer executing a rapid fill across the toms, sticks bouncing in alternating strokes, wrists loose, the pattern cascading from high to low",
        "Pulling back a bowstring to full draw, the fingers hooked around the string, shoulder blades squeezing together, the arrow steady against the cheek",
        "Braiding rope by weaving three strands over and under in alternating sequence, the tension kept even, the finished braid tight and uniform",
        "A fencer lunging forward, the explosive extension of the back leg, the sword arm driving the point forward, the whole body a straight line of force",
        "Juggling three balls in a steady cascade, each throw a controlled arc, hands catching and releasing in a rhythm that becomes automatic and flowing",
        "Pouring molten steel from a crucible, both hands gripping the long handle, tilting slowly, the glowing orange stream flowing precisely into the mold below",
        "Signing your name in cursive, the pen moving in a continuous fluid motion, each letter connected to the next without lifting from the paper",
        "A violinist's left hand vibrating on the string to produce vibrato, the fingertip oscillating rapidly, the forearm trembling with controlled tension and speed",
        "Cracking an egg one-handed against the rim of a bowl, the shell splitting cleanly, the yolk sliding out whole while your fingers separate the halves apart",
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
        "A double rainbow arcing across a dark storm sky, the colors impossibly vivid against the charcoal clouds, the second bow faint and reversed",
        "The Northern Lights rippling across the sky in curtains of green and violet, the light shifting and folding like luminous silk in slow motion",
        "A stained glass window lit from behind by afternoon sun, casting jeweled patches of ruby, emerald, and sapphire across a stone floor",
        "Ink dropped into a glass of water, the black tendrils unfurling in slow spiraling plumes, blooming outward in fractal fingers of dark diffusion",
        "A time-lapse of a city at night, headlights and taillights becoming rivers of white and red flowing through the grid of dark streets",
        "A prism splitting a beam of white sunlight into a perfect spectrum, each color bleeding into the next across the tabletop in a narrow rainbow",
        "The surface of a soap bubble catching the light, swirling with iridescent bands of pink and green and gold before popping into nothing",
        "A forest floor carpeted in autumn leaves, every shade from pale yellow through burnt orange to deep crimson, the color so dense it seems painted",
        "An enormous full moon rising orange over the ocean, its reflection wobbling on the water in a long bright column that reaches to the shore",
        "Frost patterns on a windowpane at dawn, intricate fern-like crystals spreading across the glass, each branch a perfect repeating fractal of ice",
        "A chandelier with a thousand crystal drops catching the light and scattering tiny rainbows across the ballroom ceiling in shifting constellations",
        "The stark silhouette of a lone tree on a hilltop against a blazing white winter sky, every branch and twig drawn in sharp black lines",
        "Bioluminescent plankton glowing electric blue in the surf at night, each breaking wave leaving a trail of ghostly light on the dark sand",
        "A photographer's darkroom under red safelight, the image slowly appearing on the paper in the developer tray, shadows forming first then details emerging",
        "Cherry blossoms falling in a gentle wind, thousands of pale pink petals drifting down through slanting golden light like soft fragrant snow",
        "A deep blue glacier calving into the sea, the freshly exposed ice face glowing an unearthly turquoise, sunlight penetrating the ancient compressed crystal",
        "The Milky Way seen from a mountaintop with zero light pollution, a dense band of stars and dust stretching across the entire dome of the sky",
        "A mosaic floor in an ancient Roman villa, thousands of tiny colored tiles forming a portrait of a woman whose painted eyes still stare back vividly",
        "Light refracting through a cut diamond, the facets throwing tiny spectral flashes across the jeweler's black velvet tray with every slight rotation",
        "A field of lavender in full bloom stretching to the horizon, the purple rows perfectly parallel, the color so saturated it looks almost artificial",
        "The shadow of a cloud passing over a green valley, the dark shape sliding across fields and forests, the sunlit land bright on either side",
        "An X-ray image of a hand held up to a light box, the bones glowing white, the joints and knuckles articulated in ghostly skeletal detail",
        "A long-exposure photograph of star trails, the points of light drawn into concentric circles around the pole star, the sky spinning in frozen motion",
        "Sunlight piercing through a dense forest canopy in distinct golden shafts, dust motes drifting lazily through each beam, the floor dappled in shifting coins of light",
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
        "The sharp sting of betrayal when you discover someone you trusted has been lying to you, the ground shifting under your assumptions",
        "A rush of pride watching someone you mentored succeed brilliantly, your chest swelling, your eyes stinging, the joy entirely selfless and pure",
        "The gnawing anxiety of waiting for medical test results, every minute stretching, your mind cycling between hope and catastrophe in an endless loop",
        "The giddy euphoria of falling in love, everything brighter and funnier, food tasting better, music sounding richer, the whole world rewritten by one person",
        "A pang of guilt that hits you at three in the morning, a careless word replaying on a loop, the wish to unsay it physically painful",
        "The quiet satisfaction of finishing a long difficult project, setting down the tools, stepping back, the fatigue and the pride mixing into warm stillness",
        "Overwhelming gratitude that makes your throat tight and your eyes burn, the realization that someone went far out of their way for you without being asked",
        "The sickening drop of shame when you realize everyone in the room heard what you thought was a private whisper, heat flooding your face and neck",
        "A sudden burst of courage that surprises even you, stepping forward when every instinct says to retreat, your voice steady despite the trembling inside",
        "The bittersweet ache of watching your child leave home for the first time, pride and loss tangled so tightly they feel like the same emotion",
        "A flash of irrational rage at an inanimate object that will not cooperate, the frustration boiling over into a fury you know is absurd but cannot stop",
        "The floating relief after a panic attack finally subsides, your body limp, your mind quiet, every muscle unclenching one by one like a fist slowly opening",
        "The warm glow of being forgiven for something you thought was unforgivable, the weight lifting, the relationship mended, the gratitude almost unbearable",
        "A wave of homesickness so strong it makes your stomach ache, triggered by a scent or a song, the longing for a place that no longer exists as you remember it",
        "The thrilling nervousness before stepping onto a stage, adrenaline sharpening everything, fear and excitement indistinguishable, your heart a drum in your ears",
        "Deep embarrassment at a public mistake, the desire to disappear, the awareness that everyone is watching, the seconds stretching into an eternity of exposure",
        "The peaceful acceptance that comes after a long struggle, not giving up but letting go, the fight draining away and leaving something still and clear behind",
        "A jealousy so sharp it surprises you, watching someone effortlessly receive what you worked years for, the unfairness burning bright and ugly in your chest",
        "The tender vulnerability of telling someone you love them for the first time, the words leaving your mouth and hanging in the air, unreturnable and exposed",
        "Awe that makes you feel tiny and vast at the same time, standing before something so beautiful or immense that your sense of self dissolves momentarily",
        "The dull ache of regret for a path not taken, a door closed years ago that you still sometimes stand before in your mind, wondering what lay behind it",
        "A fit of uncontrollable laughter that feeds on itself, tears streaming, stomach cramping, every attempt to stop only making it worse, the joy almost painful",
        "The strange numbness that follows a shock, the world continuing normally around you while you stand perfectly still, unable to feel anything at all yet",
        "The fierce protectiveness that flares when someone threatens a person you love, a primal heat rising in your chest, fists clenching, every sense sharpening instantly",
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
