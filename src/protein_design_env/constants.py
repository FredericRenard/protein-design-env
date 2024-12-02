from protein_design_env.amino_acids import AminoAcids

CHARGE_PENALTY = -10
REWARD_PER_MOTIF = 1

MIN_SEQUENCE_LENGTH = 15
MAX_SEQUENCE_LENGTH = 25
DEFAULT_SEQUENCE_LENGTH = 15

DEFAULT_MOTIF = [AminoAcids.ARGININE.value, AminoAcids.LYSINE.value, AminoAcids.ARGININE.value]

MIN_MOTIF_LENGTH = 2
MAX_MOTIF_LENGTH = 4

NUM_AMINO_ACIDS = len(AminoAcids)
AMINO_ACIDS_VALUES = [aa.value for aa in AminoAcids]
