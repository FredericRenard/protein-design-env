import numpy as np
import pytest
from protein_design_env.amino_acids import AminoAcids
from protein_design_env.constants import (
    CHARGE_PENALTY,
    DEFAULT_MOTIF,
    DEFAULT_SEQUENCE_LENGTH,
    MAX_MOTIF_LENGTH,
    MAX_SEQUENCE_LENGTH,
    MIN_MOTIF_LENGTH,
    MIN_SEQUENCE_LENGTH,
    NUM_AMINO_ACIDS,
    REWARD_PER_MOTIF,
)
from protein_design_env.environment import Environment


class TestEnvironment:
    def setup_method(self) -> None:
        self.env = Environment()

    def test_env_step(self) -> None:
        """Mock an episode of length 4."""
        self.env.reset()
        self.env.sequence_length = 3

        # 1 is ALANINE, the sequence is of neutral charge.
        obs, reward, terminated, truncated, info = self.env.step(1)

        assert self.env.state == [1]
        expected_obs = np.hstack(
            [
                [1],
                [0 for _ in range(MAX_SEQUENCE_LENGTH - 1)],
                1,
                self.env.motif,
                self.env.sequence_length,
                0,
            ]
        )
        np.testing.assert_array_equal(obs, expected_obs)
        assert reward == 0
        assert not terminated
        assert not truncated

        # 2 is ARGININE of charge +1, the sequence is positive.
        obs, reward, terminated, truncated, info = self.env.step(2)

        assert self.env.state == [1, 2]
        expected_obs = np.hstack(
            [
                [1, 2],
                [0 for _ in range(MAX_SEQUENCE_LENGTH - 2)],
                2,
                self.env.motif,
                self.env.sequence_length,
                +1,
            ]
        )
        np.testing.assert_array_equal(obs, expected_obs)
        assert reward == CHARGE_PENALTY
        assert not terminated
        assert not truncated

        # 5 is CYSTEINE of charge 0, the sequence is positive.
        obs, reward, terminated, truncated, info = self.env.step(5)

        assert self.env.state == [1, 2, 5]
        expected_obs = np.hstack(
            [
                [1, 2, 5],
                [0 for _ in range(MAX_SEQUENCE_LENGTH - 3)],
                3,
                self.env.motif,
                self.env.sequence_length,
                +1,
            ]
        )
        np.testing.assert_array_equal(obs, expected_obs)
        assert reward == CHARGE_PENALTY
        assert terminated
        assert truncated

    @pytest.mark.parametrize("invalid_action", [0, -5, 22])
    def test_env_step_raises_if_action_is_invalid(self, invalid_action: int) -> None:
        self.env.reset()
        with pytest.raises(ValueError):
            self.env.step(invalid_action)

    @pytest.mark.parametrize("change_motif_at_each_episode", [True, False])
    @pytest.mark.parametrize("change_sequence_length_at_each_episode", [True, False])
    def test_env_reset(
        self, change_motif_at_each_episode: bool, change_sequence_length_at_each_episode: bool
    ) -> None:
        self.env.change_motif_at_each_episode = change_motif_at_each_episode
        self.env.change_sequence_length_at_each_episode = change_sequence_length_at_each_episode

        obs, info = self.env.reset()

        assert self.env.state == []
        self._assert_motif_is_correct(self.env.motif)
        assert MIN_SEQUENCE_LENGTH <= self.env.sequence_length <= MAX_SEQUENCE_LENGTH

        extpected_obs = np.hstack(
            [
                [0 for _ in range(MAX_SEQUENCE_LENGTH)],
                0,
                self.env.motif,
                self.env.sequence_length,
                0,
            ]
        )
        np.testing.assert_array_equal(obs, extpected_obs)

    def test_get_charge(self) -> None:
        self.env.state = []
        assert self.env._get_charge() == 0

        self.env.state.append(AminoAcids.GLUTAMIC_ACID)
        assert self.env._get_charge() == -1

        self.env.state.append(AminoAcids.LYSINE)
        assert self.env._get_charge() == 0

    def test_get_reward(self) -> None:
        # The default motif has three positive charges.
        self.env.state = DEFAULT_MOTIF.copy()
        assert self.env._get_reward() == REWARD_PER_MOTIF + CHARGE_PENALTY

        # Make the sequence is now neutral in charge.
        self.env.state.extend(
            [
                AminoAcids.ASPARTIC_ACID.value,
                AminoAcids.ASPARTIC_ACID.value,
                AminoAcids.ASPARTIC_ACID.value,
            ]
        )

        assert self.env._get_reward() == REWARD_PER_MOTIF

        # Add another motif while staying neutral in charge.
        self.env.state.extend(self.env.state)
        assert self.env._get_reward() == 2 * REWARD_PER_MOTIF

    def test_generate_sequence(self) -> None:
        assert self.env._generate_sequence_length() == DEFAULT_SEQUENCE_LENGTH

        self.env.change_sequence_length_at_each_episode = True

        for _ in range(20):
            assert (
                MIN_SEQUENCE_LENGTH <= self.env._generate_sequence_length() <= MAX_SEQUENCE_LENGTH
            )

    def test_generate_motif(self) -> None:
        np.testing.assert_array_equal(self.env._generate_motif(), DEFAULT_MOTIF)

        self.env.change_motif_at_each_episode = True

        for _ in range(20):
            motif = self.env._generate_motif()
            self._assert_motif_is_correct(motif)

    def test_pad_state(self) -> None:
        self.env.state = []
        np.testing.assert_array_equal(
            self.env._pad_state(), [0 for _ in range(MAX_SEQUENCE_LENGTH)]
        )

        self.env.state = [5, 7, 1, 8]
        np.testing.assert_array_equal(
            self.env._pad_state(), [5, 7, 1, 8] + [0 for _ in range(MAX_SEQUENCE_LENGTH - 4)]
        )

    def test_get_observation(self) -> None:
        self.env.state = [5, 7, 1, 8]
        charge = self.env._get_charge()

        np.testing.assert_array_equal(
            self.env._get_observation(),
            np.hstack([self.env._pad_state(), 4, self.env.motif, self.env.sequence_length, charge]),
        )

    def _assert_motif_is_correct(self, motif: list[int]) -> None:
        assert np.all(np.asarray(motif) >= 1)
        assert np.all(np.asarray(motif) <= NUM_AMINO_ACIDS)
        assert MIN_MOTIF_LENGTH <= len(motif) <= MAX_MOTIF_LENGTH
