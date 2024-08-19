from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from os.path import join
from pathlib import Path


class BadmintonPreprocessor:
    def __init__(self, args: Namespace):
        self.args = args

        self.train_path = join(args.root, "train.csv")
        self.test_path = join(args.root, "test.csv")
        self.n_players = args.n_agents

        self.player_loc_cols = [
            "player_A_x",
            "player_A_y",
            "player_B_x",
            "player_B_y",
            "player_C_x",
            "player_C_y",
            "player_D_x",
            "player_D_y",
        ]

        self.shot_type_categories = ['發短球','放小球','挑球','切球','殺球','防守回抽','擋小球','推撲球','平球','防守回挑','後場抽平球','小平球','長球','發長球','過度切球','勾球']
        self.shot_type_enc = OneHotEncoder(categories=[self.shot_type_categories])
        # self.shot_type_enc = OneHotEncoder()
        self.hit_player_enc = OneHotEncoder()
    
    def normalize(data, data_max, data_min):
        return (data - data_min) * 2 / (data_max - data_min) - 1

    def _preprocess_single(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        displacements = df[self.player_loc_cols].values  # 位移
        nor_max = displacements.max()
        nor_min = displacements.min()
        normalize = lambda data, data_max, data_min: (data - data_min) * 2 / (data_max - data_min) - 1
        displacements = normalize(displacements, nor_max, nor_min)
        print("data_max: ", nor_max)
        print("data_min: ",nor_min)
        # velocity
        df[
            ["delta_A_x", "delta_A_y", "delta_B_x", "delta_B_y", "delta_C_x", "delta_C_y", "delta_D_x", "delta_D_y"]
        ] = df.groupby("rally_id")[
            [
                "player_A_x",
                "player_A_y",
                "player_B_x",
                "player_B_y",
                "player_C_x",
                "player_C_y",
                "player_D_x",
                "player_D_y",
            ]
        ].diff()

        df["delta_time"] = df.groupby("rally_id")["end_frame_num"].shift(1) - df.groupby("rally_id")[
            "start_frame_num"
        ].shift(1)
        df.loc[df["delta_time"] == 0, "delta_time"] = 1
        # df["delta_time"] = df.groupby("rally_id")["start_frame_num"].shift(-1) - df["start_frame_num"]
        df["velocity_A_x"] = df["delta_A_x"] / df["delta_time"]
        df["velocity_A_y"] = df["delta_A_y"] / df["delta_time"]
        df["velocity_B_x"] = df["delta_B_x"] / df["delta_time"]
        df["velocity_B_y"] = df["delta_B_y"] / df["delta_time"]
        df["velocity_C_x"] = df["delta_C_x"] / df["delta_time"]
        df["velocity_C_y"] = df["delta_C_y"] / df["delta_time"]
        df["velocity_D_x"] = df["delta_D_x"] / df["delta_time"]
        df["velocity_D_y"] = df["delta_D_y"] / df["delta_time"]

        df.loc[
            df.groupby("rally_id").head(1).index,
            [
                "velocity_A_x",
                "velocity_A_y",
                "velocity_B_x",
                "velocity_B_y",
                "velocity_C_x",
                "velocity_C_y",
                "velocity_D_x",
                "velocity_D_y",
            ],
        ] = 0
        velocity = df[
            [
                "velocity_A_x",
                "velocity_A_y",
                "velocity_B_x",
                "velocity_B_y",
                "velocity_C_x",
                "velocity_C_y",
                "velocity_D_x",
                "velocity_D_y",
            ]
        ].values

   
        # self.loc_feat = data_utils.normalize(
        #         self.loc_feat, self.loc_max, self.loc_min)
        # self.vel_feat = data_utils.normalize(
        #         self.vel_feat, self.vel_max, self.vel_min)

        # shot_types
        shot_types = self.shot_type_enc.fit_transform(df[["ball_type"]].values).toarray()
        
        # hit_players
        hit_player = self.hit_player_enc.fit_transform(df[["player"]].values).toarray()

        unique_rallies = np.unique(df["rally_id"])
        new_locs = []
        new_vels = []
        new_shot = []
        new_hit = []
        for rally_id in unique_rallies:
            rally_data = df[df["rally_id"] == rally_id]
            n_timestamps = len(rally_data)

            indices = df["rally_id"] == rally_id

            locs_rally = displacements[indices].reshape(n_timestamps, self.n_players, 2)
            vels_rally = velocity[indices].reshape(n_timestamps, self.n_players, 2)
            shot_rally = np.repeat(shot_types[indices][:, np.newaxis, :], 4, axis=1)
            shot_rally = np.expand_dims(shot_rally, axis=0).reshape(n_timestamps, self.n_players, shot_rally.shape[2])

            hit_rally = np.repeat(hit_player[indices][:, np.newaxis, :], 4, axis=1)
            hit_rally = np.expand_dims(hit_rally, axis=0).reshape(n_timestamps, self.n_players, hit_rally.shape[2])
            new_locs.append(locs_rally)
            new_vels.append(vels_rally)
            new_shot.append(shot_rally)
            new_hit.append(hit_rally)

        displacements = new_locs
        velocity = new_vels
        shot_types = new_shot
        hit_player = new_hit

        max_len = max(len(x) for x in displacements)
        locs_pad = []
        for seq in displacements:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            locs_pad.append(batch_pad)
        displacements = np.stack(locs_pad, axis=0)

        vels_pad = []
        for seq in velocity:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            vels_pad.append(batch_pad)
        velocity = np.stack(vels_pad, axis=0)

        shot_pad = []
        for seq in shot_types:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            shot_pad.append(batch_pad)
        shot_types = np.stack(shot_pad, axis=0)

        hit_pad = []
        for seq in hit_player:
            pad_len = max_len - len(seq)
            batch_pad = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
            hit_pad.append(batch_pad)
        hit_player = np.stack(hit_pad, axis=0)

        displacements = np.transpose(displacements, (1, 0, 2, 3))
        velocity = np.transpose(velocity, (1, 0, 2, 3))
        shot_types = np.transpose(shot_types, (1, 0, 2, 3))
        hit_player = np.transpose(hit_player, (1, 0, 2, 3))
        seq_len, _, _, features = displacements.shape
        displacements = displacements.reshape((seq_len, -1, features))
        velocity = velocity.reshape((seq_len, -1, features))
        shot_types = shot_types.reshape((seq_len, -1, shot_types.shape[-1]))
        hit_player = hit_player.reshape((seq_len, -1, hit_player.shape[-1]))
        return displacements, velocity, shot_types, hit_player

    def _save(
        self,
        locs: np.ndarray,
        vels: np.array,
        shot_types: np.ndarray,
        hit_players: np.ndarray,
        output_dir: Path,
    ) -> None:
        np.save(output_dir.joinpath("displacements.npy"), locs)
        np.save(output_dir.joinpath("velocity.npy"), vels)
        np.save(output_dir.joinpath("goals.npy"), shot_types)
        np.save(output_dir.joinpath("hit.npy"), hit_players)

    def preprocess(self):
        # Train
        train_df = pd.read_csv(self.train_path)
        train_output_dir = Path(self.args.root).joinpath("train")
        train_output_dir.mkdir(parents=True, exist_ok=True)
        locs, vels, shot_types, hit_players = self._preprocess_single(train_df)
        self._save(locs, vels, shot_types, hit_players, train_output_dir)

        # Test
        test_df = pd.read_csv(self.test_path)
        test_output_dir = Path(self.args.root).joinpath("test")
        test_output_dir.mkdir(parents=True, exist_ok=True)
        locs, vels, shot_types, hit_players = self._preprocess_single(test_df)
        self._save(locs, vels, shot_types, hit_players, test_output_dir)


def preprocess_cli(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("-n", "--n_agents", type=int, default=4, help="number of agents")  # fmt: skip
    parser.add_argument("-val", "--val_size", type=float, default=0.2, help="percentage of validation set")  # fmt: skip
    parser.add_argument("-r", "--root", type=str, default=join("data", "badminton", "doubles"), help="dataset directory")  # fmt: skip
    return parser


def preprocess_main(args: Namespace):
    preprocessor = BadmintonPreprocessor(args)
    preprocessor.preprocess()
