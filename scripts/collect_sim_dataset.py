import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# Dataset collection is an offline rollout job.  TorchDynamo/Inductor can spend a
# very long time compiling TensorDict-heavy policy code here, so keep it off by
# default unless the caller explicitly overrides these environment variables.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import hydra
import torch
import wandb
from omegaconf import OmegaConf
from tensordict import MemoryMappedTensor
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from tqdm import tqdm

# Add project root to path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from active_adaptation.utils.motion import MotionData
from isaaclab.app import AppLauncher
from scripts.utils.helpers import make_env_policy


def _infer_mempath_from_train_sh() -> str | None:
    train_sh = Path(__file__).resolve().parents[1] / "train.sh"
    if not train_sh.exists():
        return None

    pattern = re.compile(r"^\s*(?:export\s+)?MEMPATH=(.+?)\s*$")
    for line in train_sh.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip().strip("\"'")
    return None


def _configure_mempath(mempath: str | None):
    if mempath:
        os.environ["MEMPATH"] = mempath
        return

    if os.environ.get("MEMPATH"):
        return

    inferred = _infer_mempath_from_train_sh()
    if inferred:
        os.environ["MEMPATH"] = inferred
        print(f"[collect] Using MEMPATH from train.sh: {inferred}", flush=True)


def _dataset_root() -> str:
    path_root = os.environ.get("MEMPATH")
    if path_root is None:
        path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    return path_root


def _validate_dataset_paths(cfg):
    dataset_cfg = cfg.task.command.get("dataset", None)
    if dataset_cfg is None:
        return

    missing = []
    path_root = _dataset_root()
    for mem_path in dataset_cfg.get("mem_paths", []):
        path = mem_path if os.path.isabs(mem_path) else os.path.join(path_root, mem_path)
        if not os.path.isdir(path):
            missing.append(path)

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Motion dataset path not found. Set MEMPATH or pass --mempath.\n"
            f"Missing:\n  {joined}"
        )


def _download_run_cfg_and_checkpoint(run_path: str, iterations: int | None):
    api = wandb.Api()
    run = api.run(run_path)
    print(f"Loading run {run.name}")

    root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
    os.makedirs(root, exist_ok=True)

    checkpoints = []
    for file in run.files():
        if "checkpoint" in file.name:
            checkpoints.append(file)
        elif file.name in ("cfg.yaml", "files/cfg.yaml", "config.yaml"):
            file.download(root, replace=True)

    if not checkpoints:
        raise RuntimeError(f"No checkpoint files found in run {run_path}")

    if iterations is None:
        def sort_by_iter(file):
            number_str = file.name[:-3].split("_")[-1]
            return 100000 if number_str == "final" else int(number_str)

        checkpoints.sort(key=sort_by_iter)
        checkpoint = checkpoints[-1]
    else:
        matches = [f for f in checkpoints if f.name == f"checkpoint_{iterations}.pt"]
        if not matches:
            raise RuntimeError(f"checkpoint_{iterations}.pt not found in run {run_path}")
        checkpoint = matches[0]

    print(f"Downloading {checkpoint.name}")
    checkpoint.download(root, replace=True)

    try:
        cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
    except FileNotFoundError:
        cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))

    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = os.path.join(root, checkpoint.name)
    cfg.vecnorm = "eval"
    return run, cfg


def _resolve_output_path(output: str, run_name: str) -> str:
    output = output.format(run_name=run_name)
    if os.path.isabs(output):
        return output

    return os.path.join(_dataset_root(), output)


def _zero_init_noise(cfg):
    init_noise = cfg.task.command.get("init_noise", None)
    if init_noise is not None:
        for key in init_noise.keys():
            init_noise[key] = 0.0


def _prepare_cfg(cfg, args):
    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="collect_sim_dataset", version_base=None):
            task_cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        cfg.task.reward = task_cfg.task.reward
        cfg.task.termination = task_cfg.task.termination
        cfg.task.observation = task_cfg.task.observation
        cfg.task.action = task_cfg.task.action
        cfg.task.randomization = task_cfg.task.randomization
        cfg.task.robot = task_cfg.task.robot
        cfg.task.flags = task_cfg.task.flags
        if args.terrain:
            cfg.task.terrain = task_cfg.task.terrain
        if args.command:
            cfg.task.command = task_cfg.task.command

    cfg.app.headless = args.headless
    cfg.app.enable_cameras = False
    cfg.eval_render = False
    cfg.task.num_envs = args.num_envs
    if "dataset" in cfg.task.command:
        cfg.task.command.dataset.sample_once = True

    if not args.keep_randomization:
        cfg.task.randomization = {}
    if not args.keep_init_noise:
        _zero_init_noise(cfg)

    # We control traversal manually and never call step_and_maybe_reset.
    cfg.task.termination = {}
    return cfg


def _get_base_env(env):
    base = env
    while hasattr(base, "base_env"):
        base = base.base_env
    return base


def _make_name_mapping(motion_names, asset_names, device):
    motion_idx = []
    asset_idx = []
    for i, name in enumerate(motion_names):
        if name in asset_names:
            motion_idx.append(i)
            asset_idx.append(asset_names.index(name))
    return (
        torch.tensor(motion_idx, dtype=torch.long, device=device),
        torch.tensor(asset_idx, dtype=torch.long, device=device),
    )


@torch.no_grad()
def _load_motion_batch(command, ds_index: int, motion_ids: torch.Tensor):
    dataset = command.dataset
    dataset.sample_once = True
    dataset._refreshing = False
    ds = dataset.datasets[ds_index]
    motion_ids_cpu = motion_ids.to(device=dataset.ds_device, dtype=torch.long)
    env_count = motion_ids.numel()

    local_starts = ds.starts[motion_ids_cpu]
    local_ends = ds.ends[motion_ids_cpu] - 1
    steps = torch.arange(dataset.max_step_size, device=dataset.ds_device, dtype=torch.long)
    local_idx = (local_starts.unsqueeze(1) + steps).clamp(max=local_ends.unsqueeze(1))

    dataset._buf_A[:env_count, :dataset.max_step_size] = dataset._to_float(
        ds.data[local_idx].to(dataset.device),
        dtype=torch.float16,
    )
    dataset._len_A[:env_count] = ds.lengths[motion_ids_cpu].clamp_max(dataset.max_step_size).to(dataset.device)

    if env_count < dataset.env_size:
        dataset._len_A[env_count:] = 1
        dataset._buf_A[env_count:] = dataset._buf_A[:1]

    return dataset._len_A[:env_count].clone()


@torch.no_grad()
def _collect_current_frame(base_env, command, body_map, joint_map):
    from active_adaptation.utils.math import quat_apply_inverse

    asset = command.asset
    dataset = command.dataset
    device = base_env.device
    env_count = base_env.num_envs

    # Start from the reference frame so unsupported/missing names keep valid values.
    ref = dataset.get_slice(None, command.t, 1)
    frame = {field: getattr(ref, field)[:, 0].clone() for field in ref.__dataclass_fields__}

    origins = base_env.scene.env_origins
    root_pos = asset.data.root_pos_w - origins
    root_quat = asset.data.root_quat_w

    frame["root_pos_w"][:] = root_pos
    frame["root_quat_w"][:] = root_quat
    frame["root_lin_vel_w"][:] = asset.data.root_lin_vel_w
    frame["root_ang_vel_w"][:] = asset.data.root_ang_vel_w

    joint_motion_idx, joint_asset_idx = joint_map
    if joint_motion_idx.numel():
        frame["joint_pos"][:, joint_motion_idx] = asset.data.joint_pos[:, joint_asset_idx]
        frame["joint_vel"][:, joint_motion_idx] = asset.data.joint_vel[:, joint_asset_idx]

    body_motion_idx, body_asset_idx = body_map
    if body_motion_idx.numel():
        frame["body_pos_w"][:, body_motion_idx] = asset.data.body_pos_w[:, body_asset_idx] - origins.unsqueeze(1)
        frame["body_quat_w"][:, body_motion_idx] = asset.data.body_quat_w[:, body_asset_idx]

    frame["body_pos_b"][:] = quat_apply_inverse(
        root_quat.unsqueeze(1),
        frame["body_pos_w"] - root_pos.unsqueeze(1),
    )

    frame["motion_id"][:] = 0
    frame["step"][:] = command.t
    return frame


def _copy_info_to_json(info: dict[str, torch.Tensor], num_motions: int):
    json_info = {}
    for key, value in info.items():
        value = value.detach().cpu()
        json_info[key] = value.reshape(num_motions, -1).tolist()
    return json_info


def _allocate_output(total: int, body_count: int, joint_count: int, float_dtype, int_dtype):
    return {
        "motion_id": MemoryMappedTensor.empty(total, dtype=int_dtype),
        "step": MemoryMappedTensor.empty(total, dtype=int_dtype),
        "root_pos_w": MemoryMappedTensor.empty(total, 3, dtype=float_dtype),
        "root_quat_w": MemoryMappedTensor.empty(total, 4, dtype=float_dtype),
        "root_lin_vel_w": MemoryMappedTensor.empty(total, 3, dtype=float_dtype),
        "root_ang_vel_w": MemoryMappedTensor.empty(total, 3, dtype=float_dtype),
        "joint_pos": MemoryMappedTensor.empty(total, joint_count, dtype=float_dtype),
        "joint_vel": MemoryMappedTensor.empty(total, joint_count, dtype=float_dtype),
        "body_pos_w": MemoryMappedTensor.empty(total, body_count, 3, dtype=float_dtype),
        "body_pos_b": MemoryMappedTensor.empty(total, body_count, 3, dtype=float_dtype),
        "body_quat_w": MemoryMappedTensor.empty(total, body_count, 4, dtype=float_dtype),
    }


def _write_frame(mm, frame, env_indices, motion_ids, step_idx, out_starts, float_dtype, int_dtype):
    rows = out_starts[motion_ids] + step_idx
    mm["motion_id"][rows] = motion_ids.to(dtype=int_dtype).cpu()
    mm["step"][rows] = torch.full((len(motion_ids),), step_idx, dtype=int_dtype)
    for key, value in frame.items():
        if key in ("motion_id", "step"):
            continue
        mm[key][rows] = value[env_indices].detach().to(dtype=float_dtype).cpu()


@torch.no_grad()
def collect_dataset(cfg, output_path: str, args):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    print("[collect] Launching Isaac app...", flush=True)
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env = None
    try:
        print(f"[collect] Creating env and loading policy with num_envs={cfg.task.num_envs}...", flush=True)
        env, agent, _, _ = make_env_policy(cfg)
        print("[collect] Env and policy created.", flush=True)
        base_env = _get_base_env(env)
        command = base_env.command_manager
        source_ds = command.dataset.datasets[args.dataset_index]

        num_motions = source_ds.num_motions
        if args.max_motions is not None:
            num_motions = min(num_motions, args.max_motions)
        lengths = source_ds.lengths[:num_motions].clamp_max(command.dataset.max_step_size).cpu()
        starts = torch.zeros(num_motions, dtype=torch.long)
        starts[1:] = torch.cumsum(lengths[:-1].to(torch.long), dim=0)
        ends = starts + lengths.to(torch.long)
        total = int(ends[-1].item()) if num_motions else 0

        if total == 0:
            raise RuntimeError("Source dataset is empty.")
        print(f"[collect] Source dataset: {num_motions} motions, {total} frames.", flush=True)

        if os.path.exists(output_path):
            if not args.overwrite:
                raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(output_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        mm = _allocate_output(
            total,
            len(source_ds.body_names),
            len(source_ds.joint_names),
            getattr(torch, args.storage_float_dtype),
            getattr(torch, args.storage_int_dtype),
        )

        body_map = _make_name_mapping(source_ds.body_names, command.asset.body_names, base_env.device)
        joint_map = _make_name_mapping(source_ds.joint_names, command.asset.joint_names, base_env.device)
        missing_bodies = sorted(set(source_ds.body_names) - set(command.asset.body_names))
        missing_joints = sorted(set(source_ds.joint_names) - set(command.asset.joint_names))
        if missing_bodies:
            print(f"[Warn] Bodies not found on robot, keeping reference values: {missing_bodies}")
        if missing_joints:
            print(f"[Warn] Joints not found on robot, keeping reference values: {missing_joints}")

        policy = agent.get_rollout_policy("eval")
        env.eval()
        env.set_seed(cfg.seed)

        motion_id_batches = torch.arange(num_motions).split(args.num_envs)
        batch_wall_steps = [
            int(lengths[batch].max().item())
            for batch in motion_id_batches
        ]
        print(f"[collect] Writing to {output_path}", flush=True)
        print(
            f"[collect] Collecting {len(motion_id_batches)} batches, "
            f"{sum(batch_wall_steps)} batch-steps.",
            flush=True,
        )
        with set_exploration_type(ExplorationType.MODE):
            with tqdm(total=sum(batch_wall_steps), desc="Collecting batch-steps") as pbar:
                for batch_i, batch in enumerate(motion_id_batches):
                    batch = batch.to(base_env.device)
                    batch_size = batch.numel()
                    batch_lengths = _load_motion_batch(command, args.dataset_index, batch)
                    max_len = int(batch_lengths.max().item())
                    pbar.set_postfix(batch=f"{batch_i + 1}/{len(motion_id_batches)}", envs=batch_size)

                    reset_td = env.reset()
                    active_envs = torch.arange(batch_size, device=base_env.device)

                    frame = _collect_current_frame(base_env, command, body_map, joint_map)
                    _write_frame(
                        mm,
                        frame,
                        active_envs,
                        batch.cpu(),
                        0,
                        starts,
                        getattr(torch, args.storage_float_dtype),
                        getattr(torch, args.storage_int_dtype),
                    )
                    pbar.update(1)

                    td = reset_td
                    for step_idx in range(1, max_len):
                        td = policy(td)
                        stepped = env.step(td)
                        td = step_mdp(stepped)

                        still_active = (batch_lengths > step_idx).nonzero(as_tuple=False).squeeze(-1)
                        if still_active.numel() > 0:
                            frame = _collect_current_frame(base_env, command, body_map, joint_map)
                            _write_frame(
                                mm,
                                frame,
                                still_active,
                                batch[still_active].cpu(),
                                step_idx,
                                starts,
                                getattr(torch, args.storage_float_dtype),
                                getattr(torch, args.storage_int_dtype),
                            )
                        pbar.update(1)

        data = MotionData(**mm, batch_size=[total])
        data.memmap(output_path)
        meta = {
            "body_names": source_ds.body_names,
            "joint_names": source_ds.joint_names,
            "starts": starts.tolist(),
            "ends": ends.tolist(),
            "info": _copy_info_to_json(source_ds.info, num_motions),
        }
        with open(os.path.join(output_path, "meta_motion.json"), "w") as f:
            json.dump(meta, f)

        print(f"Saved simulated dataset to {output_path}", flush=True)
        print(f"Motions: {num_motions}, frames: {total}", flush=True)
        if args.force_exit:
            # Isaac/Kit teardown can hang on some headless servers after the data is already written.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", required=True, type=str)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-i", "--iterations", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default="sim_rollout_{run_name}")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--mempath", type=str, default=None)
    parser.add_argument("--terrain", action="store_true", default=False)
    parser.add_argument("--command", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--keep-randomization", action="store_true", default=False)
    parser.add_argument("--keep-init-noise", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--force-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--storage-float-dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--storage-int-dtype", type=str, default="int32", choices=["int32", "int64"])
    args = parser.parse_args()

    _configure_mempath(args.mempath)
    run, cfg = _download_run_cfg_and_checkpoint(args.run_path, args.iterations)
    cfg = _prepare_cfg(cfg, args)
    _validate_dataset_paths(cfg)
    output_path = _resolve_output_path(args.output, run.name)
    collect_dataset(cfg, output_path, args)


if __name__ == "__main__":
    main()
