"""Camera-specific YOLO11n train-only run, with no validation-based selection."""
import argparse
import csv
import hashlib
import inspect
import json
import platform
import subprocess
import time

from common import ROOT, DATA, EXP, dump
import torch
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import init_seeds, strip_optimizer


class TrainOnlyTrainer(DetectionTrainer):
    """Prevent even the final-epoch/final-checkpoint validation on aliased train data."""

    def validate(self):
        LOGGER.info('TRAIN-ONLY: skipped framework validation; YAML val aliases training images.')
        return {}, None

    def save_metrics(self, metrics):
        # Framework initializes val metric placeholders even with val=False. Do not publish those zeros.
        super().save_metrics({k: v for k, v in metrics.items() if k.startswith(('train/', 'lr/'))})

    def save_model(self):
        self.metrics = {}
        self.fitness = None
        self.best_fitness = -1.0  # internal sentinel differs from None: suppress unvalidated best.pt copies
        return super().save_model()

    def final_eval(self):
        LOGGER.info('TRAIN-ONLY: final evaluation disabled; exporting final EMA last.pt.')
        if self.last.exists():
            strip_optimizer(self.last)
        assert not self.best.exists(), 'An independently selected best.pt must not be implied.'

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        with self.csv.open() as f:
            records = list(csv.DictReader(f))
        keys = [k for k in records[0] if k.startswith('train/')]
        fig, axes = plt.subplots(1, len(keys), figsize=(5*len(keys), 4), squeeze=False)
        for axis, key in zip(axes[0], keys):
            axis.plot([int(r['epoch']) for r in records], [float(r[key]) for r in records])
            axis.set(xlabel='Epoch', ylabel=key, title=key)
            axis.grid(alpha=0.25)
        fig.suptitle('Training losses only — no independent validation/test')
        fig.tight_layout()
        fig.savefig(self.save_dir / 'training_loss.png', dpi=150)
        plt.close(fig)


def params(camera, device, epochs, name='train'):
    return dict(model=str(ROOT / 'weights/yolo11n.pt'),
                data=str(EXP / 'working_data' / camera / 'dataset.yaml'),
                project=str(EXP / camera), name=name, device=device,
                epochs=epochs, batch=16, imgsz=256, workers=2,
                seed=42, deterministic=True, val=False, patience=0,
                optimizer='AdamW', lr0=0.001, lrf=0.01, cos_lr=True, weight_decay=0.0005,
                warmup_epochs=3.0, hsv_h=0., hsv_s=0., hsv_v=0., bgr=0.,
                degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                flipud=0., fliplr=0., mosaic=0., mixup=0., copy_paste=0., cutmix=0.,
                close_mosaic=0, augmentations=[], multi_scale=0., rect=False,
                cache=False, amp=True, plots=True, save=True, save_period=50, verbose=True)


def smoke(camera, parameters):
    torch.set_num_threads(2)
    init_seeds(42, deterministic=True)
    source = YOLO(str(ROOT / 'weights/yolo11n.pt'))
    original = source.model.model[0].conv.weight.detach().clone()
    trainer = TrainOnlyTrainer(overrides=dict(parameters, name='batch_smoke', batch=4, workers=0))
    trainer.setup_model()
    first = trainer.model.model[0].conv.weight.detach().cpu()
    assert first.shape[1] == 1 and torch.equal(first, original.cpu().sum(1, keepdim=True))
    trainer.model.to(trainer.device)
    trainer.model.args = trainer.args
    loader = trainer.get_dataloader(trainer.data['train'], batch_size=4, rank=-1, mode='train')
    batch = next(iter(loader))
    shape, unique = list(batch['img'].shape), torch.unique(batch['img']).tolist()
    batch = trainer.preprocess_batch(batch)
    trainer.model.train()
    loss, parts = trainer.model(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    gradient = trainer.model.model[0].conv.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    inference = YOLO('yolo11n.yaml')
    inference.model = trainer.model.eval()
    inference.overrides = dict(source.overrides, task='detect')
    observed = []
    handle = inference.model.model[0].register_forward_pre_hook(lambda mod, args: observed.append(list(args[0].shape)))
    image = next((EXP / 'working_data' / camera / 'images/train').glob('*.png'))
    inference.predict(str(image), imgsz=256, device=parameters['device'], half=False, rect=False, verbose=False, save=False)
    handle.remove()
    assert observed and all(s[1] == 1 for s in observed)
    report = dict(status='passed', pretrained_input_shape=list(original.shape), single_channel_first_layer=list(first.shape),
                  adaptation='sum of RGB input-channel filters, checked by exact tensor equality',
                  loaded_batch_shape=shape, loaded_uint8_values=unique,
                  finite_loss=loss.detach().cpu().tolist(), finite_backward=True,
                  prediction_input_shapes=observed,
                  preprocessing='All spatial/color augmentations disabled; source binary occupancy normalized by /255; ordinary floating point weights.')
    dump(EXP / camera / 'smoke.json', report)
    (EXP / camera / 'official_loader_source.txt').write_text(inspect.getsource(BaseModel.load))
    del trainer, inference, source, loader, batch, loss
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', required=True, choices=['F3E', 'F59'])
    parser.add_argument('--device', required=True)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    out = EXP / args.camera
    out.mkdir(exist_ok=True)
    assert not (out / 'train').exists(), 'Preserve completed experiments; choose a new experiment directory.'
    audit = json.loads((EXP / 'audit/dataset_audit.json').read_text())
    assert audit['status'] == 'passed' and not audit['independent_validation_or_test']
    settings.update({k: False for k in ('sync', 'wandb', 'mlflow', 'comet', 'clearml', 'neptune', 'tensorboard') if k in settings})
    parameters = params(args.camera, args.device, args.epochs)
    dump(out / 'parameters.json', parameters)
    dump(out / 'environment.json', dict(camera=args.camera, python=platform.python_version(), executable=__import__('sys').executable,
                                       pytorch=torch.__version__, cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                       ultralytics=ultralytics.__version__, device=args.device,
                                       gpu=torch.cuda.get_device_name(int(args.device)), nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
                                       seed=42, pretrained_sha256=hashlib.sha256((ROOT/'weights/yolo11n.pt').read_bytes()).hexdigest(),
                                       environment='Existing project venv inherits host CUDA 12.4 PyTorch; no dependency upgrade.',
                                       validation='disabled, including final framework validation', selected_checkpoint='last.pt only'))
    smoke(args.camera, parameters)
    # Exercise one real optimizer epoch before committing to the full fixed-duration run.
    short = YOLO(str(ROOT / 'weights/yolo11n.pt'))
    short.train(trainer=TrainOnlyTrainer, **dict(parameters, name='epoch_smoke', epochs=1, save_period=-1))
    assert (out / 'epoch_smoke/weights/last.pt').is_file()
    del short
    torch.cuda.empty_cache()
    model = YOLO(str(ROOT / 'weights/yolo11n.pt'))
    def verify_channels(trainer):
        assert trainer.model.model[0].conv.in_channels == 1 and trainer.data['channels'] == 1
        dump(out / 'training_channel_check.json', dict(first_layer_channels=1, dataset_channels=1))
    model.add_callback('on_train_start', verify_channels)
    started = time.monotonic()
    model.train(trainer=TrainOnlyTrainer, **parameters)
    assert model.trainer.epoch + 1 == args.epochs
    assert model.trainer.last.is_file() and not model.trainer.best.exists()
    with (out / 'train/results.csv').open() as f:
        records = list(csv.DictReader(f))
    assert not any(k.startswith(('val/', 'metrics/')) for k in records[0])
    dump(out / 'training_completion.json', dict(status='completed', camera=args.camera,
                                               epochs=len(records), elapsed_seconds=time.monotonic()-started,
                                               checkpoint=str(model.trainer.last),
                                               checkpoint_sha256=hashlib.sha256(model.trainer.last.read_bytes()).hexdigest(),
                                               independent_validation=False, independent_test=False,
                                               checkpoint_selection='final epoch EMA, no validation-based selection',
                                               first_epoch_losses={k:v for k,v in records[0].items() if k.startswith('train/')},
                                               last_epoch_losses={k:v for k,v in records[-1].items() if k.startswith('train/')}))
    print('TRAINING COMPLETED', args.camera, model.trainer.last, flush=True)


if __name__ == '__main__':
    main()
