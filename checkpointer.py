import pathlib
import logging

logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info('Using checkpoint directory: {}'.format(self.checkpoint_dir))

    def get_latest_checkpoint_file(self):
        valid_files = sorted(self.checkpoint_dir.glob('*.valid'))
        if valid_files:
            latest_checkpoint = valid_files[-1].name.replace('.valid', '')
            checkpoint_file = self.checkpoint_dir / "{}.pth".format(latest_checkpoint)
            logger.info('Using checkpoint file: {}'.format(checkpoint_file))
            return checkpoint_file
        else:
            logger.info('Checkpoint directory is empty, no checkpoint to restore.')
            return None

    def get_checkpoint_file(self, epoch):
        return str(self.checkpoint_dir / "{:06}.pth".format(epoch))

    def checkpoint_saved(self, epoch):
        save_file = self.checkpoint_dir / "{:06}.valid".format(epoch)
        save_file.touch()
