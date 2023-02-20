import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import signal  
import torch

# def preemption_manager(trainer: 'pytorch_lightning.Trainer', cfg: Optional[Union[DictConfig, Dict]] = None) -> Optional[Path]:
#     """
#     preemption_manager is a helper function that manages and controls the preemption feature
#     Args:
#         trainer (pytorch_lightning.Trainer): The lightning trainer.
#         cfg (DictConfig, dict): Can have the following keys:
#             - 
#     """

class PreemptionCallback(Callback):

    def __init__(self, device, sig=signal.SIGTERM):
        self.sig = sig
        self.device = device

    @property
    def interrupted(self):
        interrupted = torch.tensor(self._interrupted).int().to(self.device)
        #dist_utils.broadcast(interrupted, 0)
        interrupted = bool(interrupted.item())
        return interrupted

    def on_train_start(self, trainer, pl_module):
        self._interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def master_handler(signum, frame):
            self.release()
            self._interrupted = True

        def ignoring_handler(signum, frame):
            self.release()

        #rank = dist_utils.get_rank()
        #if rank == 0:
        signal.signal(self.sig, master_handler)
        #else:
        #    signal.signal(self.sig, ignoring_handler)

        return self
    
    def on_train_end(self, trainer, pl_module):
        self.release()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        # check if the job was preempted
        # NOTE: "timeout_handler.interrupted" is a property which triggers a
        # distributed broadcast of "_interrupted" flag from rank 0 to all other
        # ranks, to avoid performance overheads it's best to store the result in
        # a regular local variable
        interrupted = self.interrupted

        if interrupted:
            print("---interrupted---")
        else:
            print("--not interrupted--")

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True    
