"""Optional tree proxy scoring overlap with the target's remaining layers."""
import torch
from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close


class TreeProxySideStream:
    def __init__(self, device):
        self.stream=torch.cuda.Stream(device=device)
        self.done=None

    def launch(self, logits, batch, callback):
        # Exit logits and topology/q preparation belong to the calling stream.
        self.stream.wait_stream(torch.cuda.current_stream(logits.device))
        logits.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            event=duet_record('batch_proxy_side')
            callback(logits,batch)
            duet_close('batch_proxy_side',event)
            self.done=torch.cuda.Event()
            self.done.record(self.stream)

    def finish(self):
        # Enqueue after final target logits, before acceptance or graph-pool
        # reuse. This does not block the CPU or serialize the target post layers.
        if self.done is not None:
            torch.cuda.current_stream(self.stream.device).wait_event(self.done)
            self.done=None
