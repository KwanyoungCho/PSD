"""Optional tree proxy head/scoring overlap with remaining target layers."""
import torch
from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close


class TreeProxySideStream:
    def __init__(self, device):
        self.stream=torch.cuda.Stream(device=device)
        self.done=torch.cuda.Event()
        self.pending=False

    def launch(self, logits, batch, callback):
        # Fresh normalized hidden/logits and topology/q preparation belong to
        # the calling stream. Never pass residual storage mutated by graph-post.
        self.stream.wait_stream(torch.cuda.current_stream(logits.device))
        logits.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            event=duet_record('batch_proxy_side')
            callback(logits,batch)
            duet_close('batch_proxy_side',event)
            self.done.record(self.stream)
            self.pending=True

    def finish(self):
        # Enqueue after final target logits, before acceptance or graph-pool
        # reuse. This does not block the CPU or serialize the target post layers.
        if self.pending:
            torch.cuda.current_stream(self.stream.device).wait_event(self.done)
            self.pending=False
