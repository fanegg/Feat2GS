import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "..", "vggt")))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import einops as E

class VGGT_1B(torch.nn.Module):
    def __init__(
        self,
        device="cuda",
        feat_from="encoder",
    ):
        super().__init__()

        self.device = device
        self.feat_from = feat_from

        # get model
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def forward(self, image_names):
        images = load_and_preprocess_images(image_names).to(self.device)

        h, w = images.shape[-2:]
        h, w = h // self.model.aggregator.patch_size, w // self.model.aggregator.patch_size

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):        
                if "decoder" in self.feat_from:
                    aggregated_tokens_list, patch_start_idx = self.model.aggregator(images.unsqueeze(0))
                else:
                    outfeat = self.model.aggregator(images.unsqueeze(0), ret_encfeat=True)

        if self.feat_from == "decoder":
            outfeat = aggregated_tokens_list[-1][:, :, patch_start_idx:][0]
        elif self.feat_from == "decoder_global":
            outfeat = aggregated_tokens_list[-1][:, :, patch_start_idx:, -1024:][0]
        elif self.feat_from == "decoder_frame":
            outfeat = aggregated_tokens_list[-1][:, :, patch_start_idx:, :1024][0]

        feat = E.rearrange(outfeat, "b (h w) c -> b h w c", h=h, w=w) 
        return feat
