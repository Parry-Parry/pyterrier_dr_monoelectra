from typing import Optional

import numpy as np
import more_itertools
import torch
import pyterrier as pt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

try:
    from peft import PeftConfig, PeftModel

    _PEFT_AVAILABLE = True
except Exception:
    PeftConfig = None  # type: ignore
    PeftModel = None  # type: ignore
    _PEFT_AVAILABLE = False


DEFAULT_TEMPLATE = r"query: {query}, document: {text}"

class RankLLaMa(pt.Transformer):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 config, 
                 batch_size=16, 
                 text_field='text', 
                 verbose=True, 
                 device=None, 
                 template: str = None
                 ):
        self.model_name = model.model_name
        self.model = model.to(device).eval()
        self.tokeniser = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.template = template or DEFAULT_TEMPLATE

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        batch_size: int = 32,
        text_field: str = "text",
        verbose: bool = False,
        device: Optional[str] = None,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> "RankLLaMa":
        """
        Loads either a standard Transformers checkpoint or a PEFT adapter checkpoint.
        - If `model_name` points to a PEFT adapter repo/folder, it loads the base model
        specified by the adapter's PeftConfig and attaches the adapters.
        - Otherwise it loads the model directly from `model_name`.

        Arguments beyond the original:
        torch_dtype: set to None (default) to let HF choose; or pass torch.float16 / bfloat16 etc.
        trust_remote_code: forwarded to Transformers loaders (useful for custom/model repos).
        """
        dtype = torch_dtype if torch_dtype is not None else "auto"

        peft_cfg = None
        if _PEFT_AVAILABLE:
            try:
                peft_cfg = PeftConfig.from_pretrained(model_name)
            except Exception:
                peft_cfg = None

        if peft_cfg is not None:
            # Adapter-only repo/folder: load base, then attach adapters
            base_id = peft_cfg.base_model_name_or_path

            tokenizer = AutoTokenizer.from_pretrained(
                base_id, use_fast=True, trust_remote_code=trust_remote_code
            )
            config = AutoConfig.from_pretrained(
                base_id, trust_remote_code=trust_remote_code
            )
            base = AutoModelForSequenceClassification.from_pretrained(
                base_id, torch_dtype=dtype, trust_remote_code=trust_remote_code
            )

            model = PeftModel.from_pretrained(base, model_name)  # attach adapters
            model.eval()

            # Instantiate ranker
            res = cls(
                model,
                tokenizer,
                config,
                batch_size=batch_size,
                text_field=text_field,
                verbose=verbose,
                device=device,
            )
            # Useful bookkeeping
            res.model_name = model_name  # adapter id
            res.base_model_name = base_id  # backbone id
            res.peft_config = peft_cfg  # optional handle
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, trust_remote_code=trust_remote_code
            )
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
            )
            model.eval()

            res = cls(
                model,
                tokenizer,
                config,
                batch_size=batch_size,
                text_field=text_field,
                verbose=verbose,
                device=device,
            )
            res.model_name = model_name
            res.base_model_name = getattr(config, "_name_or_path", model_name)
            res.peft_config = None

        if device is not None:
            model.to(device)

        return res

    def transform(self, inp):
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='ELECTRA scoring')
        with torch.no_grad():
            for chunk in more_itertools.chunked(it, self.batch_size):
                sequences = [
                    self.template.format(query=query, text=document)
                    for query, document in chunk
                ]
                inps = self.tokeniser(sequences, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits[:, 0].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res
