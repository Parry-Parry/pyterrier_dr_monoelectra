from typing import Optional
from more_itertools import chunked
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_dr.hgf_models import HgfBiEncoder

try:
    from peft import PeftConfig, PeftModel

    _PEFT_AVAILABLE = True
except Exception:
    PeftConfig = None  # type: ignore
    PeftModel = None  # type: ignore
    _PEFT_AVAILABLE = False

DEFAULT_QUERY_TEMPLATE = r"query:  {query} {instruction}<\s>"
DEFAULT_DOCUMENT_TEMPLATE = r"passage:  {text}<\s>"


class Promptriever(HgfBiEncoder):
    def __init__(
        self,
        model,
        tokenizer,
        config,
        batch_size=32,
        text_field="text",
        verbose=False,
        device=None,
        query_template: str = None,
        document_template: str = None,
    ):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.config = config

        self.query_template = query_template or DEFAULT_QUERY_TEMPLATE
        self.document_template = document_template or DEFAULT_DOCUMENT_TEMPLATE

    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(
                    [*map(lambda x, y: self.query_template.format(query=x, instruction=y), chunk)],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state[:, -1]
                res = F.normalize(res, p=2, dim=0)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(
                    [*map(lambda x: self.document_template.format(text=x), chunk)],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state[:, -1]
                res = F.normalize(res, p=2, dim=0)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

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
    ) -> "Promptriever":
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
            base = AutoModel.from_pretrained(
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
            model = AutoModel.from_pretrained(
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

    def __repr__(self):
        if hasattr(self, "model_name"):
            return f"Promptriever({repr(self.model_name)})"
        return "Promptriever()"

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        with pta.validate.any(inp) as v:
            v.query_frame(extra_columns=['query', 'instruction'], mode='query_encoder')
            v.document_frame(extra_columns=[self.text_field], mode='doc_encoder')
            v.result_frame(extra_columns=["query", 'instruction', self.text_field], mode='scorer')
            v.columns(includes=['query', 'instruction', self.text_field], mode='scorer')
            v.columns(includes=['query_vec', self.text_field], mode='scorer')
            v.columns(includes=['query', 'instruction', 'doc_vec'], mode='scorer')
            v.columns(includes=['query_vec', 'doc_vec'], mode='scorer')
            v.columns(includes=['query', 'instruction'], mode='query_encoder')
            v.columns(includes=[self.text_field], mode='doc_encoder')

        if v.mode == 'scorer':
            return self.scorer()(inp)
        elif v.mode == 'query_encoder':
            return self.query_encoder()(inp)
        elif v.mode == 'doc_encoder':
            return self.doc_encoder()(inp)


class BiQueryEncoder(pt.Transformer):
    def __init__(self, bi_encoder_model: Promptriever, verbose=None, batch_size=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_encoder_model.encode_queries(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=['query'])
        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        enc = self.encode(it)
        return inp.assign(query_vec=[enc[i] for i in inv])

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.query_encoder()'
