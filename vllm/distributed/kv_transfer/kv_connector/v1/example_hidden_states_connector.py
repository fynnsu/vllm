# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from KV cache."""
    block_size = kv_cache.shape[1]
    return kv_cache[slot_mapping // block_size, slot_mapping % block_size][:num_tokens]


@dataclass
class PendingSave:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    block_ids: list[int]


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[PendingSave] = field(default_factory=list)


class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """
    Simple debug implementation of a HiddenStatesConnector.

    Simply extracts the hidden states from the kv cache and stores them to disk.
    Must be used in conjunction with the `extract_hidden_states` spec decoding method.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """
        Indicates whether this connector prefers KV blocks that hold KV data for all
        layers, which can speed up KV data transfers. Defaults to False.
        """
        # Must be False so that drafter kv cache isn't merged with verifier's
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []  # set by self.register_kv_caches
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        assert self._vllm_config.speculative_config is not None, (
            "ExampleHiddenStatesConnector only works when using "
            "'extract_hidden_states' speculative method"
        )
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        # Find the KV cache group index for hidden states
        self._hs_group_idx = 0
        if kv_cache_config is not None:
            for i, group in enumerate(kv_cache_config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, HiddenStateCacheSpec):
                    self._hs_group_idx = i
                    break

        # Scheduler-side state
        self._pending_saves: dict[str, PendingSave] = {}

        # Worker-side state (set by register_kv_caches)
        self._kv_cache: torch.Tensor | None = None

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass  # Empty implementation of abstract method

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # Empty implementation of abstract method

    def wait_for_save(self):
        pass  # Empty implementation of abstract method

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        # Filter layers to only include CacheOnlyAttentionLayers
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )
        self._kv_cache = kv_caches[self.cache_layers[0]]

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # Hidden states are already cached by CacheOnlyAttentionLayer during
        # forward. Extraction happens in get_finished once all tokens are done.
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        connector_metadata = self._get_connector_metadata()
        if not isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata):
            return None, None
        if not connector_metadata.pending_saves:
            return None, None

        assert self._kv_cache is not None

        finished_sending: set[str] = set()
        for pending in connector_metadata.pending_saves:
            slots: list[int] = []
            for block_id in pending.block_ids:
                for offset in range(self._block_size):
                    slots.append(block_id * self._block_size + offset)

            num_tokens = pending.token_ids.shape[0]
            slot_mapping = torch.tensor(
                slots[:num_tokens],
                dtype=torch.long,
                device=self._kv_cache.device,
            )

            hidden_states = extract_from_kv_cache(
                self._kv_cache, slot_mapping, num_tokens
            )
            tensors = {
                "hidden_states": hidden_states.detach().cpu(),
                "token_ids": pending.token_ids.detach().cpu(),
            }
            os.makedirs(self._storage_path, exist_ok=True)
            safetensors.torch.save_file(tensors, pending.filename)
            finished_sending.add(pending.req_id)

        return finished_sending or None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # This connector is store-only, so we don't need to load any tokens
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # Usually used to handle allocation of new blocks for requests that are loading
        # tokens from connector's external kv cache. We never load from external cache
        # so this is a no-op.
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ExampleHiddenStatesConnectorMetadata()

        # Transfer pending saves into metadata (scheduler → worker bridge)
        meta.pending_saves = list(self._pending_saves.values())
        self._pending_saves.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        Returns True to delay block freeing until get_finished extracts
        the hidden states from the KV cache.
        """
        req_id = request.request_id
        filename = os.path.join(self._storage_path, f"{req_id}.safetensors")
        token_ids = torch.tensor(request.prompt_token_ids or [])
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
        )
        return True, {"hidden_states_path": filename}

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[self._hs_group_idx])

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """

        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        # NHD means we have (num_tokens, num_heads)
        # HND means we have (num_heads, num_tokens)
        # For now, we only support NHD layout since this keeps the
        # hidden states for each token together in memory.
        # HND is primarily used when sharding heads across devices.
        return "NHD"
