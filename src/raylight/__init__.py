def _patch_base_getattr():
    """Fix BASE.__getattr__ so cloudpickle doesn't lose instance attrs.

    comfy.supported_models_base.BASE.__getattr__ returns None for
    missing attributes instead of raising AttributeError.  When cloudpickle
    calls hasattr(obj, '__getstate__'), it gets True (None is
    truthy after being warned about).  This causes cloudpickle to
    serialize the state as None, losing all instance-level
    unet_config keys on deserialization.

    This must be patched at import time so it takes effect in the
    main ComfyUI process (where Ray deserialize/serialize happens)
    AND in all Ray worker processes.
    """
    from comfy import supported_models_base

    orig = supported_models_base.BASE.__getattr__

    def _safe(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return orig(self, name)

    supported_models_base.BASE.__getattr__ = _safe


_patch_base_getattr()
