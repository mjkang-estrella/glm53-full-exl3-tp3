"""Fail-closed startup hook for the isolated full-model TP3 candidate."""

import os
import traceback

if os.environ.get("VLLM_GLM53_EXL3_TP3_FULL_MODEL") == "1":
    try:
        from tp3_full_model_patch import apply_patches

        apply_patches()
        from sparse_mla_tp3_patch import apply_patches as apply_sparse_mla_tp3_patches

        apply_sparse_mla_tp3_patches()
        from tp3_logit_capture_patch import apply_patches as apply_logit_capture_patches

        apply_logit_capture_patches()

        if os.path.isfile("/capture/QUALITY_CONTROL.json"):
            from quality_experiment_patch import apply_patches as apply_quality_patches

            apply_quality_patches()

        if os.environ.get("VLLM_GLM53_META_PROFILE") == "1":
            from meta_model_profile_patch import apply_patches as apply_meta_profile_patches

            apply_meta_profile_patches()
    except BaseException:
        # Python normally reports and ignores sitecustomize failures. That is
        # unsafe here: an unpatched fallback can allocate the physical full
        # model. Terminate before application imports instead.
        traceback.print_exc()
        os._exit(70)
