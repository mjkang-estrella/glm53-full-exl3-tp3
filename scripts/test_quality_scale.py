"""CPU regression tests for the opt-in scale path; no weights/GPU required."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"runtime"))
from quality_experiment_patch import scaled_runtime, validate_control, apply_stored_scales


class ScaleTest(unittest.TestCase):
    def test_fp16_scale_change_and_exact_restoration(self):
        base=torch.tensor([.123,-.875,2.5],dtype=torch.float16)
        sv2,sv3=base.clone(),base.clone()
        store=SimpleNamespace(layer=33,mixed_groups={
            2:{"experts":[7],"packs":[{"down":SimpleNamespace(svh=sv2)}]},
            3:{"experts":[8],"packs":[{"down":SimpleNamespace(svh=sv3)}]}})
        ptr=sv2.data_ptr()
        apply_stored_scales(store,{"id":"first","gains":{"2":1.025,"3":1.0}})
        self.assertEqual(sv2.data_ptr(),ptr)
        self.assertTrue(torch.equal(sv2,(base.float()*1.025).half()))
        self.assertTrue(torch.equal(sv3,base))
        apply_stored_scales(store,{"id":"second","expert_gains":{"33":{"7":.95}}})
        self.assertTrue(torch.equal(sv2,(base.float()*.95).half()))
        apply_stored_scales(store,{"id":"restore","gains":{"2":1.0,"3":1.0}})
        self.assertTrue(torch.equal(sv2,base))
        self.assertTrue(torch.equal(sv3,base))

    def test_group_scale_is_applied_only_to_requested_bits(self):
        # Independent mathematical output, including negative and zero values.
        value=torch.tensor([[2.,-4.,0.],[.25,3.,-8.]])
        runtime=SimpleNamespace(apply_exl3_fused_moe=lambda *a,**kw:value.clone())
        proxy=scaled_runtime(runtime,{"2":.975,"3":1.0})
        k2=proxy.apply_exl3_fused_moe(None,None,None,SimpleNamespace(_exl3_bits=2))
        k3=proxy.apply_exl3_fused_moe(None,None,None,SimpleNamespace(_exl3_bits=3))
        torch.testing.assert_close(k2,value*.975,rtol=0,atol=0)
        self.assertTrue(torch.equal(k3,value))
        self.assertTrue(torch.equal(value,torch.tensor([[2.,-4.,0.],[.25,3.,-8.]])))

    def test_gain_bounds_fail_closed(self):
        for value in (float("nan"),float("inf"),0.,1.2):
            with self.assertRaises(ValueError):
                validate_control({"schema":"glm53.quality-control.v1","id":"case",
                                  "gains":{"2":value,"3":1.0}})

    def test_unbounded_capture_fails(self):
        with self.assertRaises(ValueError):
            validate_control({"schema":"glm53.quality-control.v1","id":"case",
                              "capture":{"id":"trace","layers":[33],"rows":1000000}})


if __name__=="__main__":
    unittest.main()
