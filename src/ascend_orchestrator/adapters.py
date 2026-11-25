"""Version-pinned offline msAgent and controlled CANNBot adapter examples."""

import os

from .models import ContractError, NotFoundError, digest, evidence_document, new_id, now


DEFAULT_CAPABILITIES = [
    {
        "capability_id": "msagent.offline.profiler-analysis",
        "provider": "msagent",
        "version": "prototype-offline-v1",
        "kind": "OFFLINE_ANALYSIS",
        "inputs": ["profiling_data", "question"],
        "outputs": ["analysis_report", "structured_findings"],
        "resource_requirement": "NONE",
        "supports_cancel": False,
        "supports_resume": False,
        "timeout_seconds": 30,
        "executes_npu": False,
    },
    {
        "capability_id": "cannbot.controlled.patch-example",
        "provider": "cannbot",
        "version": "prototype-skill-v1",
        "kind": "CONTROLLED_SKILL",
        "supported_dsl": ["AscendC"],
        "supported_phases": ["IMPLEMENT", "OPERATOR_OPTIMIZE"],
        "inputs": ["task_description", "workspace"],
        "outputs": ["patch", "notes", "required_next_tests"],
        "resource_requirement": "NONE",
        "writes_workspace": True,
        "executes_npu": False,
        "supports_cancel": False,
        "supports_resume": False,
        "timeout_seconds": 30,
    },
]


class AdapterRegistry:
    def __init__(self, control):
        self.control = control
        for capability in DEFAULT_CAPABILITIES:
            if not control.store.get("capabilities", capability["capability_id"]):
                control.store.put("capabilities", capability)

    def list(self):
        return self.control.store.all("capabilities")

    def invoke(self, capability_id, inputs, request_id=None):
        payload = {"capability_id": capability_id, "inputs": inputs}
        def write():
            capability = self.control.store.get("capabilities", capability_id)
            if not capability:
                raise NotFoundError("capability is not registered: %s" % capability_id)
            if inputs.get("simulate_error"):
                result = {
                    "invocation_id": new_id("invoke"), "status": "FAILED",
                    "capability_id": capability_id, "version": capability["version"],
                    "error": "adapter failure is explicit and is not an empty experimental result",
                    "limitations": ["No domain conclusion available"], "created_at": now(),
                }
                self.control.store.event("CAPABILITY_FAILED", capability_id, after="FAILED")
                return result
            if capability["provider"] == "msagent":
                result = self._msagent(capability, inputs)
            else:
                result = self._cannbot(capability, inputs)
            result.update({"invocation_id": new_id("invoke"), "status": "SUCCEEDED",
                           "capability_id": capability_id, "version": capability["version"],
                           "created_at": now()})
            if inputs.get("task_id") and inputs.get("step_id"):
                evidence = evidence_document(
                    inputs["task_id"], inputs["step_id"],
                    {"kind": capability["kind"], "name": capability_id, "version": capability["version"]},
                    correctness="NOT_APPLICABLE", candidate_id=inputs.get("candidate_id"),
                    limitations=result["limitations"], artifact_digest=result.get("patch_digest"),
                )
                self.control.store.put("evidence", evidence)
                result["evidence_ids"] = [evidence["evidence_id"]]
            self.control.store.event("CAPABILITY_INVOKED", capability_id, after=result["status"],
                                     data={"version": capability["version"], "provider": capability["provider"]})
            return result
        return self.control._write(request_id, "capability_invoke", payload, write)

    @staticmethod
    def _msagent(capability, inputs):
        profile = inputs.get("profiling_data")
        if profile is None:
            raise ContractError("profiling_data is required")
        return {
            "analysis_report": "Offline prototype analysis; hypotheses require controlled experiments.",
            "structured_findings": [{"kind": "HYPOTHESIS", "summary": "Inspect the supplied hotspot evidence"}],
            "findings": [],
            "proposed_hypotheses": ["Validate the dominant kernel with a frozen comparison protocol"],
            "required_tests": ["controlled-profiling"],
            "limitations": ["Offline adapter example", "No optimization candidate is accepted automatically"],
        }

    def _cannbot(self, capability, inputs):
        workspace = inputs.get("workspace")
        if not workspace:
            raise ContractError("an isolated workspace is required")
        if "\x00" in workspace:
            raise ContractError("invalid workspace")
        root = os.path.abspath(workspace)
        os.makedirs(root, exist_ok=True)
        patch_path = os.path.join(root, "prototype.patch")
        patch = "--- a/kernel.cpp\n+++ b/kernel.cpp\n@@ prototype-controlled-skill @@\n"
        with open(patch_path, "w", encoding="utf-8") as handle:
            handle.write(patch)
        return {
            "patch": patch_path,
            "patch_digest": digest(patch),
            "notes": "Controlled write confined to the supplied workspace",
            "required_next_tests": ["build", "single-card-correctness"],
            "limitations": ["Patch is an adapter contract example, not a functional AscendC optimization"],
        }
