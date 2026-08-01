"""Command-line GPU diagnostic for ZeSolver."""

from __future__ import annotations

import argparse
import json
import sys

from .gpu_support import DistributionKind, GpuRuntimeContext, build_gpu_provisioning_plan, probe_gpu_capability


def _context_from_args(args: argparse.Namespace) -> GpuRuntimeContext:
    return GpuRuntimeContext(
        distribution_kind=DistributionKind(args.distribution_kind),
        allow_environment_mutation=bool(args.allow_environment_mutation),
        python_executable=sys.executable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose optional NVIDIA/CuPy acceleration for ZeSolver.")
    parser.add_argument("--json", action="store_true", help="emit stable JSON")
    parser.add_argument("--self-test", action="store_true", help="run the CUDA allocation self-test when CuPy is installed")
    parser.add_argument("--show-install-plan", action="store_true", help="include the provisioning plan")
    parser.add_argument(
        "--distribution-kind",
        choices=[item.value for item in DistributionKind],
        default=DistributionKind.UNKNOWN.value,
    )
    parser.add_argument("--allow-environment-mutation", action="store_true")
    args = parser.parse_args(argv)
    context = _context_from_args(args)
    report = probe_gpu_capability(context, run_self_test=True if args.self_test else True)
    plan = build_gpu_provisioning_plan(report, context) if args.show_install_plan else None
    if args.json:
        payload = report.to_dict()
        payload["provisioning_available"] = bool(plan and plan.command)
        if plan is not None:
            payload["provisioning_plan"] = plan.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print("ZeSolver GPU diagnostic")
    print(f"platform={report.platform} arch={report.architecture}")
    print(f"backend={report.effective_backend.value} reason={report.reason_code.value}")
    print(report.human_message)
    if plan is not None:
        print(f"provisioning={plan.status.value}")
        if plan.command:
            print("command=" + " ".join(plan.command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
