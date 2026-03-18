from __future__ import annotations

from benchmarks.runner import MemoripyBenchmarkAdapter, run_benchmarks


def main() -> None:
    summary = run_benchmarks(MemoripyBenchmarkAdapter())
    print(f"adapter={summary['adapter']} score={summary['earned_score']:.1f}/{summary['max_score']:.1f}")
    for result in summary["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['scenario_id']}: {result['description']}")


if __name__ == "__main__":
    main()
